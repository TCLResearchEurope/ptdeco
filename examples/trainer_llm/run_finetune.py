import gzip
import json
import logging
import pathlib
import shutil
import time
from typing import Any, Optional

import peft
import ptdeco
import torch
import transformers  # type: ignore

import builder
import configurator
import datasets_hf
import metrics
import utils

PPL_N_SAMPLES = 1000
LOADER_SEED = 42


logger = logging.getLogger(__name__)


class CustomTrainer(transformers.Trainer):
    def __init__(
        self,
        *args: Any,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.model.config.pad_token_id
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.train_loader

    def get_eval_dataloader(
        self, eval_dataset: Optional[torch.utils.data.Dataset] = None
    ) -> torch.utils.data.DataLoader:
        return self.test_loader


def make_dataloader_perplexity(
    config: configurator.FinetuneConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.utils.data.DataLoader, int]:

    perplexity_ds = datasets_hf.get_dataset(config.perplexity_data_name)
    perplexity_n = len(perplexity_ds)
    msg = f"Created perplexity dataset {config.perplexity_data_name}, "
    msg += f"{perplexity_n} examples"
    logger.info(msg)

    perplexity_dl = datasets_hf.prepare_dataloader_v1(
        dataset=perplexity_ds,
        tokenizer=tokenizer,
        max_seqlen=config.perplexity_data_max_length,
        batch_size=config.perplexity_data_batch_size,
        separator=config.perplexity_data_separator,
        nsamples=PPL_N_SAMPLES,
        varied_seqlen=False,
        seed=LOADER_SEED,
    )

    return perplexity_dl, perplexity_n


def make_dataloader_train(
    config: configurator.FinetuneConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.utils.data.DataLoader, int]:

    train_ds = datasets_hf.get_dataset(config.train_data_name)
    train_n = len(train_ds)

    logger.info(f"Created train dataset {config.train_data_name}, {train_n} examples")

    train_dl = datasets_hf.prepare_dataloader_v1(
        dataset=train_ds,
        tokenizer=tokenizer,
        max_seqlen=config.train_data_max_length,
        batch_size=config.train_data_batch_size,
        separator=config.train_data_separator,
        nsamples=config.train_data_n_samples,
        varied_seqlen=False,
        seed=LOADER_SEED,
    )

    return train_dl, train_n


def make_dataloader_test(
    config: configurator.FinetuneConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.utils.data.DataLoader, int]:

    test_ds = datasets_hf.get_dataset(config.test_data_name)
    test_n = len(test_ds)

    logger.info(f"Created test dataset {config.test_data_name}, {test_n} examples")

    test_dl = datasets_hf.prepare_dataloader_v1(
        dataset=test_ds,
        tokenizer=tokenizer,
        max_seqlen=config.test_data_max_length,
        batch_size=config.test_data_batch_size,
        separator=config.test_data_separator,
        nsamples=config.test_data_n_samples,
        varied_seqlen=False,
        seed=LOADER_SEED,
    )

    return test_dl, test_n


def make_optimizer_and_scheduler(
    model: torch.nn.Module, n_train: int, config: configurator.FinetuneConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    eff_bs = config.train_data_batch_size * config.gradient_accumulation_steps
    num_training_steps = config.num_train_epochs * ((n_train - 1) // eff_bs + 1)

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": num_training_steps,
    }
    if config.lr_scheduler_type == "cosine_with_warmup":
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            **kwargs_lr_scheduler
        )
    elif config.lr_scheduler_type == "linear_with_warmup":
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            **kwargs_lr_scheduler
        )
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


def make_lora_config(
    model: torch.nn.Module, config: configurator.FinetuneConfig
) -> Optional[peft.LoraConfig]:

    with open(config.decompose_config, "rt") as f:
        deco_config = json.load(f)

    decomposed_module_names = list(deco_config.keys())
    linear_decomposed_module_names = [f"{e}.0" for e in decomposed_module_names] + [
        f"{e}.1" for e in decomposed_module_names
    ]
    non_decomposed_module_names = [
        name
        for name, module in model.named_modules()
        if name not in linear_decomposed_module_names
        and isinstance(module, torch.nn.Linear)
        and "lm_head" not in name
    ]

    min_rank_to_finetune = 32
    rank_pattern = {}
    alpha_pattern = {}
    target_modules = []
    for decomposed_module_name in decomposed_module_names:
        rank = deco_config[decomposed_module_name]["modules"]["0"]["out_features"]
        if rank >= min_rank_to_finetune:
            lora_rank = max(rank // 32, 8)
            first_linear_name = f"{decomposed_module_name}.0"
            second_linear_name = f"{decomposed_module_name}.1"
            rank_pattern[first_linear_name] = lora_rank
            rank_pattern[second_linear_name] = lora_rank
            alpha_pattern[first_linear_name] = lora_rank // 2
            alpha_pattern[second_linear_name] = lora_rank // 2
            logger.info(
                f"Setting lora rank to: {lora_rank} for {decomposed_module_name}"
            )
            target_modules.extend([first_linear_name, second_linear_name])
        else:
            logger.info(
                f"Skipping fine-tuning {decomposed_module_name} - rank is to low {rank}"
            )

    if not config.finetune_only_decomposed:
        target_modules += non_decomposed_module_names
    if len(target_modules) > 0:
        return peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=peft.TaskType.CAUSAL_LM,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            target_modules=target_modules,
        )
    else:
        return None


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    # 1. SETUP

    start_eval = time.perf_counter()
    transformers.utils.logging.disable_progress_bar()
    config = configurator.FinetuneConfig(**config_raw)
    dtype = utils.conv_str_to_dtype(config.decomposed_model_dtype)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. CREATE MODEL AND LORA CONFIG

    egc = config.decomposed_model_enable_gradient_checkpointing
    model, tokenizer = builder.make_model_and_tokenizer(
        model_name=config.decomposed_model_name,
        model_revision=config.decomposed_model_revision,
        model_custom_builder_path=config.decomposed_model_custom_builder_path,
        model_custom_builder_config=config.decomposed_model_custom_builder_config,
        enable_gradient_checkpointing=egc,
        dtype=dtype,
        log_linears=False,
    )
    model.to(device)
    params_orig = metrics.get_params(model) / 1.0e6
    gflops_orig = metrics.get_giga_flops(model, tensor_size=(1, 512))

    builder.apply_decompose_config_and_state_dict_in_place(
        model=model,
        decompose_config_path=config.decompose_config,
        state_dict_path=config.decompose_state_dict,
        device=device,
        dtype=dtype,
        log_linears=True,
    )

    params_final = metrics.get_params(model) / 1.0e6
    gflops_final = metrics.get_giga_flops(model, tensor_size=(1, 512))

    lora_config = make_lora_config(model, config)
    if lora_config is None:
        logger.warning("No modules to finetune found, exiting")
        return

    # 3. INITIAL MODEL EVALUATION

    perplexity_dl, _ = make_dataloader_perplexity(config, tokenizer)
    train_dl, train_n = make_dataloader_train(config, tokenizer)
    test_dl, _ = make_dataloader_test(config, tokenizer)

    with torch.no_grad():
        perplexity_orig = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    logger.info(f"{perplexity_orig=}")

    time_lm_eval_initial = -1.0
    if config.lm_eval_initial and config.lm_eval_tasks:
        start_eval = time.perf_counter()

        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            tasks=config.lm_eval_tasks,
        )
        logger.info("\n" + lm_eval_results_str)
        lm_eval_path = output_path / "lm_eval_initial.json.gz"
        with gzip.open(lm_eval_path, "wt") as f:
            json.dump(lm_eval_results, f)
        logger.info(f"Initial lm_eval results saved to {lm_eval_path}")
        time_lm_eval_initial = time.perf_counter() - start_eval
        logger.info(f"Initial lm_eval took {time_lm_eval_initial:.2f} s")

    # 4. ACTUAL FINETUNING
    start_finetune = time.perf_counter()

    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer, lr_scheduler = make_optimizer_and_scheduler(model, train_n, config)

    training_args = transformers.TrainingArguments(
        output_dir=output_path,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_data_batch_size,
        per_device_eval_batch_size=config.test_data_batch_size,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        disable_tqdm=True,
        load_best_model_at_end=True,
        eval_steps=config.eval_steps,
        evaluation_strategy=config.eval_strategy,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config.decomposed_model_enable_gradient_checkpointing,
        report_to=["tensorboard"],
    )

    es = transformers.EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience
    )
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_dl,
        test_loader=test_dl,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[es],
    )

    # required to enable gradient_checkpointing
    model.enable_input_require_grads()

    ptdeco.utils.free_gpu_reserved_memory()

    model.train()
    trainer.train()
    ptdeco.utils.free_gpu_reserved_memory()
    time_finetune = time.perf_counter() - start_finetune
    logger.info(f"Fine-tuning took {time_finetune:.2f} s")

    # 5. SAVE TRAINING OUTPUTS

    model = model.merge_and_unload()
    ptdeco.utils.free_gpu_reserved_memory()
    shutil.copy2(config.decompose_config, output_path / "decompose_config.json")
    torch.save(model.state_dict(), output_path / "decompose_state_dict.pt")

    # 6. FINAL MODEL EVALUATION

    with torch.no_grad():
        perplexity_final = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    logger.info(f"{perplexity_final=}")

    if config.lm_eval_tasks:
        start_eval = time.perf_counter()

        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            tasks=config.lm_eval_tasks,
        )
        logger.info("\n" + lm_eval_results_str)
        lm_eval_path = output_path / "lm_eval_final.json.gz"
        with gzip.open(lm_eval_path, "wt") as f:
            json.dump(lm_eval_results, f)
        logger.info(f"Final lm_eval results saved to {lm_eval_path}")
        time_lm_eval_final = time.perf_counter() - start_eval
        logger.info(f"Final lm_eval took {time_lm_eval_initial:.2f} s")

    # 7. SAVE SUMMARY
    device_str = str(device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(device)

    summary = {
        "mparams_orig": params_orig,
        "gflops_orig": gflops_orig,
        "perplexity_initial": perplexity_orig,
        "perplexity_final": perplexity_final,
        "mparams_final": params_final,
        "gflops_final": gflops_final,
        "time_finetune": time_finetune,
        "time_lm_eval_initial": time_lm_eval_initial,
        "time_lm_eval_final": time_lm_eval_final,
        "device": device_str,
    }
    with open(output_path / "summary.json", "wt") as f:
        json.dump(summary, f)
