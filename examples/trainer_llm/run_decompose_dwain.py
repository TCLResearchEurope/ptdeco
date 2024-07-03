import collections.abc
import json
import logging
import pathlib
import time
from typing import Any

import ptdeco
import torch
import transformers  # type: ignore

import builder
import configurator
import datasets_hf
import dwain_wrapper_module
import metrics
import utils

PPL_N_SAMPLES = 1000
LOADER_SEED = 42

logger = logging.getLogger(__name__)


def make_inifinte_iterator(
    dl: collections.abc.Iterable[Any],
) -> collections.abc.Generator[Any, None, None]:
    while True:
        for x in dl:
            yield x


def make_dataloaders(
    config: configurator.DecomposeDWAINConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[
    torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    torch.utils.data.DataLoader[dict[str, torch.Tensor]],
]:
    decomposition_ds = datasets_hf.get_dataset(config.decomposition_data_name)

    logger.info(
        f"Created decomposition dataset {config.decomposition_data_name}, "
        f"{len(decomposition_ds)} examples"
    )

    decomposition_dl = datasets_hf.prepare_dataloader_v2(
        dataset=decomposition_ds,
        tokenizer=tokenizer,
        max_seqlen=config.decomposition_data_max_length,
        batch_size=config.decomposition_data_batch_size,
        separator=config.decomposition_data_separator,
        seed=LOADER_SEED,
    )

    perplexity_ds = datasets_hf.get_dataset(config.perplexity_data_name)

    logger.info(
        f"Created perplexity dataset {config.perplexity_data_name}, "
        f"{len(perplexity_ds)} examples"
    )

    perplexity_dl = datasets_hf.prepare_dataloader_v1(
        dataset=perplexity_ds,
        tokenizer=tokenizer,
        max_seqlen=config.perplexity_data_max_length,
        batch_size=config.perplexity_data_batch_size,
        separator=config.perplexity_data_separator,
        nsamples=PPL_N_SAMPLES,
        seed=LOADER_SEED,
    )
    return decomposition_dl, perplexity_dl


def make_finetune_fn(
    config: configurator.DecomposeDWAINConfig,
    ft_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
) -> collections.abc.Callable[
    [torch.nn.Module, torch.device, list[str]], torch.nn.Module
]:
    if config.finetuning_run and config.finetuning_use_lora:
        logger.info("Creating lora finetuning function")
        return lambda m, device, decomposed_modules: dwain_wrapper_module.finetune_lora(
            model=m,
            device=device,
            decomposed_modules=decomposed_modules,
            ft_iterator=ft_iterator,
            num_steps=config.finetuning_num_steps,
            lr=config.finetuning_lr,
            num_last_modules_to_finetune=config.finetuning_num_last_finetuned_modules,
            use_rank_pattern=config.finetuning_use_rank_pattern,
            min_rank_to_finetune=config.finetuning_lora_min_rank,
        )
    elif config.finetuning_run and not config.finetuning_use_lora:
        logger.info("Creating full finetuning function")
        return lambda m, device, decomposed_modules: dwain_wrapper_module.finetune_full(
            model=m,
            device=device,
            decomposed_modules=decomposed_modules,
            ft_iterator=ft_iterator,
            num_steps=config.finetuning_num_steps,
            lr=config.finetuning_lr,
            num_last_modules_to_finetune=config.finetuning_num_last_finetuned_modules,
        )
    else:
        logger.info("Creating empty finetuning function")
        return lambda m, device, decomposed_modules: m


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    # 1. SETUP

    start = time.perf_counter()
    transformers.utils.logging.disable_progress_bar()
    config = configurator.DecomposeDWAINConfig(**config_raw)
    dtype = utils.conv_str_to_dtype(config.decomposed_model_dtype)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. CREATE MODEL

    egc = config.decomposed_model_enable_gradient_checkpointing
    model, tokenizer = builder.make_model_and_tokenizer(
        model_name=config.decomposed_model_name,
        model_revision=config.decomposed_model_revision,
        enable_gradient_checkpointing=egc,
        dtype=dtype,
        log_linears=True,
    )
    model.to(device)
    builder.validate_module_names(model, config.blacklisted_modules)

    # 3. PREPARE DATALOADERS

    decomposition_dl, perplexity_dl = make_dataloaders(config, tokenizer)

    # 4. LOG INITIAL STATISTICS

    with torch.no_grad():
        perplexity_initial = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    params_initial = metrics.get_params(model) / 1.0e6
    gflops_initial = metrics.get_giga_flops(model, tensor_size=(1, 512))

    logger.info(f"{perplexity_initial=} {params_initial=} {gflops_initial=}")

    # 5. DO ACTUAL DECOMPOSITION

    model_wrapped = dwain_wrapper_module.WrapperModule(model)
    model_wrapped.eval()

    decomposition_it = make_inifinte_iterator(decomposition_dl)

    finetune_fn = make_finetune_fn(config, decomposition_it)

    blacklisted_module_names_wrapped = dwain_wrapper_module.add_prefix(
        config.blacklisted_modules
    )

    decompose_config = ptdeco.dwain.decompose_in_place(
        module=model_wrapped,
        device=device,
        blacklisted_module_names=blacklisted_module_names_wrapped,
        data_iterator=decomposition_it,
        loss_fn=dwain_wrapper_module.ce_loss,
        finetune_fn=finetune_fn,
        metric_iterator=decomposition_it,
        nsr_final_threshold=config.nsr_final_threshold,
        num_data_steps=config.num_data_steps,
        num_metric_steps=config.num_metric_steps,
        min_rank=config.min_rank,
        trade_off_factor=config.trade_off_factor,
        reduction_factor=config.reduction_factor,
        max_accepted_ppl_diff=config.max_accepted_ppl_diff,
        decompose_in_float64=config.decompose_in_float64,
        precomputing_covariance_num_splits=config.precomputing_covariance_num_splits,
    )

    # 6. SERIALIZE MODEL

    dwain_wrapper_module.save_raw_model_decompose_config_and_state_dict(
        output_path, decompose_config, model.state_dict()
    )

    # 7. LOG FINAL STATISTICS

    with torch.no_grad():
        perplexity_final = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    params_final = metrics.get_params(model) / 1.0e6
    gflops_final = metrics.get_giga_flops(model, tensor_size=(1, 512))
    params_frac = params_final / params_initial * 100.0
    gflops_frac = gflops_final / gflops_initial * 100.0

    logger.info(f"{perplexity_initial=} -> {perplexity_final=}")
    logger.info(f"{params_initial=} -> {params_final=} {params_frac:.2f}")
    logger.info(f"{gflops_initial=} -> {gflops_final=} {gflops_frac:.2f}")

    stop = time.perf_counter()
    time_decomposition_and_perplex_eval = stop - start
    logger.info(
        "Decomposition and perplexity evaluation "
        f"took {time_decomposition_and_perplex_eval:.2f} s"
    )

    # 8. RUN BENCHMARK TASKS ON LM EVAL

    time_lm_eval = -1.0

    if config.lm_eval_tasks:
        start = time.perf_counter()
        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model_wrapped.raw_model,
            tokenizer=tokenizer,
            device=device,
            tasks=config.lm_eval_tasks,
        )
        logger.info("\n" + lm_eval_results_str)
        lm_eval_path = output_path / "lm_eval.json"
        lm_eval_results["config"]["device"] = str(lm_eval_results["config"]["device"])
        with open(lm_eval_path, "wt") as f:
            json.dump(lm_eval_results, f)
        logger.info(f"lm_eval results saved to {lm_eval_path}")
        time_lm_eval = time.perf_counter() - start
        logger.info(f"lm_eval took {time_lm_eval:.2f} s")

    # 9. SAVE SUMMARY

    device_str = str(device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(device)

    summary = {
        "perplexity_initial": perplexity_initial,
        "perplexity_final": perplexity_final,
        "mparams_initial": params_initial,
        "mparams_final": params_final,
        "mparams_frac": params_frac,
        "gflops_initial": gflops_initial,
        "gflops_final": gflops_final,
        "gflops_frac": gflops_frac,
        "time_decomposition_and_perplex_eval": time_decomposition_and_perplex_eval,
        "time_lm_eval": time_lm_eval,
        "device": device_str,
    }

    with open(output_path / "summary.json", "wt") as f:
        json.dump(summary, f)
