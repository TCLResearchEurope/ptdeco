from typing import Any
import json
import logging
import pathlib
import time
import sys

import datasets
import torch
import transformers
import ptdeco

import configurator
import datasets_hf
import metrics


PPL_EVAL_VARIED_SEQLEN = False
LOADER_SEED = 42


logger = logging.getLogger(__name__)


def setup_logging():
    # TENSORFLOW style format
    fmt = "%(asctime)s.%(msecs)03d: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s"

    # SIMPLER style format
    # fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "datasets_hf", "metrics", "ptdeco"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def make_inifinte_iterator(dl):
    while True:
        for x in dl:
            yield x


# def make_padding_tokenizer(
#     model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
# ) -> transformers.PreTrainedTokenizer:
#     if model_name in (
#         "meta-llama/Llama-2-7b-hf",
#         "microsoft/phi-2",
#         "Qwen/Qwen1.5-1.8B",
#         "Qwen/Qwen1.5-0.5B",
#         "upstage/SOLAR-10.7B-v1.0",
#         "mistralai/Mistral-7B-Instruct-v0.2",
#     ):
#         tokenizer.pad_token = (
#             tokenizer.eos_token
#         )  # Phi-2 and LLama2 models don't have a pad token by default
#         model.config.pad_token_id = tokenizer.pad_token_id  # llama, phi
#         logger.warning(f"Setting pad_token to eos_token")

#     if model_name == "Qwen/Qwen-1_8B":
#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             "Qwen/Qwen-1.8B", trust_remote_code=True, pad_token="<|endoftext|>"
#         )
#     return tokenizer


def conv_str_to_dtype(s: str) -> torch.dtype:
    if s == "torch.float32":
        return torch.float32
    elif s == "torch.bfloat16":
        return torch.bfloat16
    elif s == "torch.float16":
        return torch.float16
    raise ValueError(f"Unknown dtype {s}")


def log_linear_submodules(m: torch.nn.Module) -> None:
    res = ["Linear modules:"]
    # for n, m in model_wrapped.named_modules():
    #     if isinstance(m, torch.nn.Linear):
    #         logger.info(f"{n}, {m.weight.shape}")

    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            res.append(f"  - {name}  # {tuple(module.weight.shape)}")
    logger.info("\n".join(res))


def add_pad_token(
    model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
):
    if model_name in (
        "meta-llama/Llama-2-7b-hf",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-1.8B",
        "upstage/SOLAR-10.7B-v1.0",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Phi-2 and LLama2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id  # llama, phi
        logger.info(f"Setting pad_token to eos_token")

    if model_name == "Qwen/Qwen-1_8B":
        "https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"


def create_model_and_tokenizer(
    config: configurator.DecomposeDWAINConfig, device: torch.device, dtype: torch.dtype
) -> tuple[transformers.AutoModelForCausalLM, transformers.PreTrainedTokenizer]:
    model_name = config.decomposed_model_name
    model_revision = config.decomposed_model_revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    logger.info(f"Creating {model_name} rev. {model_revision} with {dtype=}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if config.decomposed_model_enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    add_pad_token(model=model, tokenizer=tokenizer, model_name=model_name)
    dtype = conv_str_to_dtype(config.decomposed_model_dtype)
    model.to(device)
    model.to(dtype)
    model.eval()
    log_linear_submodules(model)
    return model, tokenizer


def create_dataloaders(
    config: configurator.DecomposeDWAINConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[
    torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    torch.utils.data.DataLoader[dict[str, torch.Tensor]],
]:
    decomposition_ds = datasets_hf.get_dataset(config.decomposition_data_name)
    decomposition_dl = datasets_hf.prepare_dataloader_v2(
        dataset=decomposition_ds,
        tokenizer=tokenizer,
        max_seqlen=config.decomposition_data_max_length,
        batch_size=config.decomposition_data_batch_size,
        separator=config.decomposition_data_separator,
        seed=LOADER_SEED,
    )

    perplexity_ds = datasets_hf.get_dataset(config.perplexity_data_name)
    perplexity_dl = datasets_hf.prepare_slicegpt_dataloader(
        dataset=perplexity_ds,
        tokenizer=tokenizer,
        max_seqlen=config.perplexity_data_max_length,
        batch_size=config.perplexity_data_batch_size,
        separator=config.perplexity_data_separator,
        nsamples=1000,
        seed=LOADER_SEED,
    )
    return decomposition_dl, perplexity_dl


def save_model(
    output_path: pathlib.Path,
    decompose_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> None:
    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    # TODO Remove model prefix from state dict !!!
    torch.save(state_dict, out_decompose_state_dict_path)


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    # 1. SETUP
    start = time.perf_counter()
    transformers.utils.logging.disable_progress_bar()
    config = configurator.DecomposeDWAINConfig(**config_raw)
    dtype = conv_str_to_dtype(config.decomposed_model_dtype)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. CREATE MODEL
    model, tokenizer = create_model_and_tokenizer(config, device, dtype)

    # 3. PREPARE DATALOADERS
    decomposition_dl, perplexity_dl = create_dataloaders(config, tokenizer)

    # 4. LOG INITIAL STATISTICS
    with torch.no_grad():
        perplexity_orig = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    params_orig = metrics.get_params(model) / 1.0e6
    gflops_orig = metrics.get_giga_flops(model, tensor_size=(1, 512))

    logger.info(f"{perplexity_orig=} {params_orig=} {gflops_orig=}")

    # 5. DO ACTUAL DECOMPOSITION

    class WrapperModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(self, x):
            return self.model(**x).logits

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return self.model.prepare_inputs_for_generation(input_ids, **kwargs)

    model_wrapped = WrapperModule(model)

    num_layers = config.num_last_decomposed_layers_to_finetune
    decompose_config = ptdeco.dwain.decompose_in_place(
        module=model_wrapped,
        device=device,
        dtype=dtype,
        blacklisted_module_names=config.blacklisted_module_names,
        data_iterator=make_inifinte_iterator(decomposition_dl),
        ft_iterator=decomposition_dl,
        metric_iterator=decomposition_dl,
        nsr_final_threshold=config.nsr_final_threshold,
        ppl_diff_threshold=config.ppl_diff_threshold,
        num_data_steps=config.num_data_steps,
        num_metric_steps=config.num_metric_steps,
        num_ft_steps=config.num_ft_steps,
        ft_lr=config.ft_lr,
        min_rank=config.min_rank,
        trade_off_factor=config.trade_off_factor,
        num_last_decomposed_layers_to_finetune=num_layers,
        run_finetuning=config.run_finetuning,
        lora_finetuning=config.lora_finetuning,
        decompose_in_float64=config.decompose_in_float64,
        precomputing_covariance_num_splits=config.precomputing_covariance_num_splits,
    )

    # 6. SERIALIZE MODEL
    save_model(output_path, decompose_config, model.state_dict())

    # 7. LOG FINAL STATISTICS

    with torch.no_grad():
        perplexity_final = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    params_final = metrics.get_params(model) / 1.0e6
    gflops_final = metrics.get_giga_flops(model, tensor_size=(1, 512))
    params_frac = params_final / params_orig * 100.0
    gflops_frac = gflops_final / gflops_orig * 100.0

    logger.info(f"{perplexity_orig=} -> {perplexity_final=}")
    logger.info(f"{params_orig=} -> {params_final=} {params_frac:.2f}")
    logger.info(f"{gflops_orig=} -> {gflops_final=} {gflops_frac:.2f}")

    stop = time.perf_counter()
    time_decomposition_and_perplex_eval = stop - start
    logger.info(
        "Decomposition and perplexity evaluation "
        f"took {time_decomposition_and_perplex_eval:.2f} s"
    )

    # 8. RUN BENCHMARK TASKS ON LM EVAL
    time_lm_eval = -1.0

    if config.lm_eval_tasks is not None and len(config.lm_eval_tasks) > 0:
        start = time.perf_counter()
        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model_wrapped.model,
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
        "perplexity_orig": perplexity_orig,
        "perplexity_final": perplexity_final,
        "mparams_orig": params_orig,
        "mparams_final": params_final,
        "mparams_frac": params_frac,
        "gflops_orig": gflops_orig,
        "gflops_final": gflops_final,
        "gflops_frac": gflops_frac,
        "time_decomposition_and_perplex_eval": time_decomposition_and_perplex_eval,
        "time_lm_eval": time_lm_eval,
        "device": device_str,
    }

    with open(output_path / "summary.json", "wt") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
