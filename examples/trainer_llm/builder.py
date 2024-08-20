import json
import logging
from typing import Optional

import ptdeco.utils
import torch
import transformers  # type:ignore

logger = logging.getLogger(__name__)


def _log_linear_submodules(m: torch.nn.Module) -> None:
    msg_list = ["All linear modules of the model:"]

    i = 1
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            bias = "+ bias" if module.bias is not None else "no bias"
            msg = f"  - {name} # ({i}) {bias} {tuple(module.weight.shape)}"
            msg_list.append(msg)
            i += 1
    logger.info("\n".join(msg_list))


def _add_pad_token(
    model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
) -> None:
    if (
        model_name
        in (
            "microsoft/phi-2",
            "upstage/SOLAR-10.7B-v1.0",
            "mistralai/Mistral-7B-Instruct-v0.2",
        )
        or model_name.startswith("meta-llama/Llama-2-")
        or model_name.startswith("meta-llama/Meta-Llama-3-")
        or model_name.startswith("meta-llama/Meta-Llama-3.1-")
        or model_name.startswith("Qwen/Qwen1.5-")
        or model_name.startswith("Qwen/Qwen2-")
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Phi-2 and LLama2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id  # llama, phi
        logger.info("Setting pad_token to eos_token")

    elif model_name == "Qwen/Qwen-1_8B":
        # See "https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Setting pad_token to <|endoftext|>")


def make_model_and_tokenizer(
    *,
    model_name: str,
    model_revision: str,
    enable_gradient_checkpointing: bool,
    dtype: torch.dtype,
    log_linears: bool = False,
) -> tuple[transformers.AutoModelForCausalLM, transformers.PreTrainedTokenizer]:
    model_name = model_name
    model_revision = model_revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    msg = f"Creating {model_name} revision={model_revision} with {dtype=} "
    msg += f"grad_checkpointing={enable_gradient_checkpointing}"
    logger.info(msg)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    _add_pad_token(model=model, tokenizer=tokenizer, model_name=model_name)
    if log_linears:
        _log_linear_submodules(model)
    model.to(dtype)
    model.eval()
    return model, tokenizer


def apply_decompose_config_and_state_dict_in_place(
    *,
    model: torch.nn.Module,
    decompose_config_path: str,
    state_dict_path: str,
    device: torch.device,
    dtype: torch.dtype,
    log_linears: bool = False,
) -> None:

    with open(decompose_config_path, "rt") as f:
        decompose_config = json.load(f)

    ptdeco.utils.apply_decompose_config_in_place(model, decompose_config)
    model.to(device)
    model.to(dtype)
    ptdeco.utils.free_gpu_reserved_memory()
    logger.info(f"Applied decompose config {decompose_config_path}")
    sd = torch.load(state_dict_path, map_location=device)

    model.load_state_dict(sd)

    logger.info(f"Loaded state dict {state_dict_path}")
    model.eval()

    if log_linears:
        _log_linear_submodules(model)


def validate_module_names(
    model: torch.nn.Module, module_names: Optional[list[str]]
) -> None:
    if module_names is not None:
        known_module_names = {name for name, _ in model.named_modules()}
        unknown_modules = [
            name for name in module_names if name not in known_module_names
        ]
        if unknown_modules:
            msg = ", ".join(unknown_modules)
            raise ValueError(f"Unknown module names specified: {msg}")
