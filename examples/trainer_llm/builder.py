import json
import logging
from typing import Optional

import ptdeco.utils
import torch
import transformers  # type:ignore

logger = logging.getLogger(__name__)


def _log_linear_submodules(m: torch.nn.Module) -> None:
    res = ["All Linear modules of the model:"]

    i = 1
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            res.append(f"  - {name}  # ({i}) {tuple(module.weight.shape)}")
            i += 1
    logger.info("\n".join(res))


def _add_pad_token(
    model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
) -> None:
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
        logger.info("Setting pad_token to eos_token")

    if model_name == "Qwen/Qwen-1_8B":
        "https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"


def make_model_and_tokenizer(
    *,
    model_name: str,
    model_revision: str,
    enable_gradient_checkpointing: bool,
    device: torch.device,
    dtype: torch.dtype,
    decompose_config_path: Optional[str] = None,
    state_dict_path: Optional[str] = None,
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

    if decompose_config_path is not None:
        with open(decompose_config_path, "rt") as f:
            decompose_config = json.load(f)

        ptdeco.utils.apply_decompose_config_in_place(model, decompose_config)
        ptdeco.utils.free_gpu_reserved_memory()
        logger.info(f"Applied decompose config {decompose_config_path}")
    model.to(device)
    model.to(dtype)
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(sd)
        logger.info(f"Loaded state dict {state_dict_path}")
    model.eval()
    if log_linears:
        _log_linear_submodules(model)
    return model, tokenizer
