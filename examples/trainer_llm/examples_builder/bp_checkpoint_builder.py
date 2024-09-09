from typing import Any

import torch
import transformers
import llm_pipelines.block_pruning.loaders as bpl


def make_model_and_tokenizer(
    *,
    model_name: str,
    model_revision: str,
    dtype: torch.dtype,
    model_builder_config: dict[str, Any],
) -> tuple[transformers.AutoModelForCausalLM, transformers.PreTrainedTokenizer]:

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=model_revision,
    )
    original_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model_path = model_builder_config["bp_model_path"]
    load_state_dict = model_builder_config["bp_load_state_dict"]
    model = bpl.load_bp_model(
        model_name=model_name,
        original_model=original_model,
        pruned_model_path=model_path,
        load_state_dict=load_state_dict,
    )

    return model, tokenizer
