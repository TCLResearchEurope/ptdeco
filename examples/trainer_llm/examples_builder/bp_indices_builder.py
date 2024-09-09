from typing import Any

import torch
import transformers
import llm_pipelines.block_pruning.pruned_blocks as bpp


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
    attn_indices = model_builder_config["bp_attn_indices"]
    mlp_indices = model_builder_config["bp_mlp_indices"]
    model = bpp.build_pruned_model_from_indices(
        model=original_model,
        model_name=model_name,
        attn_indices=attn_indices,
        mlp_indices=mlp_indices,
    )

    return model, tokenizer
