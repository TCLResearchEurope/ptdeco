from typing import Any, Literal, Optional

import pydantic
from typing_extensions import Annotated

DTYPES_PATTERN = r"^torch.float32$|^torch.bfloat16$|^torch.float16$"


class _VersionConfig(pydantic.BaseModel):
    ptdeco_trainer_llm_version: Optional[str] = None
    ptdeco_version: Optional[str] = None


class FinetuneConfig(_VersionConfig):
    task: Literal["finetune"]

    # Model specification

    decomposed_model_name: str
    decomposed_model_revision: str
    decomposed_model_custom_builder_path: Optional[str]
    decomposed_model_custom_builder_config: Optional[dict[str, Any]]
    decomposed_model_dtype: Annotated[
        str, pydantic.StringConstraints(pattern=DTYPES_PATTERN)
    ]
    decomposed_model_enable_gradient_checkpointing: bool
    decompose_config: str
    decompose_state_dict: str

    perplexity_data_name: str
    perplexity_data_separator: str
    perplexity_data_max_length: int
    perplexity_data_batch_size: int

    train_data_name: str
    train_data_separator: str
    train_data_max_length: int
    train_data_batch_size: int
    train_data_n_samples: int

    test_data_name: str
    test_data_separator: str
    test_data_max_length: int
    test_data_batch_size: int
    test_data_n_samples: int

    num_train_epochs: int
    finetune_only_decomposed: bool
    eval_strategy: str
    eval_steps: int
    save_total_limit: int
    save_steps: int
    logging_steps: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    lr_scheduler_type: Literal["linear_with_warmup", "cosine_with_warmup"]
    num_warmup_steps: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # lm_eval evaluation params
    lm_eval_initial: bool
    lm_eval_tasks: Optional[list[str]]

    model_config = pydantic.ConfigDict(extra="forbid")


class DecomposeDWAINConfig(_VersionConfig):
    task: Literal["decompose_dwain"]

    # Model specification

    decomposed_model_name: str
    decomposed_model_revision: str
    decomposed_model_custom_builder_path: Optional[str]
    decomposed_model_custom_builder_config: Optional[dict[str, Any]]
    decomposed_model_dtype: Annotated[
        str, pydantic.StringConstraints(pattern=DTYPES_PATTERN)
    ]
    decomposed_model_enable_gradient_checkpointing: bool

    # Tokenizer and data handling params
    decomposition_data_name: str | list[str]
    decomposition_data_separator: str
    decomposition_data_max_length: int
    decomposition_data_batch_size: int

    perplexity_data_name: str
    perplexity_data_separator: str
    perplexity_data_max_length: int
    perplexity_data_batch_size: int

    # Decomposition params

    num_data_steps: int
    num_metric_steps: int
    trade_off_factor: float
    reduction_factor: float
    max_accepted_ppl_diff: float
    nsr_final_threshold: float
    min_rank: int
    decompose_in_float64: bool
    precomputing_covariance_num_splits: Optional[int]
    blacklisted_modules: list[str]

    # Finetuning params

    finetuning_run: bool
    finetuning_use_lora: bool
    finetuning_lora_min_rank: int
    finetuning_lr: float = 0.0001
    finetuning_num_steps: int
    finetuning_num_last_finetuned_modules: int
    finetuning_use_rank_pattern: bool

    # lm_eval evaluation params
    lm_eval_initial: bool
    lm_eval_tasks: Optional[list[str]]

    model_config = pydantic.ConfigDict(extra="forbid")
