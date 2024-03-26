from typing import Optional
from typing_extensions import Annotated
import pydantic


DTYPES_PATTERN = r"^torch.float32$|^torch.bfloat16$|^torch.float16$"


class DecomposeDWAINConfig(pydantic.BaseModel):
    task: str

    # Model specification

    decomposed_model_name: str
    decompoed_model_revision: str
    decomposed_model_dtype: Annotated[
        str, pydantic.StringConstraints(pattern=DTYPES_PATTERN)
    ]
    decomposed_model_enable_gradient_checkpointing: bool
    # Tokenizer and Data handling params
    metric_separator: str
    metric_max_length: int
    metric_batch_size: int
    data_separator: str
    data_max_length: int
    data_batch_size: int

    # Decomposition params

    num_data_steps: int
    num_metric_steps: int
    trade_off_factor: float
    proportion_threshold: float
    ppl_diff_threshold: float
    nsr_final_threshold: float
    min_proportion: float
    min_rank: int
    blacklisted_module_names: Optional[list[str]] = None
    decompose_in_float64: bool

    # Finetuning params

    run_finetuning: bool
    lora_finetuning: bool
    ft_lr: float = 0.0001
    num_ft_steps: int
    num_last_decomposed_layers_to_finetune: int

    # lm_eval evaluation params
    lm_eval_tasks: Optional[list[str]]
