from typing import Optional

import pydantic


class DecomposeDWAINConfig(pydantic.BaseModel):
    task: str
    decompose_model_name: str
    trade_off_factor: float
    proportion_threshold: float
    ppl_diff_threshold: float
    nsr_final_threshold: float
    num_data_steps: int
    num_metric_steps: int
    blacklisted_module_names: Optional[list[str]] = None
    min_proportion: float
    min_rank: int
    max_length: int
    batch_size: int

    # Finetuning
    run_finetuning: bool
    lora_finetuning: bool
    ft_lr: float = 0.0001
    num_ft_steps: int
    num_last_decomposed_layers_to_finetune: int
