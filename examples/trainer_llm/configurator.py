import pydantic


class DecompoesDWAIN(pydantic.BaseModel):
    task: str
    proportion_threshold: float
    ppl_diff_threshold: float
    nsr_final_threshold: float
    num_data_steps: int
    num_metric_steps: int
    num_ft_steps: int
    blacklisted_module_names: list[str] = None
    blacklisted_substrings: list[str] = None
    min_proportion: float = 0.25
    ft_lr: float = 0.0001
    run_finetuning: bool = False
    trade_off_factor: float = 1.0
    start_layer_num: int = 0
    end_layer_num: int = 23
    min_rank: int = 4
    max_length: int = 512
    batch_size: int = 1
