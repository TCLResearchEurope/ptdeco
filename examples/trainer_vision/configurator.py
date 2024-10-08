import logging
from collections.abc import Iterable
from typing import Any, Literal, Optional

import composer
import pydantic
import torch

logger = logging.getLogger(__name__)


class _VersionConfig(pydantic.BaseModel):
    ptdeco_trainer_version: Optional[str] = None
    ptdeco_version: Optional[str] = None


class _DataConfig(pydantic.BaseModel):
    imagenet_root_dir: str
    trn_imagenet_classes_fname: str
    val_imagenet_classes_fname: str
    batch_size: int
    normalization: Literal["zero_to_one", "negative_one_to_one", "imagenet", "identity"]
    input_h_w: tuple[int, int]


class _TrainConfig(pydantic.BaseModel):
    finetune_only_decomposed: bool
    lr: float
    lr_t_warmup: str
    lr_scheduler: Literal["cosine", "fixed"]
    max_duration: str
    optimizer: Literal["SGD", "Adam", "AdamW"]
    precision: Optional[Literal["fp32", "amp_fp16", "amp_bf16", "amp_fp8"]]
    alg_channel_last: bool
    alg_gradient_clipping_type: Optional[Literal["norm", "value", "adaptive"]]
    alg_gradient_clipping_threshold: Optional[float] = None
    compile_config: Optional[dict[str, Any]]


class DecomposeLOCKDConfig(_DataConfig, _TrainConfig):
    task: Literal["decompose_lockd"]
    decompose_model_name: str
    proportion_threshold: float
    blacklisted_modules: list[str]
    lmbda: float
    nsr_threshold: float

    model_config = pydantic.ConfigDict(extra="forbid")


class DecomposeFALORConfig(_VersionConfig, _DataConfig):
    task: Literal["decompose_falor"]
    decompose_model_name: str
    proportion_threshold: float
    blacklisted_modules: list[str]
    kl_final_threshold: float
    nsr_final_threshold: float
    num_data_steps: int
    num_metric_steps: int
    use_float64: bool
    model_config = pydantic.ConfigDict(extra="forbid")


class DecomposeDWAINConfig(_VersionConfig, _DataConfig):
    task: Literal["decompose_dwain"]
    decompose_model_name: str

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

    finetuning_run: bool
    finetuning_lr: float
    finetuning_optimizer: Literal["SGD", "Adam", "AdamW"]
    finetuning_reverting: bool
    finetuning_batch_norms_in_eval: bool
    finetuning_num_steps: int
    finetuning_num_log_steps: int
    finetuning_num_last_finetuned_modules: int

    model_config = pydantic.ConfigDict(extra="forbid")


class FinetuneConfig(_VersionConfig, _DataConfig, _TrainConfig):
    task: Literal["finetune"]
    decompose_model_name: str
    decompose_config: str
    decompose_state_dict: str
    proportion_threshold: float
    blacklisted_modules: list[str]

    model_config = pydantic.ConfigDict(extra="forbid")


def get_precision(config: _TrainConfig) -> Optional[composer.core.Precision]:
    if config.precision is not None:
        precision = composer.core.Precision(config.precision)
    else:
        precision = None
    logger.info(f"Using precision={config.precision}")
    return precision


def get_lr_scheduler(config: _TrainConfig) -> composer.optim.ComposerScheduler:
    lr_t_warmup = config.lr_t_warmup

    if config.lr_scheduler == "cosine":
        logger.info(f"Using cosine lr schedule, {lr_t_warmup=}")
        lr_scheduler: composer.optim.ComposerScheduler = (
            composer.optim.CosineAnnealingWithWarmupScheduler(t_warmup=lr_t_warmup)
        )
    elif config.lr_scheduler == "fixed":
        logger.info(f"Using fixed lr schedule, {lr_t_warmup=}")
        lr_scheduler = composer.optim.LinearWithWarmupScheduler(
            t_warmup=lr_t_warmup,
            alpha_i=1.0,
            alpha_f=1.0,
        )
    else:
        msg = f"Unknow scheduler {config.lr_scheduler} != cosine or fixed"
        raise ValueError(msg)

    return lr_scheduler


def get_algorithms(config: _TrainConfig) -> list[composer.Algorithm]:
    algorithms: list[composer.Algorithm] = []
    pfx = "Algorithms -"

    if config.alg_channel_last:
        algorithms.append(composer.algorithms.ChannelsLast())
        logger.info(f"{pfx} using channels last")
    else:
        logger.info(f"{pfx} NOT USING channels last")

    if config.alg_gradient_clipping_type is not None:
        if config.alg_gradient_clipping_threshold is None:
            raise ValueError("clipping_threshold not specified")
        gradient_clipping = composer.algorithms.GradientClipping(
            clipping_type=config.alg_gradient_clipping_type,
            clipping_threshold=config.alg_gradient_clipping_threshold,
        )
        algorithms.append(gradient_clipping)
        msg_type = f"clipping_type={config.alg_gradient_clipping_type}"
        msg_threshold = f"clipping_threshold={config.alg_gradient_clipping_threshold}"
        logger.info(f"{pfx} using gradient clipping {msg_type} {msg_threshold}")
    else:
        logger.info(f"{pfx} NOT USING gradient clipping")
    return algorithms


def get_optimizer(
    params: Iterable[torch.nn.parameter.Parameter], config: _TrainConfig
) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer
    logger.info(f"Using optimizer {optimizer_name}")
    if optimizer_name == "Adam":
        return torch.optim.Adam(params=params, lr=config.lr)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params=params, lr=config.lr)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params=params, lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")


def get_compile_config(config: _TrainConfig) -> Optional[dict[str, Any]]:
    compile_config = config.compile_config
    logger.info(f"Using {compile_config=}")
    return compile_config
