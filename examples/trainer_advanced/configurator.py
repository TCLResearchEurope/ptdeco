import logging
from collections.abc import Iterable
from typing import Any, Optional

import composer
import torch

logger = logging.getLogger(__name__)


def get_precision(config: dict[str, Any]) -> Optional[composer.core.Precision]:
    precision = config.get("precision")
    if "precision" is not None:
        precision = composer.core.Precision(config["precision"])
    logger.info(f"Using {precision=}")
    return precision


def get_lr_scheduler(config: dict[str, Any]) -> composer.optim.ComposerScheduler:
    lr_t_warmup = config.get("lr_t_warmup", "0ba")
    logger.info(f"Using CosineAnnealingWithWarmupScheduler, {lr_t_warmup=}")
    lr_scheduler = composer.optim.CosineAnnealingWithWarmupScheduler(
        t_warmup=lr_t_warmup
    )
    return lr_scheduler


def get_algorithms(config: dict[str, Any]) -> list[composer.Algorithm]:
    algorithms: list[composer.Algorithm] = []
    pfx = "Algorithms -"
    channels_last = config.get("alg_channels_last", True)

    if channels_last:
        algorithms.append(composer.algorithms.ChannelsLast())
        logger.info(f"{pfx} using channels last")
    else:
        logger.info(f"{pfx} NOT USING channels last")

    clipping_type = config.get("alg_gradient_clipping_type", "None")
    if clipping_type is not None and clipping_type != "None":
        clipping_threshold = config.get("alg_gradient_clipping_threshold")
        if not isinstance(clipping_threshold, (float, int)):
            raise ValueError(f"{clipping_threshold=} not float")
        gradient_clipping = composer.algorithms.GradientClipping(
            clipping_type=clipping_type, clipping_threshold=clipping_threshold
        )
        algorithms.append(gradient_clipping)
        logger.info(
            f"{pfx} using gradient clipping {clipping_type=} {clipping_threshold=}"
        )
    else:
        logger.info(f"{pfx} NOT USING gradient clipping")
    return algorithms


def get_optimizer(
    params: Iterable[torch.nn.parameter.Parameter], config: dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_name = config.get("optimizer", "Adam")
    logger.info(f"Using optimizer {optimizer_name}")
    if optimizer_name == "Adam":
        return torch.optim.Adam(params=params, lr=config["lr"])
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params=params, lr=config["lr"])
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")


def get_compile_config(config: dict[str, Any]) -> Optional[dict[str, Any]]:
    compile_config = config.get("compile_config")
    logger.info(f"Using {compile_config=}")
    return compile_config
