import collections
import logging
from typing import Any, Optional

import fvcore.nn  # type: ignore
import ptdeco.utils
import timm  # type: ignore
import torch

logger = logging.getLogger(__name__)


def get_fpops(
    model: torch.nn.Module,
    b_c_h_w: tuple[int, int, int, int],
    units: str = "gflops",
    device: Optional[torch.device] = None,
    warnings_off: bool = False,
) -> float:
    if device is None:
        device = ptdeco.utils.get_default_device(model)
    x = torch.rand(size=b_c_h_w, device=device)
    fca = fvcore.nn.FlopCountAnalysis(model, x)

    if warnings_off:
        fca.unsupported_ops_warnings(False)

    # NOTE FV.CORE computes MACs not FLOPs !!!!
    # Hence 2.0 * here for proper GIGA FLOPS

    flops = 2 * fca.total()

    if units.lower() == "gflops":
        return flops / 1.0e9
    elif units.lower() == "kmapps":
        return flops / b_c_h_w[-1] / b_c_h_w[-2] / 1024.0
    raise ValueError(f"Unknown {units=}")


def get_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def get_model_stats(
    model: torch.nn.Module,
    b_c_h_w: tuple[int, int, int, int],
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    if device is None:
        device = ptdeco.utils.get_default_device(model)
    model.eval()
    return {
        "gflops": get_fpops(model, b_c_h_w=b_c_h_w, units="gflops", device=device),
        "kmapps": get_fpops(model, b_c_h_w=b_c_h_w, units="kmapps", device=device),
        "mparams": get_params(model) / 1.0e6,
    }


def log_model_stats(
    stats_logger: logging.Logger, log_prefix: str, model_stats: dict[str, Any]
) -> None:
    gflops = model_stats["gflops"]
    kmapps = model_stats["kmapps"]
    mparams = model_stats["mparams"]
    msg = f"{log_prefix} gflops={gflops:.2f} kmapps={kmapps:.2f} Mparams={mparams:.2f}"
    acc = model_stats.get("accuracy_val")
    if acc is not None:
        msg += f" {100*acc:.2f}"
    stats_logger.info(msg)


def make_model(model_name: str) -> torch.nn.Module:
    builder, model_name = model_name.split(".", maxsplit=1)

    logger.info(f"Creating model: {builder} {model_name}")

    if builder == "timm":
        return timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError(f"Unknown model builder {builder}")


def validate_module_names(model: torch.nn.Module, module_names: list[str]) -> None:
    known_module_names = {name for name, _ in model.named_modules()}
    unknown_modules = [name for name in module_names if name not in known_module_names]
    if unknown_modules:
        msg = ", ".join(unknown_modules)
        raise ValueError(f"Unknown module names specified: {msg}")


def log_state_dict_keys_stats(
    stats_logger: logging.Logger,
    log_prefix: str,
    model: torch.nn.Module,
    state_dict: collections.OrderedDict[str, torch.Tensor],
) -> int:
    model_state_dict_keys = set(model.state_dict().keys())
    loaded_state_dict_keys = set(state_dict.keys())
    common_state_dict_keys = model_state_dict_keys & loaded_state_dict_keys
    n_common_state_dict_keys = len(common_state_dict_keys)
    msg_model = f"num_model_sd_keys={len(model_state_dict_keys)}"
    msg_sd = f"num_loaded_sd_keys={len(loaded_state_dict_keys)}"
    msg_common = f"num_common_sd_keys={n_common_state_dict_keys}"
    stats_logger.info(f"{log_prefix} {msg_model}, {msg_sd}, {msg_common}")
    return n_common_state_dict_keys
