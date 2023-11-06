import logging

import fvcore.nn  # type: ignore
import timm  # type: ignore
import torch

logger = logging.getLogger(__name__)


def get_fpops(
    model: torch.nn.Module,
    b_c_h_w: tuple[int, int, int, int],
    units: str = "gflops",
    device: torch.device = torch.device("cpu"),
    warnings_off: bool = False,
) -> float:
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


def create_model(model_name) -> torch.nn.Module:
    builder, model_name = model_name.split(".", maxsplit=1)

    logger.info(f"Creating model: {builder} {model_name}")

    if builder == "timm":
        return timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError(f"Unknown model builder {builder}")
