import collections
from typing import Any

import torch

__all__ = ["get_module_config", "build_module_from_config"]


def _get_module_config_sequential(m: torch.nn.Sequential) -> dict[str, Any]:
    config: dict[str, Any] = {"type": "Sequential"}
    config["modules"] = {}
    for k, v in m.named_children():
        config["modules"][k] = get_module_config(v)
    return config


def _get_module_config_conv2d(m: torch.nn.Conv2d) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config["type"] = "Conv2d"
    config["in_channels"] = m.in_channels
    config["out_channels"] = m.out_channels
    config["kernel_size"] = m.kernel_size
    config["bias"] = m.bias is not None
    config["groups"] = m.groups
    config["padding"] = m.padding
    config["padding_mode"] = m.padding_mode
    config["stride"] = m.stride
    config["dilation"] = m.dilation
    return config


def _get_module_config_linear(m: torch.nn.Linear) -> dict[str, Any]:
    res: dict[str, Any] = {}
    res["type"] = "Linear"
    res["in_features"] = m.in_features
    res["out_features"] = m.out_features
    res["bias"] = m.bias is not None
    return res


def get_module_config(m: torch.nn.Module) -> dict[str, Any]:
    if isinstance(m, torch.nn.Sequential):
        return _get_module_config_sequential(m)
    elif isinstance(m, torch.nn.Conv2d):
        return _get_module_config_conv2d(m)
    elif isinstance(m, torch.nn.Linear):
        return _get_module_config_linear(m)
    else:
        raise ValueError(f"get_module_config not implemented for {type(m)}")


def _build_conv2d_from_config(config: dict[str, Any]) -> torch.nn.Conv2d:
    assert config["type"] == "Conv2d"
    return torch.nn.Conv2d(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        kernel_size=config["kernel_size"],
        groups=config["groups"],
        bias=config["bias"],
        stride=config["stride"],
        padding=config["padding"],
        padding_mode=config["padding_mode"],
        dilation=config["dilation"],
    )


def _build_linear_from_config(config: dict[str, Any]) -> torch.nn.Linear:
    assert config["type"] == "Linear"
    return torch.nn.Linear(
        in_features=config["in_features"],
        out_features=config["out_features"],
        bias=config["bias"],
    )


def _build_sequential_from_config(config: dict[str, Any]) -> torch.nn.Sequential:
    assert config["type"] == "Sequential"
    modules_config = config["modules"]
    first_key = next(iter(modules_config.keys()))
    if first_key == "0":
        modules_list = [build_module_from_config(v) for v in config["modules"].values()]
        return torch.nn.Sequential(*modules_list)
    else:
        modules_dict = collections.OrderedDict()
        for k, v in modules_config.items():
            modules_dict[k] = build_module_from_config(v)
        return torch.nn.Sequential(modules_dict)


def build_module_from_config(config: dict[str, Any]) -> torch.nn.Module:
    type = config.get("type")
    if type == "Sequential":
        return _build_sequential_from_config(config)
    elif type == "Conv2d":
        return _build_conv2d_from_config(config)
    elif type == "Linear":
        return _build_linear_from_config(config)
    else:
        raise ValueError("{type=} not supported")
