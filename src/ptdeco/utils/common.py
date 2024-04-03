import gc
import logging
from typing import Any

import torch

__all__ = [
    "to_device",
    "get_gpu_reserved_memory_gb",
    "free_gpu_reserved_memory",
    "get_num_params",
    "is_compound_module",
    "get_type_name",
    "get_default_device",
    "split_module_parent_child_name",
    "replace_submodule_in_place",
]


logger = logging.getLogger(__name__)


def to_device(
    o: torch.Tensor | dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(o, torch.Tensor):
        return o.to(device)
    elif isinstance(o, dict):
        res = {}
        for k, v in o.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.to(device)
            else:
                res[k] = v
        return res
    raise ValueError(f"Unsupported type {type(o)}")


def get_gpu_reserved_memory_gb() -> float:
    mem = sum(
        torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count())
    )
    return mem / (1024.0**3)


def free_gpu_reserved_memory() -> None:
    if torch.cuda.is_available():
        memory_before = get_gpu_reserved_memory_gb()
        gc.collect()
        torch.cuda.empty_cache()
        memory_after = get_gpu_reserved_memory_gb()
        logger.info(
            f"GPU memory: {memory_before:.2f} -> {memory_after:.2f} GB"
            f" ({(memory_after - memory_before):.2f} GB)"
        )


def get_num_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def is_compound_module(m: torch.nn.Module) -> bool:
    return len(list(m.children())) > 0


def get_type_name(o: Any) -> str:
    to = type(o)
    return to.__module__ + "." + to.__name__


def get_default_device(module: torch.nn.Module) -> torch.device:
    p = next(module.parameters(), None)
    if p is None:
        return torch.device("cpu")
    else:
        return p.device


def split_module_parent_child_name(target: str) -> tuple[str, str]:
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_submodule_in_place(
    root_module: torch.nn.Module, submodule_name: str, new_submodule: torch.nn.Module
) -> None:
    parent_name, child_name = split_module_parent_child_name(submodule_name)
    parent_module = root_module.get_submodule(parent_name)
    setattr(parent_module, child_name, new_submodule)
