from typing import Any

import torch

__all__ = [
    "is_compound_module",
    "get_type_name",
    "get_default_device",
    "split_module_parent_child_name",
    "replace_submodule_in_place",
]


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
