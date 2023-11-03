from typing import Any

import torch


def is_compound_module(m: torch.nn.Module) -> bool:
    return len(list(m.children())) > 0


def get_type_name(o: Any) -> str:
    to = type(o)
    return to.__module__ + "." + to.__name__


def split_module_parent_child_name(target: str) -> tuple[str, str]:
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def get_default_device(module: torch.nn.Module) -> torch.device:
    p = next(module.parameters(), None)
    if p is None:
        return torch.device("cpu")
    else:
        return p.device


def calc_per_channel_noise_to_signal_ratio(
    x: torch.Tensor,
    y: torch.Tensor,
    non_channel_dim: tuple[int, ...] = (0, 2, 3),
    epsilon: float = 1e-3,
) -> torch.Tensor:
    y_per_channel_variance = torch.square(torch.std(y, dim=non_channel_dim))
    per_channel_squared_difference = torch.square((x - y)).mean(dim=non_channel_dim)

    return torch.divide(
        per_channel_squared_difference, y_per_channel_variance + epsilon
    ).mean()


def calc_kl_divergence(
    q_logits: torch.Tensor,
    p_logits: torch.Tensor,
) -> torch.Tensor:
    q_prob = torch.softmax(q_logits, dim=-1)
    p_prob = torch.softmax(p_logits, dim=-1)
    return torch.multiply(p_prob, torch.log(torch.divide(p_prob, q_prob))).sum(dim=1)


def calc_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    return torch.maximum(
        calc_kl_divergence(student_logits, teacher_logits),
        calc_kl_divergence(teacher_logits, student_logits),
    ).mean()
