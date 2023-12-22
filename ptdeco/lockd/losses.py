import torch

from . import decomposition

__all__ = [
    "calc_entropy_from_logits",
    "get_entropy_dict",
    "get_entropy_loss",
    "get_nsr_dict",
    "get_nsr_loss",
    "get_proportion_dict",
    "get_proportion_loss",
]


def calc_entropy_from_logits(
    logits: torch.Tensor, epsilon: float = 0.01
) -> torch.Tensor:
    probs_ = torch.sigmoid(logits)[..., None]
    probs = torch.cat([probs_, 1.0 - probs_], dim=1)
    return torch.maximum(
        -(probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon)
    )


def get_entropy_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    entropy_dict = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            entropy_dict[submodule_name] = calc_entropy_from_logits(
                submodule.get_logits()
            )
    return entropy_dict


def get_entropy_loss(wrapped_module: torch.nn.Module) -> torch.Tensor:
    entropy_list = []

    for submodule in wrapped_module.modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            entropy_list.append(calc_entropy_from_logits(submodule.get_logits()))

    return torch.stack(entropy_list).mean()


def get_nsr_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    nsr_dict: dict[str, torch.Tensor] = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            nsr_dict[submodule_name] = submodule.get_nsr()
    return nsr_dict


def get_nsr_loss(wrapped_module: torch.nn.Module, nsr_threshold: float) -> torch.Tensor:
    nsr_list: list[torch.Tensor] = []
    for submodule in wrapped_module.modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            nsr = submodule.get_nsr()
            nsr_list.append(torch.relu(nsr - nsr_threshold) / nsr_threshold)
    return torch.stack(nsr_list).mean()


def get_proportion_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    proportion_dict = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            proportion_dict[submodule_name] = decomposition.calc_propotion_from_logits(
                submodule.get_logits()
            )
    return proportion_dict


def get_proportion_loss(wrapped_module: torch.nn.Module) -> torch.Tensor:
    proportion_list: list[torch.Tensor] = []
    for submodule in wrapped_module.modules():
        if isinstance(submodule, decomposition.WrappedLOCKDModule):
            proportion_list.append(
                decomposition.calc_propotion_from_logits(submodule.get_logits())
            )
    return torch.stack(proportion_list).mean()
