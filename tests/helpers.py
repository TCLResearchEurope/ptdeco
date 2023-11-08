import torch

import ptdeco


def is_compound_module(m: torch.nn.Module) -> bool:
    return len(list(m.children())) > 0


def set_min_logits(
    module: torch.nn.Module,
) -> None:
    for child_module in module.children():
        if isinstance(child_module, ptdeco.lockd.WrappedModule):
            with torch.no_grad():
                logits = child_module.get_logits()
                new_logits = -10 * torch.ones_like(logits)
                new_logits[0] = 10.0
                logits.copy_(new_logits)
        elif is_compound_module(child_module):
            set_min_logits(child_module)


def set_half_logits(
    module: torch.nn.Module,
) -> None:
    for child_module in module.children():
        if isinstance(child_module, ptdeco.lockd.WrappedModule):
            with torch.no_grad():
                logits = child_module.get_logits()
                new_logits = -10 * torch.ones_like(logits)
                for i in range(0, len(logits), 2):
                    new_logits[i] = 10.0
                logits.copy_(new_logits)
        elif is_compound_module(child_module):
            set_half_logits(child_module)
