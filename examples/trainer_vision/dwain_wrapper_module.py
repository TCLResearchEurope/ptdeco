import collections
import time
import logging
from typing import Any

import torch

import ptdeco


logger = logging.getLogger(__name__)


class WrapperModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.raw_model = model

    def forward(self, x: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self.raw_model(x["inputs"])


def ce_loss(input_dict: dict[str, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
    target = input_dict["targets"]
    return torch.nn.functional.cross_entropy(input=output, target=target)


def finetune_full(
    *,
    model: torch.nn.Module,
    device: torch.device,
    ft_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    decomposed_modules: list[str],
    num_last_modules_to_finetune: int = 8,
    num_steps: int = 100,
    lr: float = 0.0001,
) -> torch.nn.Module:

    if len(decomposed_modules) == 0:
        logger.info("Skipping full fine-tuning - empty list of decomposed modules")
        return model

    start = time.perf_counter()
    decomposed_modules_to_finetune = decomposed_modules[-num_last_modules_to_finetune:]
    for name, param in model.named_parameters():
        if any(
            [e in name for e in decomposed_modules_to_finetune]
        ):
            msg = f"full fine-tuning - enabling grad for {name}, {param.requires_grad=}"
            logger.info(msg)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    # lr_scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=10,
    #     num_training_steps=num_steps,
    # )
    counter = 0
    model.train()
    total_loss = 0.0
    for step in range(num_steps):
        batch = ptdeco.utils.to_device(next(ft_iterator), device)
        counter += 1
        if step > num_steps:
            break
        optimizer.zero_grad()
        outputs = model(batch)
        loss = ce_loss(batch, outputs)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        total_loss += loss.item()

        if step % 10 == 0:
            logger.info(f"Step: {step}/{num_steps}, loss: {total_loss / counter}")
    model.eval()
    stop = time.perf_counter()
    logger.info(f"Full fine-tuning took {stop-start:.2f} seconds")
    return model
