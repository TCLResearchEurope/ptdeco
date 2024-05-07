import collections
import json
import logging
import pathlib
import time
from typing import Any

import ptdeco
import torch

PREFIX = "raw_model."

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


def add_prefix(module_names: list[str]) -> list[str]:
    return [PREFIX + m_name for m_name in module_names]


def strip_prefix_dict(d: dict[str, Any]) -> dict[str, Any]:

    res: dict[str, Any] = {}
    if isinstance(d, collections.OrderedDict):
        res = collections.OrderedDict()

    for k, v in d.items():
        if k.startswith(PREFIX):
            res[k[len(PREFIX) :]] = v  # noqa: E203 black vs flake
        else:
            res[k] = v
    return res


def save_raw_model_decompose_config_and_state_dict(
    output_path: pathlib.Path,
    decompose_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> None:
    out_decompose_config_path = output_path / "decompose_config.json"

    with open(out_decompose_config_path, "wt") as f:
        json.dump(strip_prefix_dict(decompose_config), f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"

    torch.save(strip_prefix_dict(state_dict), out_decompose_state_dict_path)


def finetune_full(
    *,
    model: torch.nn.Module,
    device: torch.device,
    ft_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    decomposed_modules: list[str],
    num_last_modules_to_finetune: int = 8,
    num_steps: int = 100,
    num_log_steps: int = 10,
    lr: float = 0.0001,
) -> torch.nn.Module:

    if len(decomposed_modules) == 0:
        logger.info("Skipping full fine-tuning - empty list of decomposed modules")
        return model

    start = time.perf_counter()
    decomposed_modules_to_finetune = decomposed_modules[-num_last_modules_to_finetune:]
    for name, param in model.named_parameters():
        if any([e in name for e in decomposed_modules_to_finetune]):
            msg = f"full fine-tuning - enabling grad for {name}, {param.requires_grad=}"
            logger.info(msg)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

        if step % num_log_steps == 0:
            logger.info(f"Step: {step}/{num_steps}, loss: {total_loss / counter}")
            total_loss = 0.0
            counter = 0
    model.eval()
    stop = time.perf_counter()
    logger.info(f"Full fine-tuning took {stop-start:.2f} seconds")
    return model
