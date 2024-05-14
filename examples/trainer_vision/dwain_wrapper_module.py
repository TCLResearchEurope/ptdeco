import collections
import json
import logging
import os
import pathlib
import time
from typing import Any, Optional

import ptdeco
import timm  # type:ignore
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


_BATCH_NORM_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    timm.layers.norm_act.BatchNormAct2d,
)


def _batch_norms_in_eval(m: torch.nn.Module) -> None:
    for mod_name, mod in m.named_modules():
        if isinstance(mod, _BATCH_NORM_TYPES):
            mod.eval()
            mod_type_name = ptdeco.utils.get_type_name(mod)
            logger.info(f"Switching {mod_name} ({mod_type_name}) to eval mode")


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
    reverting_checkpoints_dir: Optional[pathlib.Path] = None,
    optimizer_name: str,
    batch_norms_in_eval: bool,
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
    if optimizer_name == "SGD":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        logger.info("Using SGD optimizer")
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        logger.info("Using AdamW optimizer")
    else:
        raise ValueError(f"Unknown {optimizer_name=} only SGD and AdamW are allowed")

    counter = 0
    model.train()
    if batch_norms_in_eval:
        _batch_norms_in_eval(model)
    total_loss = 0.0
    initial_loss = float("nan")
    last_loss = float("nan")

    if reverting_checkpoints_dir is not None:
        pid = os.getpid()
        sd_path = reverting_checkpoints_dir / f"tmp_reverting_state_dict_{pid}.pt"
        torch.save(model.state_dict(), sd_path)

    for step in range(1, num_steps + 1):
        batch = ptdeco.utils.to_device(next(ft_iterator), device)
        counter += 1
        optimizer.zero_grad()
        outputs = model(batch)
        loss = ce_loss(batch, outputs)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        total_loss += loss.item()

        if step % num_log_steps == 0:
            logger.info(f"Step: {step}/{num_steps}, loss: {total_loss / counter}")
            # Thist cheks for NaN
            if initial_loss != initial_loss:
                initial_loss = total_loss / counter
            last_loss = total_loss / counter
            total_loss = 0.0
            counter = 0

    if reverting_checkpoints_dir is not None:
        if initial_loss == initial_loss and initial_loss < last_loss:
            loss_msg = f"{initial_loss=:.4f} < {last_loss=:.4f}"
            logger.info(f"{loss_msg}: keeping the orig weights")
            model.load_state_dict(torch.load(sd_path))
        elif initial_loss == initial_loss and initial_loss >= last_loss:
            loss_msg = f"{initial_loss=:.4f} >= {last_loss=:.4f}"
            logger.info(f"{loss_msg}: using the fine-tuned weights")
        sd_path.unlink(missing_ok=True)

    model.eval()
    stop = time.perf_counter()
    logger.info(f"Full fine-tuning took {stop-start:.2f} seconds")
    return model
