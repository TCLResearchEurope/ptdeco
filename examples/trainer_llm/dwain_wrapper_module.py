import collections
import json
import logging
import pathlib
from typing import Any, Optional, cast

import peft
import ptdeco.utils
import torch
import transformers  # type: ignore

logger = logging.getLogger(__name__)

PREFIX = "raw_model."

# Model wrapper returning logits

class WrapperModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.raw_model = model
        self.config = model.config

    def forward(self, x: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self.raw_model(input_ids=x["input_ids"], **kwargs).logits


def add_prefix(module_names: Optional[list[str]]) -> Optional[list[str]]:
    if module_names is None:
        return None
    return [PREFIX + m_name for m_name in module_names]


def strip_prefix_list(module_names: list[str]) -> list[str]:
    res = []

    for m in module_names:
        if m.startswith(PREFIX):
            res.append(m[len(PREFIX) :])  # noqa: E203 black vs flake
        else:
            res.append(m)
    return res


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
    lr: float = 0.0001,
) -> torch.nn.Module:

    if len(decomposed_modules) == 0:
        return model
    decomposed_modules_to_finetune = decomposed_modules[-num_last_modules_to_finetune:]
    for name, param in model.named_parameters():
        if any(
            [e in name for e in decomposed_modules_to_finetune]
        ):  # and ('Wqkv' in name or 'out_proj' in name):
            msg = f"full fine-tuning - enabling grad for {name}, {param.requires_grad=}"
            logger.info(msg)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_steps,
    )
    counter = 0
    model.train()
    total_loss = 0.0
    for step in range(num_steps):
        batch = ptdeco.utils.to_device(next(ft_iterator), device)
        # TODO: ML remove this cast
        batch = cast(dict[str, torch.Tensor], batch)
        counter += 1
        if step > num_steps:
            break
        optimizer.zero_grad()
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        outputs = model(batch)
        loss = ptdeco.dwain.compute_loss(
            logits=outputs, labels=labels, attention_mask=attention_mask
        )
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

        if step % 10 == 0:
            logger.info(f"Step: {step}/{num_steps}, loss: {total_loss / counter}")
    model.eval()
    return model


def finetune_lora(
    *,
    model: torch.nn.Module,
    device: torch.device,
    ft_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    decomposed_modules: list[str],
    num_last_modules_to_finetune: int = 8,
    num_steps: int = 100,
    lr: float = 0.0001,
    min_rank_to_finetune: int = 32,
    use_rank_pattern: bool = False,
) -> torch.nn.Module:
    if len(decomposed_modules) == 0:
        return model

    decomposed_submodules_to_finetune = decomposed_modules[
        -num_last_modules_to_finetune:
    ]
    for name, param in model.named_parameters():
        if any([e in name for e in decomposed_submodules_to_finetune]):
            msg = f"lora fine-tuning - enabling grad for {name}, {param.requires_grad=}"
            logger.info(msg)
        else:
            param.requires_grad = False
    rank_pattern = {}
    alpha_pattern = {}
    target_modules = []
    for module_name in decomposed_submodules_to_finetune:
        first_module_name = f"{module_name}.0"
        second_module_name = f"{module_name}.1"
        rank = model.get_submodule(first_module_name).out_features
        status = "skipping"
        if rank >= min_rank_to_finetune:
            status = "fine-tuning"
            rank_pattern[first_module_name] = rank // 16
            rank_pattern[second_module_name] = rank // 16
            alpha_pattern[first_module_name] = rank // 32
            alpha_pattern[second_module_name] = rank // 32
            target_modules.extend([first_module_name, second_module_name])
        logger.info(f"{module_name} {status} - {rank=} {min_rank_to_finetune=}")
    if len(rank_pattern) == 0:
        msg = f"Skipping fine-tuning - no modules with rank>={min_rank_to_finetune}"
        logger.info(msg)
        return model

    logger.info(f"Fine-tuning {len(rank_pattern)} modules")

    if not use_rank_pattern:
        rank_pattern = {}
        alpha_pattern = {}
    else:
        msg = "Using rank/alpha patterns. Watch out for overfitting on small datasets"
        logger.warning(msg)
        rank_pattern = strip_prefix_dict(rank_pattern)
        alpha_pattern = strip_prefix_dict(alpha_pattern)
        for k, v in rank_pattern.items():
            logger.info(f"rank_pattern[{k}] = {v}")
        for k, v in alpha_pattern.items():
            logger.info(f"alpha_pattern[{k}] = {v}")

    logger.info(f"Fine-tuning {len(target_modules)} modules")

    lora_config = peft.LoraConfig(
        r=16,
        target_modules=strip_prefix_list(target_modules),
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    peft_model = peft.get_peft_model(model.raw_model, lora_config)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_steps,
    )
    counter = 0
    peft_model.train()
    total_loss = 0.0
    for step in range(num_steps):
        input_dict = ptdeco.utils.to_device(next(ft_iterator), device)
        # TODO: ML remove this cast
        input_dict = cast(dict[str, torch.Tensor], input_dict)
        input_ids = input_dict["input_ids"]
        labels = input_dict["labels"]
        attention_mask = input_dict["attention_mask"]
        counter += 1
        if step > num_steps:
            break
        optimizer.zero_grad()
        outputs = peft_model(input_ids=input_ids)
        loss = ptdeco.dwain.compute_loss(
            logits=outputs.logits, labels=labels, attention_mask=attention_mask
        )
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step % 10 == 0:
            logger.info(
                f"Step: {step}/{num_steps}, loss: {total_loss / counter}, "
                f"lr: {lr_scheduler.get_last_lr()}"
            )
    peft_model.eval()
    model.raw_model = peft_model.merge_and_unload()
    return model
