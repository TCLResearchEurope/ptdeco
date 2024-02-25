from typing import Any

import logging
import time

import torch

logger = logging.getLogger(__name__)


def _sync_gpus() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


def _map_tensors(
    obj: Any, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: _map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj


def calc_perplexity(
    model,
    testloader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    pad_token_id: int,
) -> float:
    _sync_gpus()

    t1 = time.perf_counter()

    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

    nlls = []

    logging.info("Perplexity evaluation started")
    for batch in testloader:
        batch = _map_tensors(batch, device=device)

        logits = model(**batch).logits

        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    perplexity = torch.exp(nlls_tensor.mean())

    _sync_gpus()

    t2 = time.perf_counter()

    logging.info("Perplexity evaluation finished")

    return perplexity.item()