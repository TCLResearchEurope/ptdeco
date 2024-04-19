import collections.abc
from typing import Optional

import torch
import torchmetrics


def calc_accuracy(
    *,
    model: torch.nn.Module,
    val_iter: collections.abc.Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    n_batches: Optional[int] = None,
) -> float:

    model.eval()

    model.to(device)

    if n_batches is None:
        if isinstance(val_iter, collections.abc.Sized):
            n_batches = len(val_iter)
        else:
            raise ValueError("Neither len(val_iterator) nor n_batches is avaialble")

    with torch.inference_mode():
        metrics = torchmetrics.classification.MulticlassAccuracy(
            num_classes=1000,
        )

        metrics.to(device)

        for i, batch in enumerate(val_iter):
            if i >= n_batches:
                break
            inputs, targets = batch["inputs"], batch["targets"]
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = torch.argmax(targets, dim=1)
            outputs = torch.softmax(outputs, dim=1)
            metrics.update(outputs, targets)
        res = metrics.compute().item()
    return res
