from typing import Any, Optional

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


def calc_lm_eval_metrics(
    model: torch.nn.Module,
    tasks: list[str],
    tokenizer,
    device: torch.device,
    suppress_initialize_tasks: bool = False,
) -> tuple[dict[str, Any], str]:

    import lm_eval

    if not suppress_initialize_tasks:
        lm_eval.tasks.initialize_tasks()

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model, tokenizer=tokenizer, device=device
    )

    num_fewshot = 0
    results = lm_eval.evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size="auto",
        max_batch_size=None,
        device=device,
    )
    results_str = lm_eval.utils.make_table(results)
    return results, results_str


def get_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def get_giga_flops(
    model: torch.nn.Module,
    tensor_size: tuple[int, ...],
    device: Optional[torch.device] = None,
    warnings_off: bool = False,
) -> float:
    import ptdeco
    import fvcore.nn

    if device is None:
        device = ptdeco.utils.get_default_device(model)
    x = torch.ones(size=tensor_size, device=device, dtype=torch.int64)
    fca = fvcore.nn.FlopCountAnalysis(model, x)

    if warnings_off:
        fca.unsupported_ops_warnings(False)

    # NOTE FV.CORE computes MACs not FLOPs !!!!
    # Hence 2.0 * here for proper GIGA FLOPS

    flops = 2 * fca.total()

    return flops / 1.0e9
