import collections
import json
import logging
import pathlib
import time
from typing import Any

import nvidia.dali.plugin.pytorch  # type:ignore
import ptdeco.falor
import torch
import torchmetrics


import builder
import configurator
import datasets_dali

logger = logging.getLogger(__name__)


def make_image_iterator(
    train_dataloader: collections.abc.Iterator[dict[str, torch.Tensor]],
) -> collections.abc.Iterator[torch.Tensor]:
    for d in train_dataloader:
        yield d["inputs"].permute(0, 3, 1, 2)


def calc_accuracy(*, model, valid_pipeline, device, n_batches=None):

    model.eval()

    model.to(device)

    val_iter = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            valid_pipeline, ["inputs", "targets"]
        )
    )
    if n_batches is None:
        n_batches = len(val_iter)

    # pbar = tqdm(total=n_batches)
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
            # pbar.update(1)
        res = metrics.compute().item()

    del valid_pipeline

    return res


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    config = configurator.DecomposeFALORConfig(**config_raw)
    b_c_h_w = (1, 3, *config.input_h_w)

    train_pipeline, valid_pipeline = datasets_dali.make_imagenet_pipelines(
        imagenet_root_dir=config.imagenet_root_dir,
        trn_image_classes_fname=config.trn_imagenet_classes_fname,
        val_image_classes_fname=config.val_imagenet_classes_fname,
        batch_size=config.batch_size,
        normalization=config.normalization,
        h_w=config.input_h_w,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataloader = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            [train_pipeline], ["inputs", "targets"]
        )
    )

    data_iterator = make_image_iterator(train_dataloader)

    model = builder.make_model(config.decompose_model_name)
    t_val_start = time.perf_counter()
    accuracy_val_initial = calc_accuracy(
        model=model,
        valid_pipeline=valid_pipeline,
        device=device,
    )
    t_val = time.perf_counter() - t_val_start
    logger.info(f"Initial accuracy {accuracy_val_initial=}, eval took  {t_val:.2f} s")

    builder.validate_module_names(model, config.blacklisted_modules)

    model.to(device)
    stats_initial = builder.get_model_stats(model, b_c_h_w)
    stats_initial["accuracy_val"] = accuracy_val_initial

    decompose_config = ptdeco.falor.decompose_in_place(
        module=model,
        device=device,
        data_iterator=data_iterator,
        proportion_threshold=config.proportion_threshold,
        kl_final_threshold=config.kl_final_threshold,
        nsr_final_threshold=config.nsr_final_threshold,
        num_data_steps=config.num_data_steps,
        num_metric_steps=config.num_metric_steps,
        blacklisted_module_names=config.blacklisted_modules,
    )
    stats_final = builder.get_model_stats(model, b_c_h_w)
    t_val_start = time.perf_counter()
    accuracy_val_final = calc_accuracy(
        model=model,
        valid_pipeline=valid_pipeline,
        device=device,
    )
    t_val = time.perf_counter() - t_val_start
    logger.info(f"Final accuracy {accuracy_val_final=},  eval took  {t_val:.2f} s")
    stats_final["accuracy_val"] = accuracy_val_final

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    builder.log_model_stats(logger, "Original model  :", stats_initial)
    builder.log_model_stats(logger, "Decomposed model:", stats_final)
