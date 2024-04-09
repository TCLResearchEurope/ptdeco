import collections
import json
import logging
import pathlib
from typing import Any

import nvidia.dali.plugin.pytorch  # type:ignore
import ptdeco.falor
import torch

import builder
import configurator
import datasets_dali

logger = logging.getLogger(__name__)


def make_image_iterator(
    train_dataloader: collections.abc.Iterator[dict[str, torch.Tensor]],
) -> collections.abc.Iterator[torch.Tensor]:
    for d in train_dataloader:
        yield d["inputs"].permute(0, 3, 1, 2)


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
    del valid_pipeline

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
    builder.validate_module_names(model, config.blacklisted_modules)

    model.to(device)
    model_orig_stats = builder.get_model_stats(model, b_c_h_w)

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
    model_deco_stats = builder.get_model_stats(model, b_c_h_w)

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    builder.log_model_stats(logger, "Original model  :", model_orig_stats)
    builder.log_model_stats(logger, "Decomposed model:", model_deco_stats)
