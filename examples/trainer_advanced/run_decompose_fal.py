import json
import logging
import pathlib
from typing import Any

import nvidia.dali.plugin.pytorch
import ptdeco.fal
import torch

import datasets_dali
import models

logger = logging.getLogger(__name__)


def make_image_iterator(train_dataloader):
    for d in train_dataloader:
        yield d["inputs"].permute(0, 3, 1, 2)


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:
    h_w = (int(config["input_h_w"][0]), int(config["input_h_w"][1]))
    b_c_h_w = (1, 3, *h_w)
    train_pipeline, valid_pipeline = datasets_dali.make_imagenet_pipelines(
        imagenet_root_dir=config["imagenet_root_dir"],
        trn_image_classes_fname=config["trn_imagenet_classes_fname"],
        val_image_classes_fname=config["val_imagenet_classes_fname"],
        batch_size=config["batch_size"],
        normalization=config["normalization"],
        h_w=h_w,
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

    model = models.create_model(config["model_name"])
    model.to(device)
    model_orig_stats = model.get_model_stats(model, b_c_h_w)

    decompose_config = ptdeco.fal.decompose_in_place(
        module=model,
        device=device,
        data_iterator=data_iterator,
        proportion_threshold=config["decompose_proportion_threshold"],
        kl_final_threshold=config["kl_final_threshold"],
        nsr_final_threshold=config["nsr_final_threshold"],
        num_data_steps=config["num_data_steps"],
        num_metric_steps=config["num_metric_steps"],
        blacklisted_module_names=config["decompose_blacklisted_modules"],
    )
    model_deco_stats = model.get_model_stats(model, b_c_h_w)

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    model.log_model_stats(logger, "Original model  :", model_orig_stats)
    model.log_model_stats(logger, "Decomposed model:", model_deco_stats)
