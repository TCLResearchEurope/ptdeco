import json
import pathlib
from typing import Any

import nvidia.dali.plugin.pytorch
import ptdeco.fal
import torch

import datasets_dali
import models


def make_image_iterator(train_dataloader):
    for d in train_dataloader:
        yield d["inputs"].permute(0, 3, 1, 2)


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:
    train_pipeline, valid_pipeline = datasets_dali.make_imagenet_pipelines(
        imagenet_root_dir=config["imagenet_root_dir"],
        trn_image_classes_fname=config["trn_imagenet_classes_fname"],
        val_image_classes_fname=config["val_imagenet_classes_fname"],
        batch_size=config["batch_size"],
        normalization=config["normalization"],
        h_w=(int(config["input_h_w"][0]), int(config["input_h_w"][1])),
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

    model = models.create_model(config)
    model.to(device)
    decompose_config = ptdeco.fal.decompose_in_place(
        module=model,
        device=device,
        data_iterator=data_iterator,
    )

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)
