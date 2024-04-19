import collections
import json
import logging
import pathlib
import time
from typing import Any

import nvidia.dali.plugin.pytorch  # type:ignore
import ptdeco.falor
import torch


import builder
import configurator
import datasets_dali
import metrics

logger = logging.getLogger(__name__)


def make_image_iterator(
    train_dataloader: collections.abc.Iterator[dict[str, torch.Tensor]],
) -> collections.abc.Iterator[dict[str, torch.Tensor]]:
    for d in train_dataloader:
        d["inputs"] = d["inputs"].permute(0, 3, 1, 2)
        yield d


def no_finetune(
    m: torch.nn.Module, device: torch.device, decomposed_modules: list[str]
) -> torch.nn.Module:
    return m


def ce_loss(input_dict: dict[str, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
    target = input_dict["targets"]
    return torch.nn.functional.cross_entropy(input=output, target=target)


class WrapperModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.raw_model = model

    def forward(self, x: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self.raw_model(x["inputs"])


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    config = configurator.DecomposeDWAINConfig(**config_raw)
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
    val_dataloader = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            valid_pipeline, ["inputs", "targets"]
        )
    )

    decomposition_it = make_image_iterator(train_dataloader)

    model = builder.make_model(config.decompose_model_name, log_linears_an_conv1x1=True)
    builder.validate_module_names(model, config.blacklisted_module_names)
    model.to(device)

    t_eval_start = time.perf_counter()
    accuracy_val_initial = 100.0 * metrics.calc_accuracy(
        model=model,
        val_iter=val_dataloader,
        device=device,
    )
    t_eval_intial = time.perf_counter() - t_eval_start

    stats_initial = builder.get_model_stats(model, b_c_h_w)
    stats_initial.update(
        builder.get_decomposeable_model_stats(
            model, b_c_h_w, ptdeco.falor.is_decomposeable_module
        )
    )
    stats_initial["accuracy_val"] = accuracy_val_initial
    builder.log_model_stats(logger, "Original model:", stats_initial)

    model_wrapped = WrapperModule(model)

    t_decomposition_start = time.perf_counter()
    decompose_config = ptdeco.dwain.decompose_in_place(
        module=model_wrapped,
        device=device,
        blacklisted_module_names=config.blacklisted_module_names,
        data_iterator=decomposition_it,
        loss_fn=ce_loss,
        finetune_fn=no_finetune,
        metric_iterator=decomposition_it,
        nsr_final_threshold=config.nsr_final_threshold,
        ppl_diff_threshold=config.ppl_diff_threshold,
        num_data_steps=config.num_data_steps,
        num_metric_steps=config.num_metric_steps,
        min_rank=config.min_rank,
        trade_off_factor=config.trade_off_factor,
        decompose_in_float64=config.decompose_in_float64,
        precomputing_covariance_num_splits=config.precomputing_covariance_num_splits,
    )

    t_decomposition = time.perf_counter() - t_decomposition_start

    stats_final = builder.get_model_stats(model, b_c_h_w)
    t_eval_start = time.perf_counter()
    accuracy_val_final = 100.0 * metrics.calc_accuracy(
        model=model,
        val_iter=val_dataloader,
        device=device,
    )
    t_eval_final = time.perf_counter() - t_eval_start
    s = f"Final accuracy {accuracy_val_final:.2f},  eval took  {t_eval_final:.2f} s"
    logger.info(s)
    stats_final["accuracy_val"] = accuracy_val_final

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    builder.log_model_stats(logger, "Original model  :", stats_initial)
    builder.log_model_stats(logger, "Decomposed model:", stats_final)

    device_str = str(device)
    if "cuda" in device_str:
        device_str += " @ " + torch.cuda.get_device_name(device)

    summary = {
        "accuracy_val_initial": accuracy_val_initial,
        "accuracy_val_final": accuracy_val_final,
        "mparams_initial": stats_initial["mparams"],
        # number of parameters in decomposeable operations
        "mparams_initial_decomposeable": stats_initial["mparams_decomposeable"],
        "mparams_final": stats_final["mparams"],
        "mparams_frac": stats_final["mparams"] / stats_initial["mparams"] * 100.0,
        "gflops_initial": stats_initial["gflops"],
        # number of gflops in decomposeable operations
        "gflops_initial_decomposeable": stats_initial["gflops_decomposeable"],
        "gflops_final": stats_final["gflops"],
        "gflops_frac": stats_final["gflops"] / stats_initial["gflops"] * 100.0,
        "kmapps_initial": stats_initial["kmapps"],
        "kmapps_finall": stats_final["kmapps"],
        # Should be the same as "gflops_frac", but we log it for completeness
        "kmapps_frac": stats_final["kmapps"] / stats_initial["kmapps"] * 100.0,
        "time_eval_initial": t_eval_intial,
        "time_decomposition": t_decomposition,
        "time_eval_final": t_eval_final,
        "device": device_str,
    }

    with open(output_path / "summary.json", "wt") as f:
        json.dump(summary, f)
