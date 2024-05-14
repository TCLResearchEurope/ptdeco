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
    data_iterator = make_image_iterator(train_dataloader)

    model = builder.make_model(
        config.decompose_model_name, log_linears_and_conv1x1=True
    )
    builder.validate_module_names(model, config.blacklisted_modules)
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

    t_decomposition_start = time.perf_counter()
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
        use_float64=config.use_float64,
        use_mean=False,
        use_damping=True,
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
