# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Table of contents <!-- SKIP -->
#
# <!-- TOC -->
# * [Imports and setup](#id_1)
#
# * [Helper functions](#id_2)
#
# * [CONFIGURATION](#id_3)
#
# * [Main - genterate tflites](#id_4)
#
# * [Main - generate table with results](#id_5)
# <!-- TOC -->

# %% [markdown]
# ## Imports and setup <a class="anchor" id="id_1"></a>

# %%
# #! rm nb_deco_timing_analysis.py; nbtoc.py -i nb_deco_timing_analysis.ipynb

# %%
# # ! black nb_deco_timing_analysis.py

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))

# %%
# SETUP LOGGING, something of the imports breaks it if it is not here

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    fmt = (
        "%(asctime)s.%(msecs)03d500: %(levelname).1s "
        + "%(name)s.py:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__, "datasets_dali"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


setup_logging()

# %%
from typing import Any, Optional
import collections
import json

import pathlib
import time

import tabulate
import tinynn.converter  # type:ignore
import timm
import torch


import ptdeco

# %% [markdown]
# ## Helper functions <a class="anchor" id="id_2"></a>


# %%
class NoGlobals:
    def __init__(self, logger=None):
        self.logger = logger

    @staticmethod
    def _get_global_ids():
        return [v for v in globals().keys() if not v.startswith("_")]

    def _keep_only_ids(self, ids):
        ids_all = list(globals().keys())
        for id_cur in ids_all:
            if not id_cur.startswith("_") and id_cur not in ids:
                if self.logger is not None:
                    self.logger.info("Deleting " + id_cur)
                del globals()[id_cur]

    def __enter__(self):
        self.globals = self._get_global_ids()

    def __exit__(self, type, value, traceback):
        self._keep_only_ids(self.globals)


# %%
class Timing:
    def __init__(self, block_name):
        self.block_name = block_name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        duration = time.perf_counter() - self.start
        logger.info(f"{self.block_name} took {duration:.2f} sec.")


# %%
def export_to_tflite_tinynn(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    tflite_path: pathlib.Path,
) -> None:
    # If your model is not on CPU you need to restore the device after usage of this function
    model.eval()
    model.cpu()
    dummy_input = torch.rand(input_shape)
    tflite_path.parent.mkdir(exist_ok=True, parents=True)
    converter = tinynn.converter.TFLiteConverter(model, dummy_input, str(tflite_path))
    converter.convert()


# %%
def get_state_dict(fname):
    checkpoint = torch.load(fname)
    res = collections.OrderedDict()

    for k, v in checkpoint["state"]["model"].items():
        if k.startswith("student_model."):
            res[k[14:]] = v
    return res


# %%
def export_deco_tflite_tinynn(
    *,
    model_name: str,
    deco_path: Optional[pathlib.Path] = None,
    ckpt_path: Optional[pathlib.Path] = None,
    tflite_path: pathlib.Path,
    input_shape: tuple[int, ...],
) -> None:
    if ckpt_path is not None and deco_path is not None:
        logger.info("Decomposing model")
        sd = get_state_dict(ckpt_path)
        dc = json.loads(pathlib.Path(deco_path).read_text())

        model = timm.create_model(model_name, pretrained=False)
        ptdeco.apply_decompose_config_in_place(model, dc)
        model.load_state_dict(sd)
    else:
        logger.info("Skipping decomposition")
        model = timm.create_model(model_name, pretrained=True)
    export_to_tflite_tinynn(model, input_shape, tflite_path)


# %%
def load_benchmark_data(data_path: pathlib.Path) -> dict[str, Any]:
    with open(data_path, "rt") as f:
        r = json.load(f)
    return r


# %%
def indent(s: str, indent: str) -> str:
    return "".join(f"{indent}{line}\n" for line in s.splitlines())


# %% [markdown]
# ## CONFIGURATION <a class="anchor" id="id_3"></a>

# %%
TFLITE_DIR = pathlib.Path("/nas/tmp/lopusz")

# %% [markdown]
# ## Main - genterate tflites <a class="anchor" id="id_4"></a>

# %%
model_name = "convnext_femto.d1_in1k"
root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
deco_path = (
    root_path
    / "2023-10-15_deco-cn1femto-nt010/output_decomposed_2023-10-17-11.10.41/decompose_config.json"
)
ckpt_path = (
    root_path
    / "2023-10-15_deco-cn1femto-nt010/output_decomposed_2023-10-17-11.10.41/checkpoints/ep200-ba1901600-rank0.pt"
)

suffix = "baseline"
tflite_path = TFLITE_DIR / f"{model_name}_{suffix}.tflite"

export_deco_tflite_tinynn(
    model_name=model_name, input_shape=(1, 3, 224, 224), tflite_path=tflite_path
)

suffix = "decomposed_nt010"
tflite_path = TFLITE_DIR / f"{model_name}_{suffix}.tflite"
export_deco_tflite_tinynn(
    model_name=model_name,
    deco_path=deco_path,
    ckpt_path=ckpt_path,
    input_shape=(1, 3, 224, 224),
    tflite_path=tflite_path,
)

# %% [markdown]
# ## Main - generate table with results <a class="anchor" id="id_5"></a>

# %%
runtimes = ["encore_cpu1", "encore_cpu4", "t1_pro_cpu1", "t1_pro_cpu4", "t1_pro_gpu"]
suffixes = ["baseline", "decomposed_nt010"]


for runtime in runtimes:
    res = []
    for suffix in suffixes:
        bench_json_fname = f"convnext_femto.d1_in1k_{suffix}_{runtime}_bench.json"
        bench_json_path = TFLITE_DIR / bench_json_fname
        d_bench = load_benchmark_data(bench_json_path)
        latency = d_bench["dbp_results"]["latency_avg"] / 1000.0
        latency_std = d_bench["dbp_results"]["latency_std"] / 1000.0
        res.append({"suffix": suffix, "latency": latency, "latency_std": latency_std})

    l0 = res[0]["latency"]

    for i, d in enumerate(res):
        if i == 0:
            d["speedup [%]"] = float("nan")
        else:
            d["speedup [%]"] = (l0 - d["latency"]) / l0 * 100.0

    print(f" * {model_name} - {runtime} \n")
    d_md = tabulate.tabulate(
        res,
        headers="keys",
        floatfmt=("", ".1f", ".1f", ".1f"),
        tablefmt="github",
    )
    d_md = indent(d_md.replace(" nan", "    "), "   ")
    print(d_md)
