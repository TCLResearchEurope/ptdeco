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

# %% [markdown] heading_collapsed=true
# ## Table of contents <!-- SKIP -->
#
# <!-- TOC -->
# * [Imports and setup](#id_1)
#
# * [CONFIGURATION](#id_2)
#
# * [Helper functions](#id_3)
#
# * [Main - convnextv2_nano.fcmae_ft_in22k_in1k](#id_4)
#
# * [Main - convnext_femto.d1_in1k](#id_5)
#
# * [Main - rexnetr_200.sw_in12k_ft_in1k](#id_6)
#
# * [Main - efficientformerv2_s2.snap_dist_in1k](#id_7)
#
# * [Main - mobilevitv2_200.cvnets_in22k_ft_in1k](#id_8)
#
# * [Main - timm.swinv2_cr_tiny_ns_224.sw_in1k](#id_9)
#
# * [Main - resnet18.a2_in1k (to refactor)](#id_10)
#
# * [Main - resnet50d.a1_in1k (to refactor)](#id_11)
# <!-- TOC -->

# %% [markdown]
# ## Imports and setup <a class="anchor" id="id_1"></a>

# %%
# isort: skip_file
# type: ignoresave_cache(cache_file, d)

# %%
# #! rm nb_deco_train_analysis.py; nbtoc.py -i nb_deco_train_analysis.ipynb

# %%
# #! black nb_deco_train_analysis.py

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.core.display import display, HTML, Markdown

display(HTML("<style>.container { width:80% !important; }</style>"))

# %%
import sys

# %%
sys.path.append("..")

# %%
import collections
import json
import logging
import pathlib
import time
from typing import Any, Optional


# %%
import fvcore.nn
import numpy as np
import tabulate
import torch
import timm
import torchmetrics

import nvidia.dali.plugin.pytorch

from tqdm.notebook import tqdm

# %%
import ptdeco

# %%
import datasets_dali

# %%
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


# %% [markdown]
# ## CONFIGURATION <a class="anchor" id="id_2"></a>

# %%
# IMAGENET_ROOT_DIR = "/nas/datasets/vision/ImageNet/pod_download/data/fix"
# TRN_IMAGENET_CLASSES_FNAME = (
#     "/nas/datasets/vision/ImageNet/pod_download/data/train_es.txt"
# )
# VAL_IMAGENET_CLASSES_FNAME = "/nas/datasets/vision/ImageNet/pod_download/data/val.txt"

IMAGENET_ROOT_DIR = "/home/lopusz/Datasets/datahub/vision/imagenet-alt"
TRN_IMAGENET_CLASSES_FNAME = (
    "/home/lopusz/Datasets/datahub/vision/imagenet-alt/train_es.txt"
)
VAL_IMAGENET_CLASSES_FNAME = "/home/lopusz/Datasets/datahub/vision/imagenet-alt/val.txt"

OUT_PATH = pathlib.Path("../out")
OUT_PATH.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda")

# %% [markdown]
# ## Helper functions <a class="anchor" id="id_3"></a>


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
def transpose(dis, to_array=True):
    last_keys = None
    d = {}

    for di in dis:
        if last_keys is None:
            # First row
            for k, v in di.items():
                d[k] = [v]
            last_keys = set(di.keys())
        else:
            # All but first row

            # Asserts keys in all rows are the same
            assert set(last_keys) == set(di.keys())

            for k, v in di.items():
                d[k].append(v)

    if to_array:
        for k in d:
            d[k] = np.array(d[k])
    return d


# %%
def count_modules(m):
    c = collections.Counter()
    for mo in m.modules():
        tt = type(mo).__module__ + "." + type(mo).__name__
        c[tt] += 1
    return c


# %%
def get_fpops(
    model: torch.nn.Module,
    b_c_h_w: tuple[int, int, int, int],
    units: str = "gflops",
    device: torch.device = torch.device("cpu"),
    warnings_off: bool = False,
):
    model.eval()
    x = torch.rand(size=b_c_h_w, device=device)
    fca = fvcore.nn.FlopCountAnalysis(model, x)

    if warnings_off:
        fca.unsupported_ops_warnings(False)

    # NOTE FV.CORE computes MACs not FLOPs !!!!
    # Hence 2.0 * here for proper GIGA FLOPS

    flops = 2 * fca.total()

    if units.lower() == "gflops":
        return flops / 1.0e9
    elif units.lower() == "kmapps":
        return flops / b_c_h_w[-1] / b_c_h_w[-2] / 1024.0
    raise ValueError(f"Unknown {units=}")


# %%
def get_num_params(m: torch.nn.Module, only_trainable: bool = False):
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


# %%
def validate(*, m, h_w, device, batch_size, normalization, n_batches):
    trn_pipeline, val_pipeline = datasets_dali.make_imagenet_pipelines(
        batch_size=batch_size,
        h_w=h_w,
        imagenet_root_dir=IMAGENET_ROOT_DIR,
        trn_image_classes_fname=TRN_IMAGENET_CLASSES_FNAME,
        val_image_classes_fname=VAL_IMAGENET_CLASSES_FNAME,
        normalization=normalization,
    )

    del trn_pipeline

    m.eval()

    m.to(device)

    val_iter = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            val_pipeline, ["inputs", "targets"]
        )
    )
    if n_batches is None:
        n_batches = len(val_iter)

    pbar = tqdm(total=n_batches)
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
            outputs = m(inputs)
            targets = torch.argmax(targets, dim=1)
            outputs = torch.softmax(outputs, dim=1)
            metrics.update(outputs, targets)
            pbar.update(1)
        res = metrics.compute().item()

    del val_pipeline

    return res


# %%
def get_state_dict(fname):
    checkpoint = torch.load(fname)
    res = collections.OrderedDict()

    for k, v in checkpoint["state"]["model"].items():
        if k.startswith("student_model."):
            res[k[14:]] = v
    return res


# %%
def add_stats(*, d, m, device, batch_size, normalization, h_w, n_batches):
    kmapps = get_fpops(m, (1, 3, *h_w), "kmapps")
    params = get_num_params(m)
    acc = validate(
        m=m,
        h_w=h_w,
        device=device,
        batch_size=batch_size,
        normalization=normalization,
        n_batches=n_batches,
    )
    d["params"] = params / 1.0e6
    d["kmapps"] = kmapps
    d["acc"] = acc * 100.0


# %%
def add_val_data(
    d,
    model_name,
    *,
    batch_size=50,
    ckpt_path=None,
    deco_path=None,
    normalization="imagenet",
    h_w=(224, 224),
    n_batches=None,
) -> None:
    if ckpt_path is not None and deco_path is not None:
        sd = get_state_dict(ckpt_path)
        dc = json.loads(pathlib.Path(deco_path).read_text())

        m = timm.create_model(model_name, pretrained=False)
        ptdeco.apply_decompose_config_in_place(m, dc)
        m.load_state_dict(sd)
    else:
        m = timm.create_model(model_name, pretrained=True)
    add_stats(
        d=d,
        m=m,
        device=device,
        batch_size=batch_size,
        normalization=normalization,
        h_w=h_w,
        n_batches=n_batches,
    )
    del m


# %%
def load_cache(cache_path: pathlib.Path) -> dict[str, Any]:
    if cache_path.exists():
        with open(cache_path, "rt") as f:
            return json.load(f)
    else:
        return {}


def save_cache(cache_path: pathlib.Path, d: dict[str, Any]) -> None:
    d_old = load_cache(cache_path)
    if d == d_old:
        logger.warning("New cache identical as old, SKIPPING cache saving")
    else:
        with open(cache_path, "wt") as f:
            f.write(json.dumps(d) + "\n")


# %%
def create_table(model_name, configs, force=False, tablefmt="github"):
    VAL_FIELDS = [
        "name",
        "nsr_thr",
        "h_w",
        "n_batches",
        "batch_size",
        "ckpt_path",
        "deco_path",
        "normalization",
    ]

    cache_file = OUT_PATH / (model_name + ".json")

    if force and cache_file.exists():
        cache_file.unlink()

    d = load_cache(cache_file)

    for c in configs:
        config_name = c["name"]
        if config_name not in d:
            nsr_thr = c.get("nsr_thr", float("nan"))

            # Validation params
            h_w = c.get("h_w", (224, 224))
            n_batches = c.get("n_batches")
            batch_size = c.get("batch_size", 50)
            ckpt_path = c.get("ckpt_path")
            deco_path = c.get("deco_path")
            normalization = c.get("normalization", "imagenet")

            d_new = {
                "name": config_name,
                "nsr_thr": nsr_thr,
            }
            add_val_data(
                d_new,
                model_name,
                batch_size=batch_size,
                h_w=h_w,
                ckpt_path=ckpt_path,
                deco_path=deco_path,
                normalization=normalization,
                n_batches=n_batches,
            )
        else:
            d_new = d[config_name]
            logger.warning(f"{config_name} already in cache, SKIPPING")

        for k, v in c.items():
            if k not in VAL_FIELDS:
                d_new[k] = v
        d[config_name] = d_new

    save_cache(cache_file, d)
    d_list = [dd for dd in d.values()]

    d_md = tabulate.tabulate(
        d_list,
        headers="keys",
        floatfmt=("", ".2f", ".2f", ".0f", ".2f", ".0f"),
        tablefmt=tablefmt,
    )
    return d, d_md


# %% [markdown]
# ## Main - convnextv2_nano.fcmae_ft_in22k_in1k <a class="anchor" id="id_4"></a>

# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"

    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "convnextv2_nano.fcmae_ft_in22k_in1k"

    n_batches = None
    batch_size = 50
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "epochs_ft": float("nan"),
            "method": "",
        },
        {
            "name": "decomposed_nt005",
            "nsr_thr": 0.05,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-05_deco-cn2nano-nt005/output_decomposed_2023-10-06-10.10.17/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-05_deco-cn2nano-nt005/output_decomposed_2023-10-06-10.10.17/checkpoints/ep200-ba7606800-rank0.pt"
            ),
            "epochs_ft": 200,
            "method": "lockd",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-05_deco-cn2nano-nt010/output_decomposed_2023-10-06-10.10.30/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-05_deco-cn2nano-nt010/output_decomposed_2023-10-06-10.10.30/checkpoints/ep200-ba7606800-rank0.pt"
            ),
            "epochs_ft": 200,
            "method": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - convnext_femto.d1_in1k <a class="anchor" id="id_5"></a>

# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"

    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "convnext_femto.d1_in1k"

    n_batches = None
    batch_size = 50
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "epochs_ft": float("nan"),
            "method": "",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "epochs_ft": 200,
            "method": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - rexnetr_200.sw_in12k_ft_in1k <a class="anchor" id="id_6"></a>

# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"
    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "rexnetr_200.sw_in12k_ft_in1k"

    n_batches = None
    batch_size = 50
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "epochs_ft": float("nan"),
            "mehtod": "",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-16_deco-rexnetr200-nt010/output_decomposed_2023-10-17-19.10.58/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-16_deco-rexnetr200-nt010/output_decomposed_2023-10-17-19.10.58/checkpoints/ep200-ba2535600-rank0.pt"
            ),
            "epochs_ft": 200,
            "mehtod": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - efficientformerv2_s2.snap_dist_in1k <a class="anchor" id="id_7"></a>

# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"

    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "efficientformerv2_s2.snap_dist_in1k"

    n_batches = None
    batch_size = 50
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "epochs_ft": float("nan"),
            "method": "",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-15_deco-eformer2s2-nt010/output_decomposed_2023-10-17-18.10.13/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-15_deco-eformer2s2-nt010/output_decomposed_2023-10-17-18.10.13/checkpoints/ep96-ba912768-rank0.pt"
            ),
            "epochs_ft": 96,
            "method": "lockd",
        },
        {
            "name": "decomposed_nt015",
            "nsr_thr": 0.15,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-16_deco-eformer2s2-nt015/output_decomposed_2023-10-19-20.10.19/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-16_deco-eformer2s2-nt015/output_decomposed_2023-10-19-20.10.19/checkpoints/ep145-ba1378660-rank0.pt"
            ),
            "epochs_ft": 145,
            "method": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - mobilevitv2_200.cvnets_in22k_ft_in1k <a class="anchor" id="id_8"></a>

# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"

    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "mobilevitv2_200.cvnets_in22k_ft_in1k"

    n_batches = None
    batch_size = 50
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (256, 256),
            "normalization": "zero_to_one",
            "batch_size": batch_size,
            "epochs_ft": float("nan"),
            "method": "",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (256, 256),
            "normalization": "zero_to_one",
            "batch_size": batch_size,
            "deco_path": (
                root_path
                / "2023-10-24_deco-mobilevitv2-200-nt010/output_decomposed_2023-10-26-18.10.00/decompose_config.json"
            ),
            "ckpt_path": (
                root_path
                / "2023-10-24_deco-mobilevitv2-200-nt010/output_decomposed_2023-10-26-18.10.00/checkpoints/ep105-ba3993570-rank0.pt"
            ),
            "epochs_ft": 105,
            "method": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - timm.swinv2_cr_tiny_ns_224.sw_in1k <a class="anchor" id="id_9"></a>


# %%
with NoGlobals():
    FORCE = False
    TABLEFMT = "github"

    root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")
    model_name = "swinv2_cr_tiny_ns_224.sw_in1k"

    n_batches = None
    configs = [
        {
            "name": "baseline",
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": 50,
            "epochs_ft": float("nan"),
            "method": "",
        },
        {
            "name": "decomposed_nt010",
            "nsr_thr": 0.10,
            "n_batches": n_batches,
            "h_w": (224, 224),
            "batch_size": 50,
            "deco_path": root_path
            / "2023-10-17_deco-swinv20-tiny-nt010/output_decomposed_2023-10-24-08.10.23/decompose_config.json",
            "ckpt_path": root_path
            / "2023-10-17_deco-swinv20-tiny-nt010/output_decomposed_2023-10-24-08.10.23/checkpoints/ep200-ba3042600-rank0.pt",
            "epohchs_ft": 200.0,
            "method": "lockd",
        },
    ]
    d, d_md = create_table(model_name, configs, force=FORCE, tablefmt=TABLEFMT)
    print(d_md.replace(" nan", "    "))
    display(Markdown(d_md))

# %% [markdown]
# ## Main - resnet18.a2_in1k (to refactor) <a class="anchor" id="id_10"></a>

# %%
# TO REFACTOR

# d0 = None

# with NoGlobals(), Timing("Validation"):
#     m = timm.create_model("resnet18.a2_in1k", pretrained=True)

#     d0 = {"name": "baseline", "nsr_thr": float("nan")}
#     add_stats(d0, m, device)

# d1 = None

# with NoGlobals(), Timing("Validation"):
#     CKPT_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-10-09_deco-rn18a2-nt003/output_decomposed_2023-10-10-10.10.06/checkpoints/ep151-ba2871567-rank0.pt"
#     )
#     DECO_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-10-09_deco-rn18a2-nt003/output_decomposed_2023-10-10-10.10.06/decompose_config.json"
#     )

#     sd = get_state_dict(CKPT_PATH)
#     dc = json.loads(pathlib.Path(DECO_PATH).read_text())

#     m = timm.create_model("resnet18.a2_in1k", pretrained=False)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     d1 = {"name": "decomposed_1", "nsr_thr": 0.03}
#     add_stats(d1, m, device)

# d1

# d2 = None

# with NoGlobals(), Timing("Validation"):
#     CKPT_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-10-03_deco-rn18a2-nt005/output_decomposed_2023-10-10-10.10.38/checkpoints/ep147-ba5590998-rank0.pt"
#     )
#     DECO_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-10-03_deco-rn18a2-nt005/output_decomposed_2023-10-10-10.10.38/decompose_config.json"
#     )

#     sd = get_state_dict(CKPT_PATH)
#     dc = json.loads(pathlib.Path(DECO_PATH).read_text())

#     m = timm.create_model("resnet18.a2_in1k", pretrained=False)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     d2 = {"name": "decomposed_2", "nsr_thr": 0.05}
#     add_stats(d2, m, device)

# d3 = None

# with NoGlobals(), Timing("Validation"):
#     CKPT_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-09-26_deco-rn18a2-nt010/output_decomposed_2023-10-05-14.10.35/checkpoints/ep200-ba7606800-rank0.pt"
#     )
#     DECO_PATH = pathlib.Path(
#         "/nas/people/michal_lopuszynski/JOBS/2023-09-26_deco-rn18a2-nt010/output_decomposed_2023-10-05-14.10.35/decompose_config.json"
#     )

#     sd = get_state_dict(CKPT_PATH)
#     dc = json.loads(pathlib.Path(DECO_PATH).read_text())

#     m = timm.create_model("resnet18.a2_in1k", pretrained=False)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     ptdeco.apply_decompose_config_in_place(m, dc)

#     m.load_state_dict(sd)

#     d3 = {"name": "decomposed_3", "nsr_thr": 0.10}
#     add_stats(d3, m, device)

# d = transpose([d0, d1, d2, d3])
# print(
#     tabulate.tabulate(
#         d, headers="keys", floatfmt=("", ".2f", ".2f", ".0f", ".2f"), tablefmt="github"
#     )
# )

# d3 = None
## Main
# CKPT_PATH = pathlib.Path(
#     "/nas/people/michal_lopuszynski/JOBS/2023-09-26_deco-rn18a2-nt010/output_decomposed_2023-10-05-14.10.35/checkpoints/ep200-ba7606800-rank0.pt"
# )
# DECO_PATH = pathlib.Path(
#     "/nas/people/michal_lopuszynski/JOBS/2023-09-26_deco-rn18a2-nt010/output_decomposed_2023-10-05-14.10.35/decompose_config.json"
# )

# sd = get_state_dict(CKPT_PATH)
# dc = json.loads(pathlib.Path(DECO_PATH).read_text())

# m = timm.create_model("resnet18.a2_in1k", pretrained=False)

# ptdeco.apply_decompose_config_in_place(m, dc)

# m.load_state_dict(sd)

# ptdeco.apply_decompose_config_in_place(m, dc)

# m.load_state_dict(sd)

# d3 = {"name": "decomposed_3", "nsr_thr": 0.10}
# add_stats(d3, m, device)

# %% [markdown]
# | name         |   nsr_thr |   params |   kmapps |   acc |
# |--------------|-----------|----------|----------|-------|
# | baseline     |    nan    |    11.69 |       71 | 70.87 |
# | decomposed_1 |      0.03 |     9.23 |       56 | 62.88 |
# | decomposed_2 |      0.05 |     7.77 |       46 | 62.80 |
# | decomposed_3 |      0.10 |     5.84 |       34 | 62.58 |
#

# %% [markdown]
# ## Main - resnet50d.a1_in1k (to refactor) <a class="anchor" id="id_11"></a>

# %%
# TO REFACTOR

# m = timm.create_model("resnet50d.a1_in1k", pretrained=True)

# d0 = {"name": "baseline", "nsr_thr": float("nan")}
# add_stats(d0, m, device)

# d0

# CKPT_PATH = pathlib.Path(
#     "/nas/people/michal_lopuszynski/JOBS/2023-10-05_deco-rn50da1-nt005/output_decomposed_2023-10-08-21.10.28/checkpoints/ep99-ba3765366-rank0.pt"
# )
# DECO_PATH = pathlib.Path(
#     "/nas/people/michal_lopuszynski/JOBS/2023-10-05_deco-rn50da1-nt005/output_decomposed_2023-10-08-21.10.28/decompose_config.json"
# )

# sd = get_state_dict(CKPT_PATH)
# dc = json.loads(pathlib.Path(DECO_PATH).read_text())

# m = timm.create_model("resnet50d.a1_in1k", pretrained=False)

# ptdeco.apply_decompose_config_in_place(m, dc)

# m.load_state_dict(sd)

# d1 = {"name": "decomposed_1", "nsr_thr": 0.05}
# add_stats(d1, m, device)

# d1

# d = transpose([d0, d1])

# print(
#     tabulate.tabulate(
#         d, headers="keys", floatfmt=("", ".2f", ".2f", ".0f", ".2f"), tablefmt="github"
#     )
# )

# %%
# root_path = pathlib.Path("/nas/people/michal_lopuszynski/JOBS/")

# model_name = "resnet50d.a1_in1k"

# d = []


# with Timing(f"{model_name} validation {model_name}"):

#     # Baseline

#     with NoGlobals():
#         d_new = {"name": "baseline", "nsr_thr": float("nan")}
#         add_val_data(d_new, model_name)
#         d.append(d_new)


#     # NSR 0.05

#     with NoGlobals():
#         d_new = {"name": "decomposed_1", "nsr_thr": 0.05}
#         deco_path = root_path / "2023-10-05_deco-rn50da1-nt005/output_decomposed_2023-10-08-21.10.28/decompose_config.json"
#         ckpt_path = root_path / "2023-10-05_deco-rn50da1-nt005/output_decomposed_2023-10-08-21.10.28/checkpoints/ep99-ba3765366-rank0.pt"
#         add_val_data(d_new, model_name, ckpt_path, deco_path)
#         d.append(d_new)

# #     # NSR 0.10

# #     with NoGlobals():
# #         d_new = {"name": "decomposed_2", "nsr_thr": 0.10}
# #         deco_path = root_path / "2023-10-05_deco-cn2nano-nt010/output_decomposed_2023-10-06-10.10.30/decompose_config.json"
# #         ckpt_path = root_path / "2023-10-05_deco-cn2nano-nt010/output_decomposed_2023-10-06-10.10.30/checkpoints/ep200-ba7606800-rank0.pt"
# #         add_val_data(d_new, model_name, ckpt_path, deco_path)
# #         d.append(d_new)

# %%
