"""Implementation of the FALOR method (FAL = Features Are LOw-Rank)

Compressing transformers: features are low-rank, but weights are not!,
Hao Yu, Jianxin Wu, AAAI Conference on Artificial Intelligence (2023)


https://doi.org/10.1609/aaai.v37i9.26304

"""

import collections
import collections.abc
import logging
import time
from typing import Any, Optional

import torch

from .. import utils
from ..utils import modconfig

logger = logging.getLogger(__name__)

__all__ = ["decompose_in_place"]


class WrappedFALORModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_weight_copy(self) -> torch.Tensor:
        raise NotImplementedError()

    def set_weight(self, weights: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_last_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def get_orig_module(self) -> torch.nn.Module:
        raise NotImplementedError()

    def get_decomposed_module(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.nn.Module:
        raise NotImplementedError()


class WrappedFALORLinear(WrappedFALORModule):
    def __init__(
        self,
        lin_orig: torch.nn.Module,
        name: Optional[str] = None,
    ):
        super().__init__()
        assert isinstance(lin_orig, torch.nn.Linear)
        self.lin_orig = lin_orig
        self.input = torch.zeros(size=(0,))
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        return self.lin_orig(x)

    def get_weight_copy(self) -> torch.Tensor:
        return self.lin_orig.weight.detach().clone()

    def set_weight(self, weights: torch.Tensor) -> None:
        self.lin_orig.weight.copy_(weights)

    def get_last_input(self) -> torch.Tensor:
        return self.input.reshape(-1, self.lin_orig.in_features)

    def get_orig_module(self) -> torch.nn.Module:
        return self.lin_orig

    def get_decomposed_module(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.nn.Module:
        use_bias = self.lin_orig.bias is not None

        lin_1 = torch.nn.Linear(
            self.lin_orig.in_features, out_features=u.shape[0], bias=False
        )
        lin_2 = torch.nn.Linear(
            u.shape[0], out_features=self.lin_orig.out_features, bias=use_bias
        )

        lin_1.weight.data = u[:, :]
        lin_2.weight.data = v[:, :]
        if use_bias:
            lin_2.bias.copy_(self.lin_orig.bias)
        return torch.nn.Sequential(lin_1, lin_2)


class WrappedFALORConv2d1x1(WrappedFALORModule):
    def __init__(
        self,
        conv_orig: torch.nn.Module,
        name: Optional[str] = None,
    ):
        super().__init__()
        assert (
            isinstance(conv_orig, torch.nn.Conv2d)
            and conv_orig.kernel_size[0] == 1
            and conv_orig.kernel_size[1] == 1
            and conv_orig.groups == 1
        )
        self.conv_orig = conv_orig
        self.input = torch.zeros(size=(0,))
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        return self.conv_orig(x)

    def get_weight_copy(self) -> torch.Tensor:
        return self.conv_orig.weight.detach().data[..., 0, 0].clone()

    def set_weight(self, weights: torch.Tensor) -> None:
        self.conv_orig.weight.copy_(weights[:, :, None, None])

    def get_last_input(self) -> torch.Tensor:
        return self.input.permute(0, 2, 3, 1).reshape(-1, self.conv_orig.in_channels)

    def get_orig_module(self) -> torch.nn.Module:
        return self.conv_orig

    def get_decomposed_module(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.nn.Module:
        use_bias = self.conv_orig.bias is not None

        conv_1 = torch.nn.Conv2d(
            in_channels=self.conv_orig.in_channels,
            out_channels=u.shape[0],
            kernel_size=1,
            bias=False,
        )
        conv_2 = torch.nn.Conv2d(
            in_channels=u.shape[0],
            out_channels=self.conv_orig.out_channels,
            kernel_size=1,
            bias=use_bias,
        )
        conv_1.weight.copy_(u[:, :, None, None])
        conv_2.weight.copy_(v[:, :, None, None])
        return torch.nn.Sequential(conv_1, conv_2)


def _accumulate_Ey_and_Eyyt(
    Ey: torch.Tensor, Eyyt: torch.Tensor, weight: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x @ weight.T
    Eyyt += torch.einsum("bp,bq->pq", y, y) / y.shape[0]
    Ey += y.mean(dim=0)
    return Ey, Eyyt


def _compute_decompositon_of_covariance_matrix(
    *,
    root_module: torch.nn.Module,
    decomposed_submodule_name: str,
    data_iterator: collections.abc.Iterator[torch.Tensor],
    weight: torch.Tensor,
    num_data_steps: int,
    device: torch.device,
) -> torch.Tensor:
    root_module.eval()
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    assert isinstance(decomposed_submodule, WrappedFALORModule)

    Ey = torch.zeros(weight.shape[0]).to(device)
    Eyyt = torch.zeros((weight.shape[0], weight.shape[0])).to(device)

    for i in range(num_data_steps):
        inputs = next(data_iterator).to(device)
        _ = root_module(inputs)
        x = decomposed_submodule.get_last_input()
        Ey, Eyyt = _accumulate_Ey_and_Eyyt(Ey=Ey, Eyyt=Eyyt, weight=weight, x=x)
    Ey /= num_data_steps
    Eyyt /= num_data_steps
    cov = Eyyt - torch.outer(Ey, Ey)
    _, u = torch.linalg.eigh(cov)
    return u


def _compute_metrics(
    *,
    x: torch.Tensor,
    root_module: torch.nn.Module,
    decomposed_submodule: torch.nn.Module,
    orig_weight: torch.Tensor,
    deco_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(decomposed_submodule, WrappedFALORModule)

    root_module.eval()

    decomposed_submodule.set_weight(deco_weight)
    y_deco = root_module(x)

    decomposed_submodule.set_weight(orig_weight)
    y_orig = root_module(x)

    nsr_final = utils.calc_per_channel_noise_to_signal_ratio(
        y=y_orig, x=y_deco, non_channel_dim=(0,)
    ).mean()
    kl_final = utils.calc_kl_loss(y_deco, y_orig)
    return nsr_final, kl_final


def _wrap_in_place(
    root_module: torch.nn.Module, decomposed_submodule_name: str
) -> None:
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    if isinstance(decomposed_submodule, torch.nn.Linear):
        wrapped_decomposed_submodule: WrappedFALORModule = WrappedFALORLinear(
            decomposed_submodule, decomposed_submodule_name
        )
    elif (
        isinstance(decomposed_submodule, torch.nn.Conv2d)
        and decomposed_submodule.kernel_size[0] == 1
        and decomposed_submodule.kernel_size[1] == 1
        and decomposed_submodule.groups == 1
    ):
        wrapped_decomposed_submodule = WrappedFALORConv2d1x1(
            decomposed_submodule, decomposed_submodule_name
        )
    else:
        raise ValueError(
            f"Cannot decompose {decomposed_submodule_name}={decomposed_submodule}"
        )
    utils.replace_submodule_in_place(
        root_module, decomposed_submodule_name, wrapped_decomposed_submodule
    )


def _unwrap_in_place(
    root_module: torch.nn.Module, decomposed_submodule_name: str
) -> None:
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    assert isinstance(decomposed_submodule, WrappedFALORModule)
    orig_module = decomposed_submodule.get_orig_module()
    utils.replace_submodule_in_place(
        root_module, decomposed_submodule_name, orig_module
    )


def _process_module(
    *,
    root_module: torch.nn.Module,
    decomposed_submodule_name: str,
    data_iterator: collections.abc.Iterator[torch.Tensor],
    nsr_final_threshold: float,
    kl_final_threshold: float,
    num_data_steps: int,
    num_metric_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    decomposed_type = utils.get_type_name(decomposed_submodule)
    _wrap_in_place(root_module, decomposed_submodule_name)
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    assert isinstance(decomposed_submodule, WrappedFALORModule)
    orig_weight = decomposed_submodule.get_weight_copy()
    orig_device = orig_weight.device
    dim_out, dim_in = orig_weight.shape
    full_rank = min(dim_in, dim_out)
    msg_prefix = f"Processing {decomposed_submodule_name}:"

    if full_rank == 1:
        _unwrap_in_place(root_module, decomposed_submodule_name)
        logger.info(f"{msg_prefix} Module has rank 1, not decomposing")
        return {
            "proportion": 1.0,
            "nsr_final": 0.0,
            "kl_final": 0.0,
            "decomposed_module": None,
        }

    msg = f"{msg_prefix} {decomposed_type} weight_shape={tuple(orig_weight.shape)}"
    logger.info(msg)
    logger.info(f"{msg_prefix} {nsr_final_threshold=:.6f} {kl_final_threshold=:.6f}")

    u = _compute_decompositon_of_covariance_matrix(
        root_module=root_module,
        decomposed_submodule_name=decomposed_submodule_name,
        data_iterator=data_iterator,
        weight=orig_weight,
        num_data_steps=num_data_steps,
        device=device,
    )

    U, V = torch.empty(0), torch.empty(0)

    i = 1

    # Best rank satisfying conditions kl < kl_threshold and nsr < nsr_threshold
    rank_best = full_rank
    rank_width = full_rank // 2
    nsr_best, kl_best = 0.0, 0.0

    while rank_width > 0:
        rank_new = rank_best - rank_width
        uk = u[:, u.shape[1] - rank_new :].to(torch.float)
        U, V = orig_weight.T @ uk, uk.T
        deco_weight = (U @ V).T

        nsr_new = 0.0
        kl_new = 0.0

        for _ in range(num_metric_steps):
            x = next(data_iterator).to(device)
            nsr_sample, kl_sample = _compute_metrics(
                x=x,
                root_module=root_module,
                decomposed_submodule=decomposed_submodule,
                orig_weight=orig_weight,
                deco_weight=deco_weight,
            )
            nsr_new += nsr_sample.item()
            kl_new += kl_sample.item()
        nsr_new /= num_metric_steps
        kl_new /= num_metric_steps

        if nsr_new < nsr_final_threshold and kl_new < kl_final_threshold:
            rank_best = rank_new
            nsr_best = nsr_new
            kl_best = kl_new
        msg_iter = f"{i=} {rank_width=} {rank_new=} {nsr_new=:.6f} {kl_new=:.6f}"
        msg_cur = f"{rank_best=} {nsr_best=:.6f} {kl_best=:.6f}"
        logger.info(f"{msg_prefix} {msg_iter} {msg_cur}")
        rank_width = rank_width // 2
        i += 1
    assert U.numel() > 0 and V.numel() > 0
    decomposed_submodule.set_weight(orig_weight)

    proportion = rank_best / full_rank
    msg_metrics = f"{proportion=:.4f} nsr={nsr_best:.6f} kl={kl_new:.6f}"
    logger.info(f"{msg_prefix} iter=FINAL rank={rank_best} {msg_metrics}")

    if full_rank != rank_best:
        new_decomposed_submodule = decomposed_submodule.get_decomposed_module(
            u=U.T, v=V.T
        )
        new_decomposed_submodule.to(orig_device)
    else:
        logger.info(f"{msg_prefix} Module decomposed to full rank, not decomposing")
        new_decomposed_submodule = None

    _unwrap_in_place(root_module, decomposed_submodule_name)
    return {
        "proportion": proportion,
        "nsr_final": nsr_new,
        "kl_final": kl_new,
        "decomposed_module": new_decomposed_submodule,
    }


def _is_decomposeable_module(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear) or (
        isinstance(module, torch.nn.Conv2d)
        and module.kernel_size[0] == 1
        and module.kernel_size[1] == 1
        and module.groups == 1
    )


def _get_decomposeable_submodule_names(module: torch.nn.Module) -> list[str]:
    return [
        name for name, mod in module.named_modules() if _is_decomposeable_module(mod)
    ]


def add_meta_to_module_config(
    module_config: dict[str, Any], module_deco_results: dict[str, Any]
) -> None:
    meta = {k: v for k, v in module_deco_results.items() if k != "decomposed_module"}
    module_config[modconfig.MODCONFIG_META_KEY] = meta


def decompose_in_place(
    *,
    module: torch.nn.Module,
    device: torch.device,
    data_iterator: collections.abc.Iterator[torch.Tensor],
    blacklisted_module_names: Optional[list[str]] = None,
    proportion_threshold: float,
    nsr_final_threshold: float,
    kl_final_threshold: float,
    num_data_steps: int,
    num_metric_steps: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    results_all = {}
    decompose_config = {}

    if blacklisted_module_names is None:
        blacklisted_module_names = []

    # Prepare decomposition for each module

    decomposable_submodules = _get_decomposeable_submodule_names(module)
    n = len(decomposable_submodules)
    for i, submodule_name in enumerate(decomposable_submodules, start=1):
        msg_prefix = f"Processing {submodule_name}: module {i} of {n}"
        if submodule_name in blacklisted_module_names:
            logger.info(f"{msg_prefix}, skipped as blacklisted")
            continue
        logger.info(f"{msg_prefix}")
        with torch.no_grad():
            result = _process_module(
                root_module=module,
                decomposed_submodule_name=submodule_name,
                data_iterator=data_iterator,
                nsr_final_threshold=nsr_final_threshold,
                kl_final_threshold=kl_final_threshold,
                num_data_steps=num_data_steps,
                num_metric_steps=num_metric_steps,
                device=device,
            )
        results_all[submodule_name] = result

    # Decompose

    decompose_counter: collections.Counter[str] = collections.Counter()
    for submodule_name in decomposable_submodules:
        msg_prefix = f"Decomposing {submodule_name}:"
        if submodule_name in blacklisted_module_names:
            logger.info(f"{msg_prefix} SKIPPED blacklisted module {submodule_name}")
            continue

        assert submodule_name in results_all
        result = results_all[submodule_name]
        new_module = result["decomposed_module"]

        if new_module is None:
            logger.info(f"{msg_prefix} SKIPPED module decomposed to full rank")
            continue

        proportion = result["proportion"]
        if proportion < proportion_threshold:
            old_module = module.get_submodule(submodule_name)
            old_module_type_name = utils.get_type_name(old_module)
            utils.replace_submodule_in_place(module, submodule_name, new_module)
            module_config = modconfig.get_module_config(new_module)
            add_meta_to_module_config(module_config, result)
            decompose_config[submodule_name] = module_config
            decompose_counter[old_module_type_name] += 1
            logger.info(f"{msg_prefix} finished {proportion=:.3f}")
        else:
            msg_prop = f"{proportion=:.3f} above {proportion_threshold=:.3f}"
            logger.info(f"{msg_prefix} SKIPPED, {msg_prop}")

    for module_type_name, count in decompose_counter.items():
        logger.info(f"Decomposed {count} instances of {module_type_name}")
    logger.info(f"Total decomposable modules {len(decomposable_submodules)}")
    stop_time = time.perf_counter()

    logger.info(f"Decomposition took {stop_time-start_time:.1f} seconds")
    return decompose_config
