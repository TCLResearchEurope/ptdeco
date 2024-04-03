# Decomposing Weights Algorithm - an Iterative techNique (DWAIN)

import collections.abc
import logging
import time
from typing import Any, Optional, cast

import torch

from .. import utils

__all__ = [
    "compute_loss",
    "decompose_in_place",
]

logger = logging.getLogger(__name__)


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


class CovarianceComputingLinearModule(torch.nn.Module):
    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter] = None,
        use_float64: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.cov = torch.zeros(
            (self.out_features, self.out_features),
            device=self.weight.device,
            dtype=torch.float32,
        )
        self.num_data_steps = 0
        self.use_float64 = use_float64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        y_reshaped = y.reshape(-1, self.out_features)
        self.cov += (
            torch.einsum("bp,bq->pq", y_reshaped, y_reshaped) / y_reshaped.shape[0]
        )
        if self.bias is not None:
            y += self.bias
        self.num_data_steps += 1
        return y

    def get_eigenvectors(self) -> torch.Tensor:
        cov = self.cov / self.num_data_steps
        damp = 0.01 * torch.mean(torch.diag(cov))
        diag = torch.arange(self.cov.shape[-1], device=cov.device)
        cov[diag, diag] = cov[diag, diag] + damp

        if self.use_float64:
            cov = cov.to(torch.float64)
        else:
            cov = cov.to(torch.float32)
        _, u = torch.linalg.eigh(cov)
        return u.to(self.weight.dtype).to("cpu")


def _accumulate_Ey_and_Eyyt(
    Ey: torch.Tensor,
    Eyyt: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x @ weight.T
    if normalize:
        y /= torch.linalg.norm(y, dim=1, keepdim=True)
    Eyyt += torch.einsum("bp,bq->pq", y, y) / y.shape[0]
    Ey += y.mean(dim=0)
    return Ey, Eyyt


def _compute_covariance_matrix_decomposition(
    *,
    root_module: torch.nn.Module,
    decomposed_submodule_name: str,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    weight: torch.Tensor,
    num_data_steps: int,
    device: torch.device,
    use_mean: bool = True,
    normalize: bool = False,
    decompose_in_float64: bool,
    dampen: bool = True,
) -> torch.Tensor:
    root_module.eval()
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    assert isinstance(decomposed_submodule, WrappedFALORModule)

    if decompose_in_float64:
        logger.info("Using float64 for decomposition")
        dtype = torch.float64
    else:
        logger.info("Using float32 for decomposition")
        dtype = torch.float32

    Ey = torch.zeros(weight.shape[0], dtype=dtype).to(device)
    Eyyt = torch.zeros((weight.shape[0], weight.shape[0]), dtype=dtype).to(device)

    for _ in range(num_data_steps):
        inputs = utils.to_device(next(data_iterator), device)
        # TODO: ML check this call
        _ = root_module(inputs)
        x = decomposed_submodule.get_last_input()
        Ey, Eyyt = _accumulate_Ey_and_Eyyt(
            Ey=Ey, Eyyt=Eyyt, weight=weight, x=x, normalize=normalize
        )
    Ey /= num_data_steps
    Eyyt /= num_data_steps
    if use_mean and not dampen:
        cov = Eyyt - torch.outer(Ey, Ey)
    else:
        cov = Eyyt
    if dampen:
        damp = 0.01 * torch.mean(torch.diag(cov))
        diag = torch.arange(cov.shape[-1], device=cov.device)
        cov[diag, diag] = cov[diag, diag] + damp
    logger.info(f"Covariance matrix dtype is {cov.dtype}")
    _, u = torch.linalg.eigh(cov)
    del cov
    return u


def _compute_metrics(
    *,
    input_dict: torch.Tensor | dict[str, torch.Tensor],
    root_module: torch.nn.Module,
    decomposed_submodule: torch.nn.Module,
    orig_weight: torch.Tensor,
    deco_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(input_dict, dict)
    assert isinstance(decomposed_submodule, WrappedFALORModule)
    x = input_dict["input_ids"]
    attention_mask = input_dict["attention_mask"]

    root_module.eval()

    decomposed_submodule.set_weight(deco_weight)
    labels = x.clone()
    y_deco = root_module(input_dict)

    decomposed_submodule.set_weight(orig_weight)
    y_orig = root_module(input_dict)
    loss_deco = compute_loss(
        logits=y_deco, labels=labels, attention_mask=attention_mask
    )

    loss_orig = compute_loss(
        logits=y_orig, labels=labels, attention_mask=attention_mask
    )

    nsr_final = utils.calc_per_channel_noise_to_signal_ratio(
        y=y_orig, x=y_deco, non_channel_dim=(0, 1), mode="mean"
    )
    ppl_deco = torch.exp(loss_deco).mean()
    ppl_orig = torch.exp(loss_orig).mean()
    return nsr_final, ppl_deco, ppl_orig


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
    del decomposed_submodule


def _get_params_for_proportion(
    proportion: float,
    in_features: int,
    out_features: int,
) -> int:
    baseline = in_features * out_features
    original_rank = min(in_features, out_features)
    proposed = (in_features + out_features) * proportion * original_rank
    if proposed < baseline:
        return int(proposed)
    else:
        return baseline


def _process_module(
    *,
    root_module: torch.nn.Module,
    decomposed_submodule_name: str,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    nsr_final_threshold: float,
    ppl_diff_threshold: float,
    num_data_steps: int,
    num_metric_steps: int,
    device: torch.device,
    metric_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    num_params: int,
    min_rank: int = 32,
    trade_off_factor: float = 1.0,
    max_accepted_ppl_diff: float = 0.1,
    use_drop_in_params_heuristic: bool = True,
    decompose_in_float64: bool = True,
    u_matrix: Optional[torch.Tensor] = None,
) -> dict[str, Any]:

    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    original_module = decomposed_submodule
    orig_weight = original_module.weight.clone()
    orig_device = decomposed_submodule.weight.device
    orig_dtype = decomposed_submodule.weight.dtype
    decomposed_type = utils.get_type_name(decomposed_submodule)
    _wrap_in_place(root_module, decomposed_submodule_name)
    decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)
    assert isinstance(decomposed_submodule, WrappedFALORModule)

    dim_out, dim_in = orig_weight.shape
    full_rank = min(dim_in, dim_out)
    msg_prefix = f"Processing {decomposed_submodule_name}:"

    if full_rank == 1:
        _unwrap_in_place(root_module, decomposed_submodule_name)
        logger.info(f"{msg_prefix} Module has rank 1, not decomposing")
        return {
            "proportion": 1.0,
            "nsr_final": 0.0,
            "ppl_final": 0.0,
            "decomposed_module": None,
        }

    msg = f"{msg_prefix} {decomposed_type} weight_shape={tuple(orig_weight.shape)}"
    logger.info(msg + f" {orig_weight.dtype}")
    logger.info(f"{msg_prefix} {nsr_final_threshold=:.6f} {ppl_diff_threshold=:.6f}")

    if u_matrix is not None:
        logger.info("Using pre-computed u_matrix")
    else:
        u_matrix = _compute_covariance_matrix_decomposition(
            root_module=root_module,
            decomposed_submodule_name=decomposed_submodule_name,
            data_iterator=data_iterator,
            weight=orig_weight,
            num_data_steps=num_data_steps,
            device=device,
            use_mean=False,
            normalize=False,
            decompose_in_float64=decompose_in_float64,
        )

    U, V = torch.empty(0), torch.empty(0)

    i = 1

    # Best rank satisfying conditions kl < kl_threshold and nsr < nsr_threshold
    rank_best = full_rank
    rank_new = full_rank
    nsr_best, ppl_best = 0.0, 0.0
    skip = False
    drop_in_params = 0

    if not use_drop_in_params_heuristic:
        min_rank = full_rank // 4

    else:
        min_rank = min_rank

    step_size = 256

    while rank_new > min_rank:
        if use_drop_in_params_heuristic:
            rank_new = rank_new // 2
        else:
            rank_new = rank_new - step_size

        if use_drop_in_params_heuristic:
            previous_params_in_module = _get_params_for_proportion(1.0, dim_in, dim_out)
            current_params_in_module = _get_params_for_proportion(
                rank_new / full_rank, dim_in, dim_out
            )
            drop_in_params = previous_params_in_module - current_params_in_module
            fraction_of_params_to_be_removed = drop_in_params / num_params
            ppl_diff_threshold = fraction_of_params_to_be_removed * trade_off_factor
        else:
            previous_params_in_module = 1.0 * dim_out * dim_in
            current_params_in_module = (rank_new / full_rank) * dim_in * dim_out
            drop_in_params = previous_params_in_module - current_params_in_module
            fraction_of_params_to_be_removed = drop_in_params / num_params
            ppl_diff_threshold = 0.05

        if drop_in_params == 0:
            continue

        # TODO: ML U@V in full precision and then cast to orig_dtype
        uk_matrix = (
            u_matrix[:, u_matrix.shape[1] - rank_new :].to(orig_dtype).to(device)
        )
        U = orig_weight.T @ uk_matrix
        V = uk_matrix.T
        deco_weight = (U @ V).T

        nsr_new = 0.0
        ppl_new = 0.0

        logger.warning(
            f"Current ppl diff threshold: {ppl_diff_threshold}, fraction of params "
            f"that can be removed: {fraction_of_params_to_be_removed}"
        )

        for _ in range(num_metric_steps):
            input_dict = utils.to_device(next(metric_iterator), device)
            nsr_sample, ppl_deco, ppl_orig = _compute_metrics(
                input_dict=input_dict,
                root_module=root_module,
                decomposed_submodule=decomposed_submodule,
                orig_weight=orig_weight,
                deco_weight=deco_weight,
            )
            ppl_diff_sample = (ppl_deco - ppl_orig) / ppl_orig
            nsr_new += nsr_sample.item()
            ppl_new += ppl_diff_sample.item()
        nsr_new /= num_metric_steps
        ppl_new /= num_metric_steps

        if (
            nsr_new < nsr_final_threshold
            and ppl_new < ppl_diff_threshold
            and ppl_new < max_accepted_ppl_diff
        ):
            rank_best = rank_new
            nsr_best = nsr_new
            ppl_best = ppl_new
            logger.info(f"Accepting rank {rank_best}/{full_rank}")
        # else:
        #     break

        msg_iter = f"{i=} {rank_new=} {nsr_new=:.6f} {ppl_new=:.6f} "
        msg_cur = f"{rank_best=} {nsr_best=:.6f} {ppl_best=:.6f}"
        logger.info(f"{msg_prefix} {msg_iter} {msg_cur}")
        logger.info(f"deco ppl: {ppl_deco}, orig ppl: {ppl_orig}")
        i += 1
    assert U.numel() > 0 and V.numel() > 0
    # decomposed_submodule.set_weight(orig_weight)

    proportion = rank_best / full_rank
    msg_metrics = f"{proportion=:.4f} nsr={nsr_best:.6f} ppl={ppl_best:.6f}"
    logger.info(f"{msg_prefix} iter=FINAL rank={rank_best} {msg_metrics}")

    decompose_decision = _check_if_decompose(
        proportion=proportion,
        in_features=dim_in,
        out_features=dim_out,
    )

    if full_rank != rank_best and not skip and decompose_decision:
        uk_matrix = (
            u_matrix[:, u_matrix.shape[1] - rank_best :].to(orig_dtype).to(device)
        )
        U, V = orig_weight.T @ uk_matrix, uk_matrix.T
        new_decomposed_submodule = decomposed_submodule.get_decomposed_module(
            u=U.T, v=V.T
        )
        new_decomposed_submodule.to(orig_device)
        new_decomposed_submodule.to(orig_dtype)
    else:
        logger.info(f"{msg_prefix} Module decomposed to full rank, not decomposing")
        new_decomposed_submodule = None
        _unwrap_in_place(root_module, decomposed_submodule_name)

    previous_params_in_module = _get_params_for_proportion(1.0, dim_in, dim_out)
    current_params_in_module = _get_params_for_proportion(proportion, dim_in, dim_out)
    if decompose_decision:
        drop_in_params = previous_params_in_module - current_params_in_module
    else:
        drop_in_params = 0

    return {
        "proportion": proportion,
        "nsr_final": nsr_new,
        "ppl_final": ppl_new,
        "decomposed_module": new_decomposed_submodule,
        "drop_in_params": drop_in_params,
    }


def _is_decomposeable_module(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear) or (
        isinstance(module, torch.nn.Conv2d)
        and module.kernel_size[0] == 1
        and module.kernel_size[1] == 1
        and module.groups == 1
    )


def _get_decomposeable_submodule_names(
    module: torch.nn.Module, blacklisted_module_names: list[str]
) -> list[str]:
    res = []
    for name, mod in module.named_modules():
        if _is_decomposeable_module(mod):
            if name in blacklisted_module_names:
                logger.info(f"Skipping blacklisted module {name}")
            else:
                res.append(name)
    return res


def _add_meta_to_module_config(
    module_config: dict[str, Any], module_deco_results: dict[str, Any]
) -> None:
    meta = {k: v for k, v in module_deco_results.items() if k != "decomposed_module"}
    module_config[utils.modconfig.MODCONFIG_META_KEY] = meta


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    loss_fc = torch.nn.CrossEntropyLoss()
    labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    attention_mask = attention_mask[..., :-1]

    # ignore padding tokens when computing the loss
    logits = logits * attention_mask.unsqueeze(-1)

    loss = loss_fc(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss


def _check_if_decompose(
    proportion: float,
    in_features: int,
    out_features: int,
) -> bool:
    baseline = in_features * out_features
    original_rank = min(in_features, out_features)
    proposed = (in_features + out_features) * proportion * original_rank
    return proposed < baseline


def _precompute_covariance_matrix_decompositions(
    module: torch.nn.Module,
    submodule_names: list[str],
    num_data_steps: int,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    # replace all layers to be decomposed by covariance computing ones
    original_linears_dict = {}

    for submodule_name in submodule_names:
        old_module = module.get_submodule(submodule_name)
        original_linears_dict[submodule_name] = old_module
        new_module = CovarianceComputingLinearModule(
            weight=old_module.weight,
            bias=old_module.bias,
        )
        logger.info(f"Replacing {submodule_name} by covariance computing wrapper.")
        utils.replace_submodule_in_place(
            root_module=module, submodule_name=submodule_name, new_submodule=new_module
        )

    module.eval()

    with torch.no_grad():
        for _ in range(num_data_steps):
            input_dict = utils.to_device(next(data_iterator), device)
            # TODO: ML remove this cast
            input_dict = cast(dict[str, torch.Tensor], input_dict)
            input_ids = input_dict["input_ids"]
            # labels = input_dict["labels"]
            # _ = module(input_ids, labels=labels)
            _ = module({"input_ids": input_ids})

    utils.free_gpu_reserved_memory()

    # compute eigenvectors
    logger.info("Computing eigenvectors ...")
    u_dict = {}

    for submodule_name in submodule_names:
        submodule = module.get_submodule(submodule_name)
        u = submodule.get_eigenvectors()
        u_dict[submodule_name] = u

    # revert to original linears
    for submodule_name in submodule_names:
        logger.info(f"Replacing {submodule_name} by original linear.")
        utils.replace_submodule_in_place(
            root_module=module,
            submodule_name=submodule_name,
            new_submodule=original_linears_dict[submodule_name],
        )

    utils.free_gpu_reserved_memory()
    return u_dict


def _precompute_covariance_matrix_decompositions_in_splits(
    module: torch.nn.Module,
    modules_to_decompose: list[str],
    num_splits: int,
    num_data_steps: int,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    u_dicts = []
    # Splitting prevents OOM
    chunk_size = len(modules_to_decompose) // num_splits
    if chunk_size == 0:
        # Special case if num_splits > len(modules_to_decompose)
        chunk_size = 1
        num_splits = len(modules_to_decompose)

    num_partitions = (
        num_splits if len(modules_to_decompose) % num_splits == 0 else num_splits + 1
    )
    for partition_index in range(num_partitions):
        sublist = modules_to_decompose[
            partition_index * chunk_size : (partition_index + 1) * chunk_size
        ]
        logger.info(f"Pre computing covariance matrices for {len(sublist)} modules")
        u_dicts.append(
            _precompute_covariance_matrix_decompositions(
                module=module,
                submodule_names=sublist,
                num_data_steps=num_data_steps,
                data_iterator=data_iterator,
                device=device,
            )
        )
    u_dict = {k: v for d in u_dicts for k, v in d.items()}
    assert len(u_dict) == len(modules_to_decompose)
    return u_dict


def decompose_in_place(
    *,
    module: torch.nn.Module,
    device: torch.device,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    num_data_steps: int,
    metric_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    num_metric_steps: int,
    blacklisted_module_names: Optional[list[str]] = None,
    nsr_final_threshold: float,
    ppl_diff_threshold: float,
    finetune_fn: collections.abc.Callable[
        [torch.nn.Module, torch.device, list[str]], torch.nn.Module
    ],
    # num_ft_steps: int,
    # ft_lr: float = 0.0001,
    # run_finetuning: bool = True,
    # lora_finetuning: bool = False,
    # num_last_decomposed_layers_to_finetune: int = 8,
    # use_rank_pattern: bool = False,
    # ft_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    min_rank: int = 32,
    # dtype: torch.dtype,
    trade_off_factor: float = 0.5,
    decompose_in_float64: bool = True,
    precomputing_covariance_num_splits: int = 0,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    num_params = utils.get_num_params(module)
    current_params = num_params

    if blacklisted_module_names is None:
        blacklisted_module_names = []
    modules_to_decompose = _get_decomposeable_submodule_names(
        module, blacklisted_module_names
    )
    n = len(modules_to_decompose)

    msgs = [f"There are {n} linear modules that can be decomposed:"]
    for i, module_name in enumerate(modules_to_decompose, start=1):
        msgs.append(f"  {i}. {module_name}")
    logger.info("\n".join(msgs))

    decompose_config = {}
    decomposed_submodules = []

    if precomputing_covariance_num_splits > 0:
        u_dict = _precompute_covariance_matrix_decompositions_in_splits(
            module=module,
            modules_to_decompose=modules_to_decompose,
            num_splits=precomputing_covariance_num_splits,
            data_iterator=data_iterator,
            num_data_steps=num_data_steps,
            device=device,
        )
    else:
        logger.info("Skipping precomputing convariance matrices")
        u_dict = {}

    utils.free_gpu_reserved_memory()

    for i, submodule_name in enumerate(reversed(modules_to_decompose)):
        msg_prefix = f"{submodule_name}: module {i} of {n}"
        logger.info(f"{msg_prefix}, processing")
        with torch.no_grad():
            old_module = module.get_submodule(submodule_name)
            msg = f"{submodule_name} START MEM={utils.get_gpu_reserved_memory_gb():.2f}"
            logger.info(msg)
            result = _process_module(
                root_module=module,
                decomposed_submodule_name=submodule_name,
                data_iterator=data_iterator,
                metric_iterator=metric_iterator,
                nsr_final_threshold=nsr_final_threshold,
                ppl_diff_threshold=ppl_diff_threshold,
                num_data_steps=num_data_steps,
                num_metric_steps=num_metric_steps,
                device=device,
                num_params=num_params,
                trade_off_factor=trade_off_factor,
                min_rank=min_rank,
                use_drop_in_params_heuristic=True,
                decompose_in_float64=decompose_in_float64,
                u_matrix=u_dict.pop(submodule_name) if len(u_dict) > 0 else None,
            )
            msg = f"{submodule_name} STOP MEM={utils.get_gpu_reserved_memory_gb():.2f}"
            logger.info(msg)
        current_params -= result["drop_in_params"]
        logger.info(f"Current params in M: {current_params / 1e6}")
        new_module = result["decomposed_module"]

        if new_module is None:
            logger.info(f"{msg_prefix}, skipped as decomposed to full rank")
            continue

        proportion = result["proportion"]
        if _check_if_decompose(
            proportion=proportion,
            in_features=old_module.in_features,
            out_features=old_module.out_features,
        ):
            decomposed_submodules.append(submodule_name)
            utils.replace_submodule_in_place(module, submodule_name, new_module)
            num_decomposed_layers = len(decomposed_submodules)
            if num_decomposed_layers > 0:
                module = finetune_fn(
                    module,
                    device,
                    decomposed_submodules,
                )
                utils.free_gpu_reserved_memory()

            # if num_decomposed_layers > 0 and run_finetuning:
            #     if lora_finetuning:
            #         n_layers = num_last_decomposed_layers_to_finetune
            #         module = _lora_finetune_decomposed_layers(
            #             model=module,
            #             device=device,
            #             ft_iterator=ft_iterator,
            #             decomposed_submodules=decomposed_submodules,
            #             num_last_decomposed_layers_to_finetune=n_layers,
            #             lr=ft_lr,
            #             num_steps=num_ft_steps,
            #             use_rank_pattern=use_rank_pattern,
            #         )
            #         utils.free_gpu_reserved_memory()
            #     else:
            #         module = _finetune_decomposed_layers(
            #             model=module,
            #             device=device,
            #             ft_iterator=ft_iterator,
            #             decomposed_submodules=decomposed_submodules,
            module_config = utils.get_module_config(new_module)
            _add_meta_to_module_config(module_config, result)
            decompose_config[submodule_name] = module_config
            logger.info(
                f"Decomposed {submodule_name}, with rank proportion: {proportion}"
            )
        utils.free_gpu_reserved_memory()

    stop_time = time.perf_counter()

    logger.info(f"Decomposition took {stop_time - start_time:.1f} seconds")
    return decompose_config
