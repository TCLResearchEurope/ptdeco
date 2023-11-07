import collections
import logging
from typing import Any, Optional, Union

import torch

from . import common, modconfig

logger = logging.getLogger(__name__)


def _to_str_int_tuple_int_int(o: Any) -> Union[str, int, tuple[int, int]]:
    if isinstance(o, str):
        return o
    elif isinstance(o, int):
        return o
    elif isinstance(o, tuple):
        assert len(o) == 2
        assert isinstance(o[0], int) and isinstance(o[1], int)
        return o[0], o[1]
    else:
        assert False


def _to_int_tuple_int_int(o: Any) -> Union[int, tuple[int, int]]:
    if isinstance(o, int):
        return o
    elif isinstance(o, tuple):
        assert len(o) == 2
        assert isinstance(o[0], int) and isinstance(o[1], int)
        return o[0], o[1]
    else:
        assert False


def calc_entropy_from_logits(
    logits: torch.Tensor, epsilon: float = 0.01
) -> torch.Tensor:
    probs_ = torch.sigmoid(logits)[..., None]
    probs = torch.cat([probs_, 1.0 - probs_], dim=1)
    return torch.maximum(
        -(probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon)
    )


def calc_propotion_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).mean()


def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_ = torch.cat([logits[:, None], torch.zeros_like(logits)[:, None]], dim=1)
    gs_sample = torch.nn.functional.gumbel_softmax(
        logits_,
        tau=0.5,
        hard=False,
    )[:, 0]
    return torch.where(logits < 0.0, 0.0, gs_sample)


class WrappedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_logits(self) -> torch.nn.Parameter:
        raise NotImplementedError

    def get_nsr(self) -> torch.Tensor:
        raise NotImplementedError

    def parameters_trainable(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError

    def get_decomposed_module_and_meta(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        raise NotImplementedError

    def get_orig_module(self) -> torch.nn.Module:
        raise NotImplementedError

    @classmethod
    def wrap(cls, m: torch.nn.Module, name: Optional[str] = None) -> torch.nn.Module:
        raise NotImplementedError


class WrappedConv2d(WrappedModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ks: int,
        padding: Union[str, Union[int, tuple[int, int]]],
        stride: Union[int, tuple[int, int]],
        bias: bool,
        groups: int,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.middle_features = min(out_features, in_features)
        self.conv_orig = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=ks,
            padding=padding,
            stride=stride,
            bias=bias,
            groups=groups,
        )
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=self.middle_features,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=self.middle_features,
            out_channels=out_features,
            kernel_size=ks,
            padding=padding,
            stride=stride,
            groups=groups,
            bias=bias,
        )

        self.logits = torch.nn.Parameter(
            data=3.0 * torch.ones(size=(self.middle_features,))
        )

        self.nsr = torch.tensor(0.0)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(self.name)  # TODO Remove this debug prints
        y0 = self.conv_orig(x)
        mask = sample_from_logits(self.logits)
        z = self.conv_1(x)
        z = mask.view(1, -1, 1, 1) * z
        z = self.conv_2(z)
        self.nsr = common.calc_per_channel_noise_to_signal_ratio(
            y=y0, x=z, non_channel_dim=(0, 2, 3)
        )
        return y0

    def get_nsr(self) -> torch.Tensor:
        return self.nsr

    def get_logits(self) -> torch.nn.Parameter:
        return self.logits

    def parameters_trainable(self) -> list[torch.nn.Parameter]:
        return (
            list(self.conv_1.parameters())
            + list(self.conv_2.parameters())
            + [self.logits]
        )

    def get_decomposed_module_and_meta(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        indices = torch.where(self.logits > 0)[0]
        if len(indices) == 0:
            max_logit = self.logits.max()
            indices = torch.where(self.logits >= max_logit)[0]
        c1 = len(indices)
        c0 = len(self.logits)
        p = c1 / c0 * 100.0
        logger.info(f"Leaving {c1} out of {c0} intermediate channels ({p:4.1f} %)")
        indices_conv1 = indices.view(-1, 1, 1, 1)
        indices_conv2 = indices.view(1, -1, 1, 1)
        new_weight_conv1 = torch.take_along_dim(
            self.conv_1.weight, dim=0, indices=indices_conv1
        )
        self.conv_1.weight.data = new_weight_conv1
        self.conv_1.out_channels = len(indices)
        self.conv_2.in_channels = len(indices)
        new_weight_conv2 = torch.take_along_dim(
            self.conv_2.weight, dim=1, indices=indices_conv2
        )
        self.conv_2.weight.data = new_weight_conv2
        meta = {"proportion": p}
        return torch.nn.Sequential(self.conv_1, self.conv_2), meta

    def get_orig_module(self) -> torch.nn.Module:
        return self.conv_orig

    @classmethod
    def wrap(
        cls, module: torch.nn.Module, name: Optional[str] = None
    ) -> "WrappedConv2d":
        if not isinstance(module, torch.nn.Conv2d):
            raise ValueError(f"{cls.__name__} can wrap only Conv2d not {type(module)}")

        new_module = cls(
            in_features=module.in_channels,
            out_features=module.out_channels,
            ks=module.kernel_size[0],
            padding=_to_str_int_tuple_int_int(module.padding),
            stride=_to_int_tuple_int_int(module.stride),
            bias=module.bias is not None,
            groups=module.groups,
            name=name,
        )
        new_module.conv_orig.weight.data = module.weight
        if module.bias is not None:
            assert isinstance(new_module.conv_orig.bias, torch.Tensor)
            new_module.conv_orig.bias.data = module.bias
        new_module.to(module.weight.device)
        return new_module


class WrappedLinear(WrappedModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hidden_features = min(in_features, out_features)
        self.lin_orig = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.lin_0 = torch.nn.Linear(
            in_features=in_features, out_features=self.hidden_features, bias=False
        )
        self.lin_1 = torch.nn.Linear(
            in_features=self.hidden_features, out_features=out_features, bias=bias
        )
        self.logits = torch.nn.Parameter(
            data=3.0 * torch.ones(size=(self.hidden_features,))
        )
        self.nsr = torch.tensor(0.0)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(self.name)  # TODO Remove this debug prints
        y = self.lin_orig(x)
        hidden = self.lin_0(x)
        mask = sample_from_logits(self.logits)
        hidden_masked = mask * hidden
        y0 = self.lin_1(hidden_masked)
        if len(x.shape) == 2:
            non_channel_dim: tuple[int, ...] = (0,)
        elif len(x.shape) == 3:
            non_channel_dim = (0, 1)
        elif len(x.shape) == 4:
            non_channel_dim = (0, 1, 2)
        else:
            msg = f"WrappedLinear: {x.shape=} not of length 2 or 3"
            raise NotImplementedError(msg)
        self.nsr = common.calc_per_channel_noise_to_signal_ratio(
            y=y, x=y0, non_channel_dim=non_channel_dim
        )
        return y

    def get_nsr(self) -> torch.Tensor:
        return self.nsr

    def get_logits(self) -> torch.nn.Parameter:
        return self.logits

    def parameters_trainable(self) -> list[torch.nn.Parameter]:
        return (
            list(self.lin_0.parameters())
            + list(self.lin_1.parameters())
            + [self.logits]
        )

    def get_decomposed_module_and_meta(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        indices = torch.where(self.logits > 0)[0]
        c1 = len(indices)
        c0 = len(self.logits)
        p = c1 / c0 * 100.0
        logger.info(f"Leaving {c1} out of {c0} intermediate channels ({p:4.1f} %)")
        indices_lin0 = indices.view(-1, 1)
        new_weight_lin0 = torch.take_along_dim(
            self.lin_0.weight, dim=0, indices=indices_lin0
        )
        self.lin_0.weight.data = new_weight_lin0
        indices_lin1 = indices.view(1, -1)
        new_weight_lin1 = torch.take_along_dim(
            self.lin_1.weight, dim=1, indices=indices_lin1
        )
        self.lin_1.weight.data = new_weight_lin1
        self.lin_0.out_features = len(indices)
        self.lin_1.in_features = len(indices)
        meta = {"proportion": p}
        return torch.nn.Sequential(self.lin_0, self.lin_1), meta

    def get_orig_module(self) -> torch.nn.Module:
        return self.lin_orig

    @classmethod
    def wrap(
        cls, module: torch.nn.Module, name: Optional[str] = None
    ) -> "WrappedLinear":
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"{cls.__name__} can wrap only Linear not {type(module)}")

        new_module = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
        new_module.lin_orig.weight.data = module.weight
        if module.bias is not None:
            new_module.lin_orig.bias.data = module.bias
        new_module.to(module.weight.device)
        return new_module


_WRAPPED_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)


def is_wrapped_module(m: torch.nn.Module) -> bool:
    if isinstance(m, WrappedModule):
        return True
    for m in m.modules():
        if isinstance(m, WrappedModule):
            return True
    return False


def _wrap(
    *,
    module: torch.nn.Module,
    module_path: tuple[str, ...],
    wrapped_counter: collections.Counter[str],
    blacklisted_module_names: set[str],
) -> None:
    if isinstance(module, WrappedModule):
        msg = f"{common.get_type_name(module)} cannot be wrapped in place"
        raise ValueError(msg)
    if isinstance(module, WrappedModule):
        msg = f"Model already wrapped, root module type {common.get_type_name(module)}"
        raise ValueError(msg)

    for child_name, child_module in module.named_children():
        full_child_name = ".".join((*module_path, child_name))
        if isinstance(child_module, WrappedModule):
            msg = "Model already wrapped, "
            msg += f"{full_child_name} type is {common.get_type_name(child_module)}"
            raise ValueError(msg)
        elif isinstance(child_module, _WRAPPED_MODULE_TYPES):
            child_module_type_name = common.get_type_name(child_module)

            # Skip blacklisted modules

            if full_child_name in blacklisted_module_names:
                msg = "Blacklisted - not wrrapping,"
                logger.info(f"{msg} {child_module_type_name} at {full_child_name}")
                continue

            # Handle wrapping for non-blacklisted modules

            logger.debug(f"Wrapping {child_module_type_name} at {full_child_name}")
            if isinstance(child_module, torch.nn.Conv2d):
                if child_module.groups == 1:
                    new_module_conv = WrappedConv2d.wrap(child_module, full_child_name)
                    setattr(module, child_name, new_module_conv)
            elif isinstance(child_module, torch.nn.Linear):
                new_module_linear = WrappedLinear.wrap(child_module, full_child_name)
                setattr(module, child_name, new_module_linear)
            else:
                assert False, f"Usupported type {type(child_module)}"

            wrapped_counter[child_module_type_name] += 1
        elif common.is_compound_module(child_module):
            _wrap(
                module=child_module,
                module_path=(*module_path, child_name),
                wrapped_counter=wrapped_counter,
                blacklisted_module_names=blacklisted_module_names,
            )


def wrap_in_place(
    module: torch.nn.Module,
    blacklisted_module_names: Optional[list[str]] = None,
) -> None:
    wrapped_counter: collections.Counter[str] = collections.Counter()
    if blacklisted_module_names is not None:
        blacklisted_module_names_set = set(blacklisted_module_names)
    else:
        blacklisted_module_names_set = set()
    _wrap(
        module=module,
        module_path=(),
        wrapped_counter=wrapped_counter,
        blacklisted_module_names=blacklisted_module_names_set,
    )
    for module_type_name, count in wrapped_counter.items():
        logger.info(f"Wrapped {count} instances of {module_type_name}")


def unwrap_in_place(wrapped_module: torch.nn.Module) -> None:
    raise NotImplementedError("Unrapping not implemented")  # TODO


def apply_decompose_config_in_place(
    module: torch.nn.Module,
    decompose_config: dict[str, Any],
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        device = common.get_default_device(module)

    for submodule_name, new_submodule_config in decompose_config.items():
        submodule = module.get_submodule(submodule_name)
        new_submodule = modconfig.build_module_from_config(new_submodule_config)
        new_submodule.to(device)
        common.replace_submodule_in_place(module, submodule_name, new_submodule)
        del submodule


def _decompose_in_place(
    *,
    module: torch.nn.Module,
    module_path: tuple[str, ...],
    proportion_threshold: float,
    decompose_config: dict[str, Any],
    decompose_counter: collections.Counter[str],
    blacklisted_module_names: set[str],
) -> None:
    if isinstance(module, WrappedModule):
        msg = f"{common.get_type_name(module)} cannot be wrapped in place"
        raise ValueError(msg)
    if isinstance(module, WrappedModule):
        msg = (
            f"Model already wrapped, root module type is {common.get_type_name(module)}"
        )
        raise ValueError(msg)

    for child_name, child_module in module.named_children():
        full_child_name = ".".join((*module_path, child_name))
        if isinstance(child_module, WrappedModule):
            child_module_type_name = common.get_type_name(child_module)
            logger.debug(f"Wrapping {child_module_type_name} at {full_child_name}")
            with torch.no_grad():
                p = calc_propotion_from_logits(child_module.get_logits()).item()
            msg_info = (
                f"{full_child_name} [{child_module_type_name}], proportion={p:.3f}"
            )
            blacklisted = full_child_name in blacklisted_module_names
            if not blacklisted and p < proportion_threshold:
                msg = f"Decomposing {msg_info}"
                logger.info(msg)
                new_module, meta = child_module.get_decomposed_module_and_meta()
                setattr(module, child_name, new_module)
                decompose_counter[child_module_type_name] += 1
                module_config = modconfig.get_module_config(new_module)
                module_config[modconfig.MODCONFIG_META_KEY] = meta
                decompose_config[full_child_name] = module_config
            else:
                old_module = child_module.get_orig_module()
                setattr(module, child_name, old_module)
                msg_start = "Reverting to orig module,"
                if blacklisted:
                    logger.info(f"{msg_start} blacklisted module - {msg_info}")
                else:
                    logger.info(f"{msg_start} proportion too high - {msg_info}")
        elif common.is_compound_module(child_module):
            _decompose_in_place(
                module=child_module,
                proportion_threshold=proportion_threshold,
                module_path=(*module_path, child_name),
                decompose_config=decompose_config,
                decompose_counter=decompose_counter,
                blacklisted_module_names=blacklisted_module_names,
            )


def decompose_in_place(
    module: torch.nn.Module,
    proportion_threshold: float,
    blacklisted_module_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    decompose_counter: collections.Counter[str] = collections.Counter()
    decompose_config: dict[str, Any] = {}
    if blacklisted_module_names is not None:
        blacklisted_module_names_set = set(blacklisted_module_names)
    else:
        blacklisted_module_names_set = set()

    _decompose_in_place(
        module=module,
        module_path=(),
        proportion_threshold=proportion_threshold,
        decompose_config=decompose_config,
        decompose_counter=decompose_counter,
        blacklisted_module_names=blacklisted_module_names_set,
    )
    for module_type_name, count in decompose_counter.items():
        logger.info(f"Decomposed {count} instances of {module_type_name}")
    return decompose_config


def get_entropy_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    entropy_dict = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, WrappedModule):
            entropy_dict[submodule_name] = calc_entropy_from_logits(
                submodule.get_logits()
            )
    return entropy_dict


def get_entropy_loss(wrapped_module: torch.nn.Module) -> torch.Tensor:
    entropy_list = []

    for submodule in wrapped_module.modules():
        if isinstance(submodule, WrappedModule):
            entropy_list.append(calc_entropy_from_logits(submodule.get_logits()))

    return torch.stack(entropy_list).mean()


def get_nsr_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    nsr_dict: dict[str, torch.Tensor] = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, WrappedModule):
            nsr_dict[submodule_name] = submodule.get_nsr()
    return nsr_dict


def get_nsr_loss(
    wrapped_module: torch.nn.Module, nsr_threshold: torch.Tensor
) -> torch.Tensor:
    nsr_list: list[torch.Tensor] = []
    for submodule in wrapped_module.modules():
        if isinstance(submodule, WrappedModule):
            nsr = submodule.get_nsr()
            nsr_list.append(torch.relu(nsr - nsr_threshold) / nsr_threshold)
    return torch.stack(nsr_list).mean()


def get_proportion_dict(wrapped_module: torch.nn.Module) -> dict[str, torch.Tensor]:
    proportion_dict = {}

    for submodule_name, submodule in wrapped_module.named_modules():
        if isinstance(submodule, WrappedModule):
            proportion_dict[submodule_name] = calc_propotion_from_logits(
                submodule.get_logits()
            )
    return proportion_dict


def get_proportion_loss(wrapped_module: torch.nn.Module) -> torch.Tensor:
    proportion_list: list[torch.Tensor] = []
    for submodule in wrapped_module.modules():
        if isinstance(submodule, WrappedModule):
            proportion_list.append(calc_propotion_from_logits(submodule.get_logits()))
    return torch.stack(proportion_list).mean()


def get_parameters_trainable(
    module: torch.nn.Module,
) -> list[torch.nn.Parameter]:
    parameterts_trainable = []

    for child_module in module.children():
        if isinstance(child_module, WrappedModule):
            parameterts_trainable.extend(child_module.parameters_trainable())
        elif common.is_compound_module(child_module):
            parameterts_trainable.extend(get_parameters_trainable(child_module))

    return parameterts_trainable
