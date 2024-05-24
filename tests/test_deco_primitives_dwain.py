import collections.abc

import pytest
import torch

import ptdeco
from ptdeco.dwain.decomposition import (
    _compute_covariance_matrix_decomposition,
    _unwrap_in_place,
    _wrap_in_place,
)

torch.set_float32_matmul_precision("highest")


def data_iterator_lin(
    bs: int, fin: int, h: int, w: int
) -> collections.abc.Generator[dict[str, torch.Tensor], None, None]:
    gen = torch.Generator()
    gen.manual_seed(1314159)
    while True:
        yield {"inp": torch.rand(bs, h, w, fin, generator=gen)}


def data_iterator_conv(
    bs: int, fin: int, h: int, w: int
) -> collections.abc.Generator[dict[str, torch.Tensor], None, None]:
    gen = torch.Generator()
    gen.manual_seed(1314159)
    while True:
        yield {"inp": torch.rand(bs, fin, h, w, generator=gen)}


class MyNetworkLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, gen: torch.Generator):
        super().__init__()
        self.mod = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.reset_parameters(gen)

    def reset_parameters(self, gen: torch.Generator) -> None:
        torch.nn.init.kaiming_uniform_(self.mod.weight, a=5**0.5, generator=gen)
        if self.mod.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mod.weight)
            bound = fan_in**-0.5 if fan_in > 0 else 0
            torch.nn.init.uniform_(self.mod.bias, -bound, bound)

    def forward(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        y = self.mod(d["inp"])
        return torch.flatten(y, start_dim=1)


class MyNetworkConv1x1(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, gen: torch.Generator):
        super().__init__()
        self.mod = torch.nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=(1, 1)
        )
        self.reset_parameters(gen)

    def reset_parameters(self, gen: torch.Generator) -> None:
        torch.nn.init.kaiming_uniform_(self.mod.weight, a=5**0.5, generator=gen)
        if self.mod.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mod.weight)
            if fan_in != 0:
                bound = fan_in**-0.5
                torch.nn.init.uniform_(self.mod.bias, -bound, bound, generator=gen)

    def forward(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        y = self.mod(d["inp"])
        return torch.flatten(y, start_dim=1)


def get_output_before_and_after_decomposition(
    *,
    root_module: torch.nn.Module,
    decomposed_submodule_name: str,
    device: torch.device,
    data_iterator: collections.abc.Iterator[dict[str, torch.Tensor]],
    deco_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = next(data_iterator)
    x = ptdeco.utils.to_device(x, device)

    with torch.no_grad():
        y0 = root_module(x)

    _wrap_in_place(root_module, decomposed_submodule_name)

    orig_weight = root_module.mod.get_weight_copy()
    with torch.no_grad():
        u = _compute_covariance_matrix_decomposition(
            root_module=root_module,
            decomposed_submodule_name=decomposed_submodule_name,
            data_iterator=data_iterator,
            weight=orig_weight,
            num_data_steps=8,
            device=device,
            decompose_in_float64=True,
        )

        uk = u[:, u.shape[1] - deco_rank :].to(torch.float)
        U, V = orig_weight.T @ uk, uk.T

        decomposed_submodule = root_module.get_submodule(decomposed_submodule_name)

        new_decomposed_submodule = decomposed_submodule.get_decomposed_module(
            u=U.T, v=V.T
        )
        new_decomposed_submodule.to(device)

    _unwrap_in_place(root_module, decomposed_submodule_name)
    ptdeco.utils.replace_submodule_in_place(
        root_module, decomposed_submodule_name, new_decomposed_submodule
    )
    with torch.no_grad():
        y1 = root_module(x)
    return y0, y1


def get_output_before_and_after_decomposition_linear(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim_in = 64
    dim_out = 32
    dim_h, dim_w = 16, 16
    gen = torch.Generator()
    gen.manual_seed(271828)
    m_lin = MyNetworkLinear(in_features=dim_in, out_features=dim_out, gen=gen)
    m_lin.to(device)

    di = data_iterator_lin(8, dim_in, dim_h, dim_w)

    y0, y1 = get_output_before_and_after_decomposition(
        root_module=m_lin,
        device=device,
        decomposed_submodule_name="mod",
        data_iterator=di,
        deco_rank=min(dim_in, dim_out),
    )
    return y0, y1


def get_output_before_and_after_decomposition_conv1x1(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim_in = 64
    dim_out = 32
    dim_h, dim_w = 16, 16

    gen = torch.Generator()
    gen.manual_seed(271828)

    m_conv = MyNetworkConv1x1(in_features=dim_in, out_features=dim_out, gen=gen)
    m_conv.to(device)

    di = data_iterator_conv(8, dim_in, dim_h, dim_w)

    y0, y1 = get_output_before_and_after_decomposition(
        root_module=m_conv,
        device=device,
        decomposed_submodule_name="mod",
        data_iterator=di,
        deco_rank=min(dim_in, dim_out),
    )
    return y0, y1


def test_linear_cpu() -> None:
    device = torch.device("cpu")
    y0, y1 = get_output_before_and_after_decomposition_linear(device)
    assert torch.abs(y0 - y1).max() < 1.0e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_linear_gpu() -> None:
    device = torch.device("cuda")
    y0, y1 = get_output_before_and_after_decomposition_linear(device)
    assert torch.abs(y0 - y1).max() < 1.0e-6


def test_conv1x1_cpu() -> None:
    device = torch.device("cpu")
    y0, y1 = get_output_before_and_after_decomposition_conv1x1(device)
    assert torch.abs(y0 - y1).max() < 1.0e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_conv1x1_gpu() -> None:
    device = torch.device("cuda")
    y0, y1 = get_output_before_and_after_decomposition_conv1x1(device)
    # TODO: Very weird that gpu nees as high threshold to work
    assert torch.abs(y0 - y1).max() < 9.0e-4
