import torch

import ptdeco
from ptdeco.falor.decomposition import (
    _compute_decompositon_of_covariance_matrix,
    _unwrap_in_place,
    _wrap_in_place,
)


def data_iterator_lin(bs, fin, h, w):
    gen = torch.Generator()
    gen.manual_seed(314159)
    while True:
        yield torch.rand(bs, h, w, fin, generator=gen)


def data_iterator_conv(bs, fin, h, w):
    gen = torch.Generator()
    gen.manual_seed(314159)
    while True:
        yield torch.rand(bs, fin, h, w, generator=gen)


class MyNetworkLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mod = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        y = self.mod(x)
        return torch.flatten(y, start_dim=1)


class MyNetworkConv1x1(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mod = torch.nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=(1, 1)
        )

    def forward(self, x):
        y = self.mod(x)
        return torch.flatten(y, start_dim=1)


def get_output_before_and_after_decomposition(
    *, root_module, decomposed_submodule_name, device, data_iterator, deco_rank
):
    x = next(data_iterator)
    with torch.no_grad():
        y0 = root_module(x)

    _wrap_in_place(root_module, decomposed_submodule_name)

    orig_weight = root_module.mod.get_weight_copy()
    with torch.no_grad():
        u = _compute_decompositon_of_covariance_matrix(
            root_module=root_module,
            decomposed_submodule_name=decomposed_submodule_name,
            data_iterator=data_iterator,
            weight=orig_weight,
            num_data_steps=8,
            device=device,
            use_float64=True,
            use_mean=False,
            use_damping=True,
        )

        deco_rank = 32
        uk = u[:, u.shape[1] - deco_rank :].to(torch.float)
        U, V = orig_weight.T @ uk, uk.T
        # deco_weight = (U @ V).T

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


def test_linear_decomposition_cpu():
    dim_in = 64
    dim_out = 32
    dim_h, dim_w = 16, 16

    device = torch.device("cpu")

    m_lin = MyNetworkLinear(in_features=dim_in, out_features=dim_out)
    m_lin.to(device)

    di = data_iterator_lin(8, dim_in, dim_h, dim_w)

    y0, y1 = get_output_before_and_after_decomposition(
        root_module=m_lin,
        device=device,
        decomposed_submodule_name="mod",
        data_iterator=di,
        deco_rank=min(dim_in, dim_out),
    )

    assert torch.max(torch.abs(y0 - y1).mean()) < 1.0e-6


def test_conv1x1_decomposition_cpu():
    dim_in = 64
    dim_out = 32
    dim_h, dim_w = 16, 16

    device = torch.device("cpu")

    m_conv = MyNetworkConv1x1(in_features=dim_in, out_features=dim_out)
    m_conv.to(device)

    di = data_iterator_conv(8, dim_in, dim_h, dim_w)

    y0, y1 = get_output_before_and_after_decomposition(
        root_module=m_conv,
        device=device,
        decomposed_submodule_name="mod",
        data_iterator=di,
        deco_rank=min(dim_in, dim_out),
    )

    assert torch.max(torch.abs(y0 - y1).mean()) < 1.0e-6
