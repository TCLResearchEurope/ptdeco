import typing

import helpers
import pytest
import torch

import ptdeco

if typing.TYPE_CHECKING:
    import timm  # type: ignore
    import torchvision  # type: ignore
else:
    try:
        import torchvision
    except ModuleNotFoundError:
        torchvision = None

    try:
        import timm
    except ModuleNotFoundError:
        timm = None

TORCHVISION_TEST_MODELS = ["resnet18"]

TIMM_TEST_MODELS = ["efficientformerv2_s0"]


def check_decompose(model: torch.nn.Module, x: torch.Tensor) -> None:
    y1 = model(x)
    ptdeco.wrap_in_place(model)
    helpers.set_half_logits(model)
    ptdeco.decompose_in_place(model, proportion_threshold=0.9)
    y2 = model(x)
    assert y1.shape == y2.shape


def check_decompose_torchvision(
    model_name: str,
    b_c_h_w: tuple[int, int, int, int],
    device: torch.device,
) -> None:
    model = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    model.to(device)
    x = torch.rand(size=b_c_h_w, device=device)
    check_decompose(model, x)


@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_decompose_torchvision_cpu(model_name: str) -> None:
    check_decompose_torchvision(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_decompose_torchvision_gpu(model_name: str) -> None:
    check_decompose_torchvision(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cuda")
    )


def check_decompose_timm(
    model_name: str, b_c_h_w: tuple[int, ...], device: torch.device
) -> None:
    model = timm.create_model(model_name, pretrained=False, num_classes=10)
    model = model.to(device)
    x = torch.rand(size=b_c_h_w, device=device)
    check_decompose(model, x)


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_decompose_timm_cpu(model_name: str) -> None:
    check_decompose_timm(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_decompose_timm_gpu(model_name: str) -> None:
    check_decompose_timm(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cuda")
    )
