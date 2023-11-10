import typing

import pytest
import torch

import ptdeco.lockd

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


def check_losses(model: torch.nn.Module, x: torch.Tensor) -> None:
    ptdeco.lockd.wrap_in_place(model)
    _ = model(x)

    d_entropy = ptdeco.lockd.get_entropy_dict(model)
    assert isinstance(d_entropy, dict)

    l_entropy = ptdeco.lockd.get_entropy_loss(model)
    assert isinstance(l_entropy, torch.Tensor) and l_entropy.device == x.device

    d_nsr = ptdeco.lockd.get_nsr_dict(model)
    assert isinstance(d_nsr, dict)

    l_nsr = ptdeco.lockd.get_nsr_loss(model, 0.01)
    assert isinstance(l_nsr, torch.Tensor) and l_nsr.device == x.device

    d_proportion = ptdeco.lockd.get_proportion_dict(model)
    assert isinstance(d_proportion, dict)
    l_proportion = ptdeco.lockd.get_proportion_loss(model)
    assert isinstance(l_proportion, torch.Tensor) and l_proportion.device == x.device


def check_wrap_torchvision(
    model_name: str,
    input_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    model = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    model.to(device)
    x = torch.rand(size=input_shape, device=device)
    check_losses(model, x)


@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_wrap_torchvision_cpu(model_name: str) -> None:
    check_wrap_torchvision(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_wrap_torchvision_gpu(model_name: str) -> None:
    check_wrap_torchvision(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cuda")
    )


def check_wrap_timm(
    model_name: str, input_shape: tuple[int, ...], device: torch.device
) -> None:
    model = timm.create_model(model_name, pretrained=False, num_classes=10)
    model = model.to(device)
    x = torch.rand(size=input_shape, device=device)
    check_losses(model, x)


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_wrap_timm_cpu(model_name: str) -> None:
    check_wrap_timm(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_wrap_timm_gpu(model_name: str) -> None:
    check_wrap_timm(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cuda")
    )
