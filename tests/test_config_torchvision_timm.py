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


def check_config(
    model1: torch.nn.Module, model2: torch.nn.Module, x: torch.Tensor
) -> None:
    ptdeco.lockd.wrap_in_place(model1)
    helpers.set_half_logits(model1)
    dc = ptdeco.lockd.decompose_in_place(model1, proportion_threshold=0.9)
    sd = model1.state_dict()
    model1.eval()
    with torch.no_grad():
        y1 = model1(x)

    ptdeco.utils.apply_decompose_config_in_place(model2, dc)
    model2.load_state_dict(sd, strict=True)
    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    torch.testing.assert_close(y1, y2)


def check_config_torchvision(
    model_name: str,
    b_c_h_w: tuple[int, ...],
    device: torch.device,
) -> None:
    model1 = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    model1.to(device)
    model2 = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    model2.to(device)
    x = torch.rand(size=b_c_h_w, device=device)
    check_config(model1, model2, x)


@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_config_torchvision_cpu(model_name: str) -> None:
    check_config_torchvision(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(torchvision is None, reason="torchvision not installed")
@pytest.mark.parametrize("model_name", TORCHVISION_TEST_MODELS)
def test_config_torchvision_gpu(model_name: str) -> None:
    check_config_torchvision(
        model_name, b_c_h_w=(5, 3, 224, 224), device=torch.device("cuda")
    )


def check_config_timm(
    model_name: str, input_shape: tuple[int, ...], device: torch.device
) -> None:
    model1 = timm.create_model(model_name, pretrained=False, num_classes=10)
    model1 = model1.to(device)
    model2 = timm.create_model(model_name, pretrained=False, num_classes=10)
    model2 = model2.to(device)
    x = torch.rand(size=input_shape, device=device)
    check_config(model1, model2, x)


@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_config_timm_cpu(model_name: str) -> None:
    check_config_timm(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cpu")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.skipif(timm is None, reason="timm not installed")
@pytest.mark.parametrize("model_name", TIMM_TEST_MODELS)
def test_config_timm_gpu(model_name: str) -> None:
    check_config_timm(
        model_name, input_shape=(5, 3, 224, 224), device=torch.device("cuda")
    )
