import torch
from torchvision.models import resnet18, resnet101, mobilenet_v2

from efficientnet_pytorch.model import EfficientNet

from torchquant.qmodule import QModuleState
from torchquant.quantizers import AffineQuantizer
from torchquant.range_observers import BatchMinMax
from torchquant.utils import qmodule_state
from torchquant.models.efficientnet import FusedEfficientNet
from torchquant.models.mobilenet import FusedMobileNetV2
from torchquant.models.resnet import FusedResNet


def _quantizer_factory(bits):
    return lambda _: AffineQuantizer(bits, BatchMinMax())


def _get_data():
    return torch.randn((8, 3, 224, 224))


def _eval_models(fp, fused):
    x = _get_data()
    fp.eval()
    fused.eval()
    y = fp(x)

    with qmodule_state(fused, QModuleState.BYPASSED):
        y_fused_dis = fused(x)

    print(y[0, :10], y_fused_dis[0, :10])
    assert torch.allclose(y, y_fused_dis)

    # check that the forward pass doesn't randomly die
    with qmodule_state(fused, QModuleState.QUANT_AWARE_TRAIN):
        y_fused = fused(x)


def test_resnet18():
    torch.manual_seed(0)
    model = resnet18()
    q_fac = _quantizer_factory(8)
    fused = FusedResNet(model, q_fac, q_fac)
    _eval_models(model, fused)


def test_resnet101():
    torch.manual_seed(0)
    model = resnet101()
    q_fac = _quantizer_factory(8)
    fused = FusedResNet(model, q_fac, q_fac)
    _eval_models(model, fused)


def test_mobilenetv2():
    torch.manual_seed(0)
    model = mobilenet_v2()
    q_fac = _quantizer_factory(8)
    fused = FusedMobileNetV2(model, q_fac, q_fac)
    _eval_models(model, fused)


def test_efficientnet():
    torch.manual_seed(0)
    model = EfficientNet.from_name("efficientnet-b0")
    q_fac = _quantizer_factory(8)
    fused = FusedEfficientNet(model, q_fac, q_fac)
    _eval_models(model, fused)
