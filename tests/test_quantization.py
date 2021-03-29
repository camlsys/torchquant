import torch
import torch.nn as nn
import torch.nn.functional as F

from torchquant.quantizers_functional import affine_quantize, gradient_clip
from torchquant.quantizers import AffineQuantizer, QfmtQuantizer
from torchquant.qmodule import QWrapper, QModuleState 
from torchquant.range_observers import BatchMinMax, ExpAvgMinMax
from torchquant.utils import qmodule_state


def test_affine_tensor_1():
    x = torch.tensor([3.9, 7.2, -10])

    n_bits = 4
    delta = (x.max() - x.min()).float() / ((2 ** n_bits) - 1)
    zero_point = torch.round(-x.min() / delta)

    x_q = affine_quantize(x, delta, zero_point, 2 ** n_bits)

    expected_x_q = torch.tensor([3.4400, 6.8800, -10.3200])
    print(x_q, expected_x_q, x_q.dtype, expected_x_q.dtype, torch.eq(x_q, expected_x_q))
    assert torch.all(torch.eq(x_q, x_q))


def test_affine_quantizer_1():
    """Test that AffineQuantizer with BatchMinMax() returns the correct tensor"""
    x = torch.tensor([3.9, 7.2, -10])

    quantizer = AffineQuantizer(4, BatchMinMax())

    quantizer.pre_observe(x)
    x_q = quantizer(x)

    n_bits = 4
    delta = (x.max() - x.min()).float() / ((2 ** n_bits) - 1)
    zero_point = torch.round(-x.min() / delta)
    expected_x_q = affine_quantize(x, delta, zero_point, 2 ** n_bits)

    assert torch.all(torch.eq(expected_x_q, x_q))


def test_affine_quantizer_symmetric():
    """Test that AffineQuantizer with BatchMinMax() returns the correct tensor"""
    x = torch.tensor([3.9, 7.2, -10])

    quantizer = AffineQuantizer(4, BatchMinMax(), symmetric=True)

    quantizer.pre_observe(x)
    x_q = quantizer(x)

    n_bits = 4
    delta = (10.0 - (-10.0)) / ((2 ** n_bits) - 1)
    zero_point = torch.round(-x.min() / delta)
    expected_x_q = affine_quantize(x, delta, zero_point, 2 ** n_bits)

    print("boo", x_q, expected_x_q)

    assert torch.all(torch.eq(expected_x_q, x_q))


def test_qfmt_basic():
    x = torch.tensor([3.9, 7.2, -10])
    expected_x_q = torch.tensor([4.0, 8.0, -10.0])

    quantizer = QfmtQuantizer(4, range_observer=BatchMinMax(), signed=True)

    quantizer.pre_observe(x)
    x_q = quantizer(x)

    assert torch.all(torch.eq(expected_x_q, x_q))


def test_qfmt_basic_signed():
    x = torch.tensor([3.9, 7.2, 10.3])
    expected_x_q = torch.tensor([4.0, 7.0, 10.0])

    quantizer = QfmtQuantizer(4, range_observer=BatchMinMax(), signed=False)

    quantizer.pre_observe(x)
    x_q = quantizer(x)

    assert torch.all(torch.eq(expected_x_q, x_q))


# TODO: Per-channel quantization

# TODO: Add a test for symmetric affine quantization

# TODO: Add a test for one-sided ranges

# TODO Add a test for detecting wrong min/max ranges

# TODO: Add fused-level tests


def _make_qwrapper(net):
    return QWrapper(
        net,
        weight_quantizer=AffineQuantizer(8, BatchMinMax()),
        acts_quantizer=AffineQuantizer(8, ExpAvgMinMax()),
    )


def test_fused_1():
    # by default QWrapper should add the hooks but quantization is off.
    net = nn.Sequential(nn.Linear(10, 20), nn.ReLU())

    x = torch.randn(64, 10)
    y = net(x)

    net_q = _make_qwrapper(net)
    y_q = net_q(x)

    assert torch.eq(y, y_q).all()


def test_fused_2():
    net = nn.Sequential(nn.Linear(5, 10), nn.ReLU())

    x = torch.randn(2, 5)
    y = net(x)

    net_q = _make_qwrapper(net)

    with qmodule_state(net_q, QModuleState.QUANT_AWARE_TRAIN):
        y_q = net_q(x)

    assert not torch.eq(y, y_q).all()


def test_fused_3():
    """Make sure we can detect unsupported patterns"""

    net = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 5))

    try:
        net_q = _make_qwrapper(net)
        assert False
    except TypeError:
        assert True


class Q_LeNet_300_100(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = _make_qwrapper(nn.Linear(784, 300))
        self.fc2 = _make_qwrapper(nn.Linear(300, 100))
        self.fc3 = _make_qwrapper(nn.Linear(100, 10))

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


def test_fused_4():
    """
    make sure we can make a simple network quantizable()
    """

    net = Q_LeNet_300_100()
    x = torch.randn(256, 28, 28, 1)
    y = net(x)

    with qmodule_state(net, QModuleState.QUANT_AWARE_TRAIN):
        y_q = net(x)

    assert not torch.eq(y_q, y).all()


def test_gradient_clipping():
    """Ensure that gradient clipping does what it should"""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    x.requires_grad = True
    range_min = torch.tensor(-1.0)
    range_max = torch.tensor(1.0)

    grad = torch.ones_like(x)
    y = gradient_clip(x, range_min, range_max)
    y.backward(gradient=grad)

    # note that the inequality is strict (i.e. shouldn't clip grads at +/- 1.0)
    assert torch.allclose(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))
