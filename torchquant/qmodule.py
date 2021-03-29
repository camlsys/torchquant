from enum import Enum
import math
from typing import Callable, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch.utils import (
    MemoryEfficientSwish,
    Conv2dDynamicSamePadding,
    Conv2dStaticSamePadding,
)

from torchquant.quantizers import Quantizer

__all__ = [
    "QModule",
    "QWrapper",
    "QOp",
    "QModuleState",
]

_SUPPORTED_ACTS = [nn.ReLU, nn.ReLU6, MemoryEfficientSwish]

_SUPPORTED_PATTERNS_BASE = [
    [nn.Linear],
    [nn.Conv2d],
    [nn.Conv2d, nn.BatchNorm2d],
    [Conv2dDynamicSamePadding],
    [Conv2dDynamicSamePadding, nn.BatchNorm2d],
    [Conv2dStaticSamePadding],
    [Conv2dStaticSamePadding, nn.BatchNorm2d],
]

SUPPORTED_PATTERNS = []
for pat in _SUPPORTED_PATTERNS_BASE:
    SUPPORTED_PATTERNS.append(list(pat))
    for act in _SUPPORTED_ACTS:
        new_pat = list(pat)
        new_pat.append(act)
        SUPPORTED_PATTERNS.append(new_pat)


class QModuleState(Enum):
    """Supported modes in the state machine implemented by QModules"""

    # fmt: off
    BYPASSED = 1                        # Quantizers off, pre-observer off
    QUANT_AWARE_TRAIN_WEIGHT_ONLY = 2   # Weight quantizer on, pre-observer on
    QUANT_AWARE_TRAIN_ACT_ONLY = 3      # Act quantizers on, pre-observer on
    QUANT_AWARE_TRAIN = 4               # All quantizers on, pre-observer on
    CALIBRATION = 5                     # Quantizers off, pre-observer on
    CALIBRATION_WEIGHT_ONLY = 6         # Quantizers off, pre-observer on for weight only
    CALIBRATION_ACT_ONLY = 7            # Quantizers off, pre-observer on for activations only
    QUANT_EVAL_WEIGHT_ONLY = 8          # Weight quantizers on, pre-observer off
    QUANT_EVAL_ACT_ONLY = 9             # Act quantizers on, pre-observer off
    QUANT_EVAL = 10                     # All quantizers on, pre-observer off
    # fmt: on

    @property
    def is_weight_quantized(self):
        return self in (
            QModuleState.QUANT_AWARE_TRAIN_WEIGHT_ONLY,
            QModuleState.QUANT_AWARE_TRAIN,
            QModuleState.QUANT_EVAL_WEIGHT_ONLY,
            QModuleState.QUANT_EVAL,
        )

    @property
    def is_act_quantized(self):
        return self in (
            QModuleState.QUANT_AWARE_TRAIN_ACT_ONLY,
            QModuleState.QUANT_AWARE_TRAIN,
            QModuleState.QUANT_EVAL_ACT_ONLY,
            QModuleState.QUANT_EVAL,
        )

    @property
    def is_weight_observed(self):
        return self in (
            QModuleState.QUANT_AWARE_TRAIN_WEIGHT_ONLY,
            QModuleState.QUANT_AWARE_TRAIN,
            QModuleState.CALIBRATION,
            QModuleState.CALIBRATION_WEIGHT_ONLY,
        )

    @property
    def is_act_observed(self):
        return self in (
            QModuleState.QUANT_AWARE_TRAIN_ACT_ONLY,
            QModuleState.QUANT_AWARE_TRAIN,
            QModuleState.CALIBRATION,
            QModuleState.CALIBRATION_ACT_ONLY,
        )


class QModule(nn.Module):
    """A Quantized Module.

    Adds a mode field, which is used for implementing the state machine"""

    def __init__(self):
        super().__init__()
        self.mode = QModuleState.BYPASSED


def _do_padding(x, layer):
    if isinstance(layer, Conv2dDynamicSamePadding):
        ih, iw = x.size()[-2:]
        kh, kw = layer.weight.size()[-2:]
        sh, sw = layer.stride
        oh, ow = math.ceil(ih / sh), math.ceil(
            iw / sw
        )  # change the output size according to stride ! ! !
        pad_h = max(
            (oh - 1) * layer.stride[0] + (kh - 1) * layer.dilation[0] + 1 - ih, 0
        )
        pad_w = max(
            (ow - 1) * layer.stride[1] + (kw - 1) * layer.dilation[1] + 1 - iw, 0
        )
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

    elif isinstance(layer, Conv2dStaticSamePadding):
        x = layer.static_padding(x)

    else:
        raise ValueError

    return x


class QWrapper(QModule):
    """
    This is the first quantization class that understands the notion of a neural network
    layer. It therefore understands quantization of weights and/or activations, applying
    the non-linearity, observing activation ranges, batch norm folding,
    bypassing/disabling quantization.
    """

    def __init__(
        self,
        layers: Union[nn.Module, Iterable[nn.Module]],
        weight_quantizer: Quantizer,
        acts_quantizer: Quantizer,
        fold_bn: bool = False,
    ):
        super().__init__()
        self.layer = None
        self.bn = None
        self.non_linearity = None

        self.fold_bn = fold_bn
        self.acts = None  # forward() stores last activation tensor

        self.weight_quantizer = weight_quantizer
        self.acts_quantizer = acts_quantizer

        self.parse_sequential_layers(layers)

    def parse_sequential_layers(self, layers):
        # TODO: maybe make more flexible to enable arbitrary activations
        # and different types of normalization layers?
        try:
            types = [type(x) for x in layers]
        except TypeError:  # not iterable
            layers = [layers]
            types = [type(x) for x in layers]

        if not (types in SUPPORTED_PATTERNS):
            raise TypeError(
                f"Provided layers are not supported for fused quantization with QWrapper: {types}."
            )

        self.layer = layers[0]

        try:
            self.bn = next(filter(lambda x: type(x) == nn.BatchNorm2d, layers))
        except StopIteration:
            self.bn = None

        try:
            self.non_linearity = next(
                filter(lambda x: type(x) in _SUPPORTED_ACTS, layers)
            )
        except StopIteration:
            self.non_linearity = None

    def extra_repr(self):
        extra_repr = (
            f"Fused Layer: {self.layer};"
            + f"\n\tWeight Quantizer: ({self.weight_quantizer});"
            + f"\n\tActivation Quantizer: ({self.acts_quantizer})"
        )

        return extra_repr

    def forward(self, x):
        layer = self.layer

        if self.mode.is_weight_observed:
            self.weight_quantizer.pre_observe(layer.weight)

        if self.mode.is_weight_quantized:
            q_weight = self.weight_quantizer(layer.weight)
            self.weight_quantizer.post_observe(q_weight)
        else:
            q_weight = layer.weight

        if isinstance(layer, nn.Conv2d):
            if isinstance(layer, (Conv2dDynamicSamePadding, Conv2dStaticSamePadding)):
                x = _do_padding(x, layer)

            acts = F.conv2d(
                x,
                weight=q_weight,
                bias=layer.bias,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
            )
        elif type(layer) == nn.Linear:
            acts = F.linear(x, q_weight, layer.bias)
        else:
            # We should never get here.
            raise TypeError

        if self.bn is not None:
            # TODO: Implement BatchNorm Folding
            acts = self.bn(acts)

        if self.non_linearity is not None:
            acts = self.non_linearity(acts)

        if self.mode.is_act_observed:
            self.acts_quantizer.pre_observe(acts)

        if self.mode.is_act_quantized:
            acts = self.acts_quantizer(acts)
            self.acts_quantizer.post_observe(acts)

        self.acts = acts

        return acts


class QOp(QModule):
    """Quantized module that wraps operators.

    This can be used to handle operations such as addition, which still need
    to be quantized"""

    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        acts_quantizer: Quantizer,
    ) -> None:
        super().__init__()
        self.op = op
        self.acts_quantizer = acts_quantizer

    def forward(self, *args):
        out = self.op(*args)
        if self.mode.is_act_observed:
            self.acts_quantizer.pre_observe(out)

        if self.mode.is_act_quantized:
            out = self.acts_quantizer(out)
            self.acts_quantizer.post_observe(out)

        return out

    def extra_repr(self) -> str:
        return f"QOp: {self.op}, Quantizer: ({self.acts_quantizer})"
