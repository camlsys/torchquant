from operator import add
from typing import Callable, Optional

import torch.nn as nn
import torch.nn.functional as F

# In recent versions of Torchvision, the mobilenet file has been split to accomodate V3
try:
    from torchvision.models.mobilenet import MobileNetV2, InvertedResidual, ConvBNReLU
except ImportError:
    from torchvision.models.mobilenetv2 import MobileNetV2, InvertedResidual, ConvBNReLU

from torchquant.quantizers import Quantizer
from torchquant.qmodule import QOp, QWrapper


def _conv_bn_relu_tolist(cbr: ConvBNReLU):
    modules = []
    for mod in cbr:
        modules.append(mod)
    return modules


def _convert_cbr(
    cbr: ConvBNReLU,
    weight_quantizer,
    acts_quantizer,
) -> QWrapper:
    cbr = _conv_bn_relu_tolist(cbr)
    mod = cbr[0]
    return QWrapper(
        cbr,
        weight_quantizer=weight_quantizer(mod),
        acts_quantizer=acts_quantizer(mod),
    )


class FusedInvertedResidual(nn.Module):
    def __init__(
        self,
        inv_res: InvertedResidual,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
    ) -> None:
        super().__init__()

        layers = []
        i = 0
        if len(inv_res.conv) == 4:
            # expand_ratio != 1 -> first layer is a CBR
            # pw
            layers.append(
                _convert_cbr(
                    inv_res.conv[0],
                    weight_quantizer=weight_quantizer,
                    acts_quantizer=acts_quantizer,
                )
            )
            i += 1

        else:
            assert len(inv_res.conv) == 3

        # dw
        cbr = _conv_bn_relu_tolist(inv_res.conv[i])
        mod = cbr[i]

        layers.append(
            _convert_cbr(
                inv_res.conv[i],
                weight_quantizer=weight_quantizer,
                acts_quantizer=acts_quantizer,
            )
        )

        conv_bn = [inv_res.conv[i + 1], inv_res.conv[i + 2]]
        mod = conv_bn[0]
        layers.append(
            QWrapper(
                conv_bn,
                weight_quantizer=weight_quantizer(mod),
                acts_quantizer=acts_quantizer(mod),
            )
        )

        self.conv = nn.Sequential(*layers)
        self.use_res_connect = inv_res.use_res_connect
        if self.use_res_connect:
            self.qadd = QOp(
                add,
                acts_quantizer=acts_quantizer(mod),
            )

    def forward(self, x):
        if self.use_res_connect:
            return self.qadd(x, self.conv(x))
        else:
            return self.conv(x)


def avg_pool_reshape(x):
    return F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)


class FusedMobileNetV2(nn.Module):
    def __init__(
        self,
        mobilenet: MobileNetV2,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
        *,
        quantize_first: bool = False,
        quantize_fc: bool = False,
        avg_pool_quantizer: Optional[Callable[[nn.Linear], Quantizer]] = None,
        fc_weight_quantizer: Optional[Callable[[nn.Linear], Quantizer]] = None,
        fc_acts_quantizer: Optional[Callable[[nn.Linear], Quantizer]] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(
            weight_quantizer=weight_quantizer,
            acts_quantizer=acts_quantizer,
        )
        fp_features = list(mobilenet.features)
        features = []

        if quantize_first:
            features.append(_convert_cbr(fp_features[0], **kwargs))
        else:
            features.append(fp_features[0])

        for inv_res in fp_features[1:-1]:
            features.append(FusedInvertedResidual(inv_res, **kwargs))

        features.append(_convert_cbr(fp_features[-1], **kwargs))
        self.features = nn.Sequential(*features)

        dropout = mobilenet.classifier[0]
        fc = mobilenet.classifier[-1]
        if quantize_fc:
            avg_pool_q = QOp(
                avg_pool_reshape,
                acts_quantizer=avg_pool_quantizer(fc),
            )
            fc_q = QWrapper(
                [fc],
                weight_quantizer=fc_weight_quantizer(fc),
                acts_quantizer=fc_acts_quantizer(fc),
            )

            self.avg_pool = avg_pool_q
            self.classifier = nn.Sequential(dropout, fc_q)
        else:
            self.avg_pool = avg_pool_reshape
            self.classifier = nn.Sequential(dropout, fc)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x
