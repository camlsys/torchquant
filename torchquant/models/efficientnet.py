from operator import add
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch.model import MBConvBlock, EfficientNet
from efficientnet_pytorch.utils import drop_connect

from torchquant.quantizers import Quantizer
from torchquant.qmodule import QOp, QWrapper


def _adaptive_avg_pool(x):
    return F.adaptive_avg_pool2d(x, 1)


def _excitation(x, x_squeezed):
    return torch.sigmoid(x_squeezed) * x


class FusedMBConvBlock(nn.Module):
    def __init__(
        self,
        block: MBConvBlock,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
    ):
        super().__init__()
        self.has_se = block.has_se
        self.expand_ratio = block._block_args.expand_ratio
        self.id_skip = block.id_skip
        self.stride = block._block_args.stride
        self.input_filters = block._block_args.input_filters
        self.output_filters = block._block_args.output_filters

        if self.expand_ratio != 1:
            mod = block._expand_conv
            bn = block._bn0
            self.expand_conv = QWrapper(
                [mod, bn, block._swish],
                weight_quantizer=weight_quantizer(mod),
                acts_quantizer=acts_quantizer(mod),
            )

        mod = block._depthwise_conv
        bn = block._bn1
        self.depthwise_conv = QWrapper(
            [mod, bn, block._swish],
            weight_quantizer=weight_quantizer(mod),
            acts_quantizer=acts_quantizer(mod),
        )

        if self.has_se:
            self.adaptive_pool = QOp(
                _adaptive_avg_pool,
                acts_quantizer=acts_quantizer(mod),
            )
            mod = block._se_reduce
            self.se_reduce = QWrapper(
                [mod, block._swish],
                weight_quantizer=weight_quantizer(mod),
                acts_quantizer=acts_quantizer(mod),
            )
            mod = block._se_expand
            self.se_expand = QWrapper(
                [mod],
                weight_quantizer=weight_quantizer(mod),
                acts_quantizer=acts_quantizer(mod),
            )
            self.excitation = QOp(
                _excitation,
                acts_quantizer=acts_quantizer(mod),
            )

        mod = block._project_conv
        bn = block._bn2
        self.project_conv = QWrapper(
            [mod, bn],
            weight_quantizer=weight_quantizer(mod),
            acts_quantizer=acts_quantizer(mod),
        )

        self.id_add = QOp(
            add,
            acts_quantizer=acts_quantizer(mod),
        )

    def forward(self, x, drop_connect_rate=None):
        inputs = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.has_se:
            x_squeezed = self.adaptive_pool(x)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = self.excitation(x, x_squeezed)

        x = self.project_conv(x)
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            if drop_connect_rate is not None:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = self.id_add(x, inputs)

        return x


class FusedEfficientNet(nn.Module):
    def __init__(
        self,
        efficientnet: EfficientNet,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
        *,
        quantize_first: bool = True,
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
        self.drop_connect_rate = efficientnet._global_params.drop_connect_rate

        mod = efficientnet._conv_stem
        bn = efficientnet._bn0
        if quantize_first:
            self.conv_stem = QWrapper(
                [mod, bn, efficientnet._swish],
                weight_quantizer=weight_quantizer(mod),
                acts_quantizer=acts_quantizer(mod),
            )
        else:
            self.conv_stem = nn.Sequential(mod, bn, efficientnet._swish)

        self.blocks = nn.ModuleList()
        for block in efficientnet._blocks:
            self.blocks.append(FusedMBConvBlock(block, **kwargs))

        mod = efficientnet._conv_head
        bn = efficientnet._bn1
        self.conv_head = QWrapper(
            [mod, bn, efficientnet._swish],
            weight_quantizer=weight_quantizer(mod),
            acts_quantizer=acts_quantizer(mod),
        )

        dropout = efficientnet._dropout
        fc = efficientnet._fc
        if quantize_fc:
            avg_pool_q = QOp(
                _adaptive_avg_pool,
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
            self.avg_pool = _adaptive_avg_pool
            self.classifier = nn.Sequential(dropout, fc)

    def forward(self, x):
        x = self.conv_stem(x)

        for idx, block in enumerate(self.blocks):
            dc_rate = self.drop_connect_rate
            if dc_rate is not None:
                dc_rate *= float(idx) / len(self.blocks)

            x = block(x, drop_connect_rate=dc_rate)

        x = self.conv_head(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)

        return x
