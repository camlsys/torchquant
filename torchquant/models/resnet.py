from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from torchquant.quantizers import Quantizer
from torchquant.qmodule import QOp, QWrapper


def _add_relu(x, y):
    a = x + y
    return F.relu(a)


class FusedBasicBlock(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
    ) -> None:
        super().__init__()
        self.fused1 = QWrapper(
            [block.conv1, block.bn1, block.relu],
            weight_quantizer=weight_quantizer(block.conv1),
            acts_quantizer=acts_quantizer(block.conv1),
        )

        self.fused2 = QWrapper(
            [block.conv2, block.bn2],
            weight_quantizer=weight_quantizer(block.conv2),
            acts_quantizer=acts_quantizer(block.conv2),
        )

        if block.downsample is not None:
            self.downsample = QWrapper(
                block.downsample,
                weight_quantizer=weight_quantizer(block.downsample),
                acts_quantizer=acts_quantizer(block.downsample),
            )
        else:
            self.downsample = None

        # we pass the preceding conv layer to this, since the activation shape will match
        self.qadd_relu = QOp(_add_relu, acts_quantizer(block.conv2))
        self.stride = block.stride

    def forward(self, x):
        identity = x

        out = self.fused1(x)
        out = self.fused2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.qadd_relu(out, identity)

        return out


class FusedBottleNeck(nn.Module):
    def __init__(
        self,
        bottleneck: Bottleneck,
        weight_quantizer: Callable[[nn.Module], Quantizer],
        acts_quantizer: Callable[[nn.Module], Quantizer],
    ) -> None:
        super().__init__()
        self.fused1 = QWrapper(
            [bottleneck.conv1, bottleneck.bn1, bottleneck.relu],
            weight_quantizer=weight_quantizer(bottleneck.conv1),
            acts_quantizer=acts_quantizer(bottleneck.conv1),
        )

        self.fused2 = QWrapper(
            [bottleneck.conv2, bottleneck.bn2, bottleneck.relu],
            weight_quantizer=weight_quantizer(bottleneck.conv2),
            acts_quantizer=acts_quantizer(bottleneck.conv2),
        )

        self.fused3 = QWrapper(
            [bottleneck.conv3, bottleneck.bn3],
            weight_quantizer=weight_quantizer(bottleneck.conv3),
            acts_quantizer=acts_quantizer(bottleneck.conv3),
        )

        if bottleneck.downsample is not None:
            self.downsample = QWrapper(
                bottleneck.downsample,
                weight_quantizer=weight_quantizer(bottleneck.downsample),
                acts_quantizer=acts_quantizer(bottleneck.downsample),
            )
        else:
            self.downsample = None

        # we pass the preceding conv layer to this, since the activation shape will match
        self.qadd_relu = QOp(
            _add_relu,
            acts_quantizer(bottleneck.conv3),
        )

    def forward(self, x):
        identity = x
        out = self.fused1(x)
        out = self.fused2(out)
        out = self.fused3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.qadd_relu(identity, out)


def _convert_blocks(
    blocks: nn.ModuleList,
    weight_quantizer: Callable[[nn.Module], Quantizer],
    acts_quantizer: Callable[[nn.Module], Quantizer],
) -> nn.Sequential:
    fused = []
    for b in blocks:
        if isinstance(b, BasicBlock):
            fused.append(
                FusedBasicBlock(
                    b,
                    weight_quantizer,
                    acts_quantizer,
                )
            )
        elif isinstance(b, Bottleneck):
            fused.append(
                FusedBottleNeck(
                    b,
                    weight_quantizer,
                    acts_quantizer,
                )
            )
        else:
            raise ValueError(type(b))

    return nn.Sequential(*fused)


class FusedResNet(nn.Module):
    def __init__(
        self,
        net: ResNet,
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

        if quantize_first:
            self.fused_1 = QWrapper(
                [net.conv1, net.bn1, net.relu],
                weight_quantizer=weight_quantizer(net.conv1),
                acts_quantizer=acts_quantizer(net.conv1),
            )
        else:
            self.fused_1 = nn.Sequential(net.conv1, net.bn1, net.relu)

        self.maxpool = net.maxpool

        convert = lambda layer: _convert_blocks(
            layer,
            weight_quantizer,
            acts_quantizer,
        )
        self.layer1 = convert(net.layer1)
        self.layer2 = convert(net.layer2)
        self.layer3 = convert(net.layer3)
        self.layer4 = convert(net.layer4)

        if quantize_fc:
            self.avgpool = QOp(
                net.avgpool,
                acts_quantizer=avg_pool_quantizer(net.fc),
            )
            self.fc = QWrapper(
                [net.fc],
                weight_quantizer=fc_weight_quantizer(net.fc),
                acts_quantizer=fc_acts_quantizer(net.fc),
            )

        else:
            self.avgpool = net.avgpool
            self.fc = net.fc

    def forward(self, x):
        x = self.fused_1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
