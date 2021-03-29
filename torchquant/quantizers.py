import abc

import torch
import torch.nn as nn
from torchquant.quantizers_functional import (
    affine_quantize,
    qfmt_quantize,
    gradient_clip,
)
from torchquant.range_observers import RangeObserver

__all__ = ["AffineQuantizer", "QfmtQuantizer", "Quantizer"]


class Quantizer(nn.Module, abc.ABC):
    """
    A base class for quantizers describing the core operations required.

    The following invocation pattern is used inside supplied `QModules`
    * `pre_observe` is invoked on tensors when training and calibrating.
    * `quantize` is invoked when training or evaluating
    * `post_observe` is invoked when training or evaluating
    """

    @abc.abstractmethod
    def pre_observe(self, x: torch.Tensor):
        """
        Observe the tensor before it is quantized.

        Example usage: monitoring ranges for range-observer based methods
        """
        pass

    @abc.abstractmethod
    def quantize(self, x: torch.Tensor):
        """Quantize the tensor"""
        pass

    @abc.abstractmethod
    def post_observe(self, x: torch.Tensor):
        """Observe the tensor after it is quantized"""
        pass

    def forward(self, x: torch.Tensor):
        return self.quantize(x)


# TODO: there is a bit of duplication between the Affine and Qfmt quantizers.
# Maybe a minor refactor would be useful?


class AffineQuantizer(Quantizer):
    def __init__(
        self,
        n_bits: int,
        range_observer: RangeObserver,
        symmetric: bool = False,
        gradient_clip: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.observer = range_observer
        self.symmetric = symmetric
        self.clip = gradient_clip

    def pre_observe(self, x):
        self.observer.observe_batch(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        range_min, range_max = self.observer.get_ranges()

        # For one-sided distributions the range is relaxed to include zero:
        x_min = min(range_min, 0)
        x_max = max(range_max, 0)

        if self.symmetric:
            x_max = max(x_min.abs(), x_max)
            x_min = -x_max

        delta = (x_max - x_min).float() / ((2 ** self.n_bits) - 1)
        zero_point = torch.round(-x_min / delta)

        x_q = affine_quantize(x, delta, zero_point, 2 ** self.n_bits)
        if self.clip:
            x_q = gradient_clip(x_q, range_min, range_max)

        return x_q

    def post_observe(self, x: torch.Tensor):
        # TODO: collect stats?
        pass


class QfmtQuantizer(Quantizer):
    def __init__(
        self,
        n_bits: int,
        signed: bool,
        range_observer: RangeObserver,
        gradient_clip: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.signed = signed
        self.observer = range_observer
        self.clip = gradient_clip

    def pre_observe(self, x: torch.Tensor):
        self.observer.observe_batch(x)

    def quantize(self, x: torch.Tensor):
        # TODO: what if range_max is also negative?
        # TODO: What's the correct behaviour when range is actually a power of 2?
        #      Shouldn't I do .ceil() instead of .floor() + 1?
        range_min, range_max = self.observer.get_ranges()

        range_abs = max(abs(range_min), range_max)
        int_bits = range_abs.log2().floor() + 1

        frac_bits = self.n_bits - int_bits
        if self.signed:
            frac_bits -= 1

        delta = 2 ** (-frac_bits)

        if self.signed:
            range_int_min = -(2 ** int_bits)
        else:
            range_int_min = 0

        range_int_max = (2 ** int_bits) - 1

        x_q = qfmt_quantize(x, delta, range_int_min, range_int_max)
        if self.clip:
            x_q = gradient_clip(x_q, range_min, range_max)

        return x_q

    def post_observe(self, x: torch.Tensor):
        # TODO: stats?
        pass
