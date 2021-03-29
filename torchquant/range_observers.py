import abc
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["RangeObserver", "BatchMinMax", "ExpAvgMinMax"]


class RangeObserver(nn.Module, abc.ABC):
    """Base-class for all range observers"""

    def __init__(self):
        super().__init__()

        self.register_buffer("range_min", None)
        self.register_buffer("range_max", None)

    @abc.abstractmethod
    def observe_batch(self, x: torch.Tensor):
        raise NotImplementedError

    def get_ranges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (self.range_min is None) and (self.range_max is None):
            raise AttributeError
        else:
            return self.range_min, self.range_max

    def force_ranges(self, range_min: torch.Tensor, range_max: torch.Tensor):
        self.range_min = range_min
        self.range_max = range_max


class BatchMinMax(RangeObserver):
    def observe_batch(self, x):
        self.range_min = x.detach().min()
        self.range_max = x.detach().max()


class ExpAvgMinMax(RangeObserver):
    def __init__(self, momentum: float =0.9):
        super().__init__()

        # It's important that this is defined as float because for some reason when a
        # python float (i.e. momentum) is multiplied by a torch.long the result is long!
        self.register_buffer("num_batches_observed", torch.tensor(0, dtype=torch.float))
        self.momentum = momentum

    def observe_batch(self, x):

        self.num_batches_observed = self.num_batches_observed + 1

        range_min = x.detach().min()
        range_max = x.detach().max()

        self.range_min = self.__exp_avg(range_min, self.range_min, self.momentum)
        self.range_max = self.__exp_avg(range_max, self.range_max, self.momentum)

    def __exp_avg(self, x, x_avg, momentum):

        if x_avg is None:  # Very first observation
            x_avg = 0.0

        x_avg_new = (x_avg * momentum) + (x * (1 - momentum))
        return x_avg_new

    def get_ranges(self):
        """Has to overwrite the parent implementation to apply bias-correction"""
        if (self.range_min is None) and (self.range_max is None):
            raise AttributeError
        else:
            bias_corr_factor = 1.0 - (self.momentum ** self.num_batches_observed)
            return self.range_min / bias_corr_factor, self.range_max / bias_corr_factor
