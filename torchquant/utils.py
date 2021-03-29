import torch.nn as nn

from torchquant.qmodule import QModule, QModuleState

__all__ = ["set_qmodule_state", "qmodule_state"]


def set_qmodule_state(module: nn.Module, mode: QModuleState):
    def set_mode_fn(m):
        if isinstance(m, QModule):
            m.mode = mode

    module.apply(set_mode_fn)


class qmodule_state(object):
    """
    Context-manager for temporarily changing quantization modes of quantized modules 
    """

    def __init__(self, module: nn.Module, mode: QModuleState):
        self.module = module
        self.mode = mode
        self.prev_modes = []

    def __enter__(self):
        for m in self.module.modules():
            if isinstance(m, QModule):
                self.prev_modes.append((m, m.mode))
                m.mode = self.mode

    def __exit__(self, type, value, traceback):
        for m, prev_mode in self.prev_modes:
            m.mode = prev_mode
