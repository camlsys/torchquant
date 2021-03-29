import torch
from torch.autograd import Function

__all__ = ["affine_quantize", "qfmt_quantize"]


class AffineQantizerFunction(Function):
    @staticmethod
    def forward(ctx, x, delta, zero_point, n_levels):
        x_int = torch.clamp(torch.round(x / delta) + zero_point, 0, n_levels - 1)
        x_float = (x_int - zero_point) * delta
        return x_float

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output, None, None, None


class QfmtQuantizerFunction(Function):
    @staticmethod
    def forward(ctx, x, delta, min_int, max_int):

        x_int = torch.clamp(torch.round(x / delta), min_int, max_int)
        x_float = x_int * delta
        return x_float

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output, None, None, None


class GradientClipper(Function):
    @staticmethod
    def forward(ctx, x, range_min, range_max):
        ctx.save_for_backward(x, range_min, range_max)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, range_min, range_max = ctx.saved_tensors
        grad_x = grad_x.clone()
        grad_x[x < range_min] = 0.0
        grad_x[x > range_max] = 0.0
        return grad_x, None, None


affine_quantize = AffineQantizerFunction.apply
qfmt_quantize = QfmtQuantizerFunction.apply
gradient_clip = GradientClipper.apply
