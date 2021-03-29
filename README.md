[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Follow @notmilad](https://img.shields.io/twitter/follow/notmilad?style=social)](https://twitter.com/notmilad)
[![Follow @satailor96](https://img.shields.io/twitter/follow/satailor96?style=social)](https://twitter.com/satailor96)

# TorchQuant: A Quantization Development Kit for Researchers

This package is aimed at researchers who are researching and developing new quantization algorithms for neural networks using PyTorch. Most of the functionality of this package is also built-into latest versions of PyTorch but they are mostly aimed towards those interested in finding quantised version of their network, and does not offer sufficient flexibility for research purposes in our experience.

## Features
* Support for Affine and Q-format quantizers with STE back-propagation.
* Easy interface for adding new types of quantizers.
* Low-level functional API for state-less quantization.
* Context-managers for quickly changing quantization mode.
* Various range-observers for monitoring activation ranges.
* Easily create quantizable models by wrapping entire layer (or a sequence of supported layers).
* Easily convert common image models to quantized versions (including preserving weights)

## Installation

TorchQuant as a package can be installed via pip:

```
$ pip install git+https://github.com/camlsys/torchquant.git
```

However, if you want to use this package as an starting point to develop your own quantisation schemes you can clone this repository directly in your project and install it in editable mode:

```
$ cd /path/to/your/project
$ git clone https://github.com/camlsys/torchquant.git
$ pip install -e .
```

## Requirements

The only dependencies we use are: PyTorch, torchvision and efficientnet_pytorch. This library is tested as supporting the latest versions available at time of writing.

---

## Package Reference


### Functional Quantizers (`quantizers_functional.py`)

These functions don't have any state and simply quantize a tensor using the passed arguments. They implement Straight-Through-Estimator (STE) for back-propagation and are therefore differentiable.

* Affine Quantization: `affine_quantize(x, delta, zero_point, n_levels)`
* Q-format Quantization: `qfmt_quantize(delta, min_int, max_int)`

### Class-based Quantizers (`quantizers.py`)

These classes wrap the functional API with some state about the ranges, num_bits etc into an `nn.Module` class. All quantizer object must inherit from the `Quantizer` class. For the modules supplied, you must provide a range observer instance. This may be subject to change.

### Range Observers (`range_observers.py`)

`RangeObserver` objects can be used inside `Quantizer` objects to keep track of tensor ranges (but it is not the case that all quantizers _must_ use them). Available types of range observers are:

* `BatchMinMax`
* `ExpAvgMinMax`

### Quantized Modules (`qmodules.py`)
These layers provide a higher-level abstraction to implement apply quantization to a single module or a sequence of commonly-used modules. `QWrapper` can wrap existing a sequence of layers, while `QOp` wraps a single operator that returns a tensor (e.g. addition).

The supported patterns by `QWrapper` are similar to PyTorch:

* Linear
* Linear + ReLU(6) / Swish
* Conv2d
* Conv2d + ReLU(6) / Swish
* Conv2d + BatchNorm2d
* Conv2d + BatchNorm2d + nn.ReLU(6) / Swish

By default these fused layers are quantizable but quantization is turned off by default. You must specify the quantizers as arguments. For example:

```py
wrapper = QWrapper(
    layers,
    weight_quantizer=AffineQuantizer(n_bits, BatchMinMax())
    acts_quantizer=AffineQuantizer(n_bits, ExpAvgMinMax())
)

op = QOp(operators.add, acts_quantizer=AffineQuantizer(n_bits, ExpAvgMinMax()))
```

These modules automatically support our state machine (see below under context managers).

## Model-level

We provide support to convert ResNets and MobileNetV2s from Torchvision and EfficientNets from efficientnet_pytorch to a fused version.
This will preserve the full precision weights.

```py
fused_model = FusedResNet(
    full_precision_model,
    weight_quantizer=lambda module: QuantizerForMyModuleWeights(module),
    acts_quantizer=lambda module: QuantizerForMyModuleActivations(module)
)
```

Note that you are passed the module so you can do any setup required for your quantizer. This API may change depending on user feedback.

## Utilities (`utils.py`)

### Context-managers

For changing the quantization mode in a `QModule` you can do:

```py
with qmodule_state(module, QModuleState.QUANT_AWARE_TRAIN):
    ...

# Alternatively:

set_qmodule_state(module, QModuleState.QUANT_AWARE_TRAIN)
```

This example showed changing to the training mode, but there are other modes of interest. The full set of modes provided are described in the `QModuleState` enum in `qmodule.py`.

*Warning*: You must set the mode explicitly. If you just call `model.train()` or `model.eval()`, the quantization mode will not change. This gives you finer control, but it is easy to forget.

## Roadmap

- [ ] Documentation generation.
- [ ] Per-Channel Quantization.
- [ ] BatchNorm Folding.
- [ ] Binary Neural Networks.
- [ ] Automated graph rewriting with `torch.fx`.
- [ ] Sophisticated research techniques for quantization added as baselines.
- [ ] Debugging tooling.
- [ ] Integration with other toolkits e.g. HuggingFace, SpeechBrain, Flower, etc.
