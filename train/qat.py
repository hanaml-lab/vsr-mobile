"""Quantisation-aware fine-tuning with learned step sizes (LSQ, Esser et al., ICLR 2020), as in DRIFT.

    prepare_qat(model, bits=8)   # every Conv2d but the flow head's becomes a QuantConv2d
    ...fine-tune...
    model = freeze(model)        # plain Conv2d again, weights snapped to their learned grid

Weights are quantised per output channel and conv inputs per tensor, symmetrically, as TensorRT's
INT8 does. The flow head is left in floating point because export/tensorrt.py keeps it (and the
warp) in FP16 on the device. ``freeze`` keeps the snapped weights and drops the activation steps:
activation ranges on the device come from TensorRT's calibration on held-out footage.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x - x * scale).detach() + x * scale


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def fake_quantize(x: torch.Tensor, step: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Round ``x`` to the signed ``bits``-bit grid of spacing ``step`` (broadcast), clipping at its ends.
    The gradient passes straight through inside the range and is zero where clipped; the step gets
    LSQ's gradient, scaled by 1 / sqrt(elements per step x Qp)."""
    qn, qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    s = _scale_grad(step, 1.0 / math.sqrt(max(x.numel() // max(step.numel(), 1), 1) * qp))
    return _round_ste((x / s).clamp(qn, qp)) * s


class QuantConv2d(nn.Conv2d):
    """A Conv2d with LSQ fake quantisation of its weights (per output channel) and its input (per tensor)."""

    def __init__(self, conv: nn.Conv2d, bits: int = 8):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        self.load_state_dict(conv.state_dict())
        self.bits = bits
        qp = 2 ** (bits - 1) - 1
        w = conv.weight.detach()
        self.weight_step = nn.Parameter(2 * w.abs().mean(dim=(1, 2, 3)).clamp_min(1e-8).view(-1, 1, 1, 1) / qp**0.5)
        self.input_step = nn.Parameter(torch.zeros(()))  # set from the first batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_step.item() == 0:
            with torch.no_grad():
                self.input_step.fill_(2 * x.abs().mean().clamp_min(1e-8).item() / (2 ** (self.bits - 1) - 1) ** 0.5)
        x = fake_quantize(x, self.input_step, self.bits)
        w = fake_quantize(self.weight, self.weight_step, self.bits)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _replace(module: nn.Module, make, prefix: str = "", skip: tuple[str, ...] = ()) -> None:
    for name, child in module.named_children():
        path = f"{prefix}{name}"
        if any(path == s or path.startswith(f"{s}.") for s in skip):
            continue
        new = make(child)
        if new is not None:
            setattr(module, name, new)
        else:
            _replace(child, make, f"{path}.", skip)


def prepare_qat(model: nn.Module, bits: int = 8, skip: tuple[str, ...] = ("flow",)) -> nn.Module:
    """Swap every Conv2d outside ``skip`` (module paths) for a QuantConv2d, in place."""
    _replace(model, lambda m: QuantConv2d(m, bits) if type(m) is nn.Conv2d else None, skip=skip)
    return model


def _frozen(m: QuantConv2d) -> nn.Conv2d:
    conv = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups,
                     m.bias is not None, m.padding_mode)
    with torch.no_grad():
        conv.weight.copy_(fake_quantize(m.weight, m.weight_step, m.bits))
        if m.bias is not None:
            conv.bias.copy_(m.bias)
    return conv


def freeze(model: nn.Module) -> nn.Module:
    """Swap every QuantConv2d back for a plain Conv2d with its weights on the learned grid, in place."""
    _replace(model, lambda m: _frozen(m) if isinstance(m, QuantConv2d) else None)
    return model
