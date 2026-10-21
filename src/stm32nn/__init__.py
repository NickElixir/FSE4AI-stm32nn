"""Host checks for an STM32N657 network.

The graded operation is ``round_fp16``. It is the pybind function itself:
a dtype the kernel rejects is raised by ``TORCH_CHECK`` in C++.

``quantize_fp16`` is the weight-only FP16 fake-quant used by the thesis
analysis. Nearest rounding is ``round_fp16``. Stochastic rounding is the
10-bit frexp path. Neither one changes activations.
"""

from pathlib import Path

import torch

from stm32nn._C import (
    compare,
    conv2d,
    export_issues,
    linear,
    max_pool2d,
    quantize_per_channel,
    round_fp16,
    to_nchw,
    to_nhwc,
)

__all__ = [
    "check_export",
    "compare",
    "conv2d",
    "linear",
    "max_pool2d",
    "quantize_fp16",
    "quantize_per_channel",
    "round_fp16",
    "to_nchw",
    "to_nhwc",
]


def quantize_fp16(tensor, rounding="nearest", generator=None):
    """Round a weight tensor to IEEE FP16 and store it back in float32.

    ``rounding="nearest"`` is round-to-nearest-even through ``round_fp16``.
    ``rounding="stochastic"`` rounds the 10-bit mantissa fraction so that the
    expected value stays on the original number. Activations are left alone:
    the caller keeps the matmul in float32.
    """
    if tensor.device.type != "cpu":
        raise ValueError("quantize_fp16 expects a CPU tensor")
    values = tensor if tensor.dtype == torch.float32 else tensor.float()
    if rounding == "nearest":
        return round_fp16(values)
    if rounding != "stochastic":
        raise ValueError(f"Unknown rounding mode: {rounding}")
    return _fp16_frexp_stochastic(values, generator)


def _stochastic_bins(scaled, generator):
    floored = torch.floor(scaled)
    frac = scaled - floored
    if generator is None:
        draw = torch.rand(scaled.shape, device=scaled.device, dtype=scaled.dtype)
    else:
        draw = torch.rand(
            scaled.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=scaled.device, dtype=scaled.dtype)
    return floored + (draw < frac).to(scaled.dtype)


def _fp16_frexp_stochastic(tensor, generator):
    """10-bit mantissa, exponent clamped to the FP16 normal range.

    Subnormals are flushed toward the smallest normal. This matches the
    thesis stochastic path, not the IEEE cast used by ``round_fp16``.
    """
    zero_mask = tensor == 0
    sign = torch.where(tensor < 0, -torch.ones_like(tensor), torch.ones_like(tensor))
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=tensor.device, dtype=torch.float32)
    magnitude = tensor.abs().clamp(min=tiny)
    exponent = torch.floor(torch.log2(magnitude))
    mantissa = magnitude / torch.pow(torch.tensor(2.0, device=tensor.device), exponent)
    levels = float(1 << 10)
    scaled = (mantissa - 1.0) * levels
    scaled_q = _stochastic_bins(scaled, generator)
    overflow = scaled_q >= levels
    scaled_q = torch.where(overflow, torch.zeros_like(scaled_q), scaled_q)
    exponent = torch.where(overflow, exponent + 1.0, exponent)
    mantissa_q = 1.0 + scaled_q / levels
    exponent = exponent.clamp(-14.0, 15.0)
    out = sign * mantissa_q * torch.pow(torch.tensor(2.0, device=tensor.device), exponent)
    return torch.where(zero_mask, torch.zeros_like(out), out)


def check_export(weights, sample, onnx_path=None):
    """Return reasons not to flash this graph.

    ``weights`` is a sequence of parameter tensors. ``sample`` is one model
    input. ``onnx_path``, when set, is scanned for FP16 tensors.
    """
    issues = list(export_issues(list(weights), sample))
    if onnx_path is not None:
        issues.extend(_onnx_fp16_issues(onnx_path))
    return issues


def _onnx_fp16_issues(path):
    file_path = Path(path)
    if not file_path.is_file():
        return [f"ONNX file not found: {file_path}"]
    try:
        import onnx
    except ImportError:
        return ["onnx is not installed; the FP16 scan of the file was skipped"]
    model = onnx.load(str(file_path))
    names = []
    for init in model.graph.initializer:
        if init.data_type in (10, 16):
            names.append(init.name)
    for value in list(model.graph.input) + list(model.graph.output):
        if value.type.tensor_type.elem_type in (10, 16):
            names.append(value.name)
    if names:
        return ["Cube.AI rejects an FP16 ONNX graph: " + ", ".join(names)]
    return []
