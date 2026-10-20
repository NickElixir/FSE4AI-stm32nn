"""PC-side helpers for an STM32N657 deployment.

The kernels do not replace Cube.AI. They round, pack, quantize, and score a
network on the host before a flash.
"""

from pathlib import Path

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
    "quantize_per_channel",
    "round_fp16",
    "to_nchw",
    "to_nhwc",
]


def check_export(weights, sample, onnx_path=None):
    """Return a list of reasons not to flash this graph.

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
