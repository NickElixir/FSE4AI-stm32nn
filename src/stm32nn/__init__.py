"""CPU float32 → IEEE binary16 → float32, implemented in C++."""

# Load libtorch/libc10 before loading the linked C++ extension.
import torch as _torch

from stm32nn._C import round_fp16

__all__ = ["round_fp16"]
del _torch
