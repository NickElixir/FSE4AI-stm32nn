"""CPU float32 → IEEE binary16 → float32, implemented in C++."""

from stm32nn._C import round_fp16

__all__ = ["round_fp16"]
