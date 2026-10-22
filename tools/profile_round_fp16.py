"""Run: .venv/bin/python -m cProfile -s cumtime tools/profile_round_fp16.py."""

import torch

from stm32nn import round_fp16

value = torch.randn(4_000_000, generator=torch.Generator().manual_seed(0))
round_fp16(value)
