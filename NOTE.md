# Project note

## A bug

Comparing `round_fp16(torch.tensor([1e-7]))` with `x.to(torch.float16).to(torch.float32)` failed. The kernel returned `5.9604645e-8`. PyTorch returned `1.1920929e-7`. The same gap showed up on other subnormals: every code was one bin low. The subnormal right shift was `15 - half_exp`. A binary16 subnormal is an integer number of units of `2^-24`, and the implicit bit of a binary32 value whose unbiased exponent is `-15` belongs nine bits above the least significant bit of that code, which is a shift of 14. Commit `a8a5bbf` sets the shift to `14 - half_exp` and rounds ties to even. After that commit the two calls agree, including on `1e-7`.

## A case the op does not handle

A float64 tensor is outside the contract. `round_fp16` requires CPU float32. On `torch.tensor([0.1, 0.2], dtype=torch.float64)` the call raises `RuntimeError` from the C++ `TORCH_CHECK` (`input must be float32`) and returns no tensor.

## A measurement

Command, run from `/tmp` against the installed wheel:

```bash
python3 -m cProfile -s cumtime /tmp/profile_round_fp16.py
```

The script was:

```python
import torch
from stm32nn import round_fp16
value = torch.randn(4_000_000)
round_fp16(value)
```

`cProfile` printed this line for the kernel:

```text
1    0.039    0.039    0.039    0.039 {built-in method stm32nn._C.round_fp16}
```

Four million elements took 0.039 seconds inside `round_fp16`.
