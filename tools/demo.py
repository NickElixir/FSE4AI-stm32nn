"""Small deterministic demonstration against the installed wheel."""

import torch

from stm32nn import round_fp16

x = torch.tensor([1e-7, -1e-7, 0.1, 1.0, -2.5, 65504.0])
y = round_fp16(x)
expected = x.to(torch.float16).to(torch.float32)
print(f"PyTorch {torch.__version__}, device={x.device}")
print("input: ", x.tolist())
print("output:", y.tolist())
assert torch.equal(y, expected)
print("Matches PyTorch: True")
generator = torch.Generator().manual_seed(0)
sample = torch.randn(1_000_000, generator=generator)
assert torch.equal(round_fp16(sample), sample.to(torch.float16).to(torch.float32))
print("1,000,000 random values match: True")
try:
    round_fp16(torch.tensor([1, 2, 3]))
except RuntimeError as exc:
    print(f"Integer input rejected by C++: {exc}")
else:
    raise AssertionError("C++ did not reject an integer tensor")
