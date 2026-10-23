# Project note

## Real bug and fix

Comparing `round_fp16(torch.tensor([1e-7]))` with
`x.to(torch.float16).to(torch.float32)` exposed a subnormal rounding bug:
the extension returned `5.9604645e-8` instead of `1.1920929e-7`.
The binary32 significand was shifted by `15 - half_exp`, one bit too far.
Commit **a8a5bbf** changes it to `14 - half_exp` and applies ties-to-even.
The commit is preserved in this repository: `git show a8a5bbf`.
`test_exact_match_on_known_values` now guards that fix, underflow,
halfway ties, signed zero and overflow. The same conversion code is shared by
the host extension and the Cortex-M55 compile check.

## Independent wheel import

A separate fresh Python process running only `import stm32nn` exposed a loader
error (`Library not loaded: @rpath/libc10.dylib`) that tests beginning with
`import torch` had masked. The package now imports torch before its linked C++
extension. Smoke checks this independent import from the neutral directory
before running the demo and pytest.

## Unsupported case

A CPU `float64` tensor is outside the contract:
`round_fp16(torch.tensor([0.1], dtype=torch.float64))` raises
`RuntimeError: input must be float32` from `TORCH_CHECK` in C++.
Only CPU float32 is accepted. This operation does not implement autograd.

## Measured observation

Measured on the Apple Silicon Mac on 22 October 2026, Python 3.12.5,
PyTorch 2.12.1, against the installed wheel:

```bash
.venv/bin/python -m cProfile -s cumtime tools/profile_round_fp16.py
```

The script creates 4,000,000 CPU float32 elements and calls the extension once.
The kernel line printed:

```text
1    0.023    0.023    0.023    0.023 {built-in method stm32nn._C.round_fp16}
```

That is 23 ms in the extension for this run, excluding tensor creation and
imports. It measures the Mac CPU, not Cortex-M55. Other machines and runs will
differ. The profile script and this command reproduce the measurement.
