# stm32nn

Function: `stm32nn.round_fp16`

Formula, in words. The input is a CPU tensor of dtype float32. Finite values are the range used below. Each element is rounded to the nearest IEEE binary16 value (1 sign bit, 5 exponent bits, 10 significand bits). Ties round to even. The rounded value is stored back in float32. The result has the same shape as the input and dtype float32. A non-contiguous input is read in logical order; the result is a contiguous tensor of the same shape.

The specification line is:

```python
x.to(torch.float16).to(torch.float32)
```

`round_fp16` rejects anything that is not a CPU float32 tensor. An integer tensor raises `RuntimeError`. The failure is the C++ `TORCH_CHECK` in the extension.

## Build, install, test

From this directory, on CPU Linux. Torch comes from the CPU index and is not part of the wheel.

```bash
python3 -m venv /tmp/build-venv
/tmp/build-venv/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
/tmp/build-venv/bin/pip wheel . --no-deps -w dist
python3 -m venv /tmp/try-venv
/tmp/try-venv/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
/tmp/try-venv/bin/pip install dist/*.whl
/tmp/try-venv/bin/pip install pytest
cp -r tests /tmp/proj-tests
cd /tmp && /tmp/try-venv/bin/python -m pytest -q /tmp/proj-tests
```

The same wheel steps are `make wheel`, `make install`, and `make test`. `make clean` removes `dist/`, `build/`, and `__pycache__/`. `./tools/smoke.sh` builds the wheel, checks that its name contains `linux`, installs it into a fresh virtual environment, calls `round_fp16` from `/tmp`, and prints `OK`.

The image is Ubuntu 24.04 in two stages. The builder installs CPU torch and runs `pip wheel`. The final stage installs CPU torch, then the wheel, and runs pytest on the copied tests.

```bash
docker build -t fse-ops:0.1 .
docker run --rm fse-ops:0.1
```

The project note is `NOTE.md`.
