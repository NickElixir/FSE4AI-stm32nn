#!/bin/bash
python3 -m venv /tmp/smoke-build &&
/tmp/smoke-build/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
/tmp/smoke-build/bin/pip wheel . --no-deps -w dist &&
ls dist | grep linux &&
python3 -m venv /tmp/smoke-try &&
/tmp/smoke-try/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
/tmp/smoke-try/bin/pip install dist/*.whl &&
cd /tmp &&
/tmp/smoke-try/bin/python -c 'import torch; from stm32nn import round_fp16; x = torch.tensor([0.1, -1.1, 2.2]); y = round_fp16(x); assert torch.equal(y, x.to(torch.float16).to(torch.float32)); print(y.tolist())' &&
echo OK
