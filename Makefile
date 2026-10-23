PYTHON ?= python3
VENV ?= .venv
PY = "$(abspath $(VENV))/bin/python"
CPU = -mcpu=cortex-m55
ARM_CXX ?= arm-none-eabi-g++
IMAGE ?= stm32nn:0.1

.PHONY: setup wheel install test lint demo smoke cortex-check docker-test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip 'setuptools>=70,<82' wheel
	@if [ "$$(uname -s)" = Linux ]; then \
		$(PY) -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cpu; \
	else $(PY) -m pip install 'torch==2.12.1'; fi
	$(PY) -m pip install pytest ruff

wheel:
	$(PY) -m pip wheel . --no-deps -w dist

install:
	$(PY) -m pip install --force-reinstall --no-deps dist/*.whl

test:
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	cp -R tests "$$tmp/tests" && cd "$$tmp" && $(PY) -m pytest -q tests

lint:
	$(PY) -m ruff check .

demo:
	$(PY) tools/demo.py

smoke:
	PYTHON=$(PYTHON) ./tools/smoke.sh

cortex-check:
	mkdir -p build/cortex-m55
	$(ARM_CXX) $(CPU) -mthumb -mfloat-abi=hard -std=c++17 -O2 \
		-ffreestanding -fno-exceptions -fno-rtti -Isrc/stm32nn \
		-c tools/cortex_m55_check.cpp -o build/cortex-m55/round_fp16.o

docker-test:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE)

clean:
	rm -rf dist build src/*.egg-info .pytest_cache .ruff_cache
	find src tests tools -type d -name __pycache__ -prune -exec rm -rf {} +
