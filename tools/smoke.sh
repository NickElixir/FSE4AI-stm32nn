#!/bin/bash
# Linux is the grading path; macOS uses PyPI and its native platform tag.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
case "$(uname -s)" in
    Linux) TORCH_SOURCE=(--index-url https://download.pytorch.org/whl/cpu); TAG=linux ;;
    Darwin) TORCH_SOURCE=(--index-url https://pypi.org/simple); TAG=macosx ;;
    *) echo 'Supported systems: Linux and macOS' >&2; exit 1 ;;
esac
cd "$ROOT" &&
"$PYTHON" -m venv "$WORK/build" &&
"$WORK/build/bin/python" -m pip install 'setuptools>=70,<82' wheel &&
"$WORK/build/bin/python" -m pip install 'torch==2.12.1' "${TORCH_SOURCE[@]}" &&
"$WORK/build/bin/python" -m pip wheel . --no-deps -w dist &&
ls dist | grep "$TAG" &&
"$PYTHON" -m venv "$WORK/try" &&
"$WORK/try/bin/python" -m pip install 'torch==2.12.1' "${TORCH_SOURCE[@]}" &&
"$WORK/try/bin/python" -m pip install dist/*.whl pytest &&
cp -R tests "$WORK/tests" &&
cp tools/demo.py "$WORK/demo.py" &&
cd "$WORK" &&
"$WORK/try/bin/python" -c 'import stm32nn; print("Installed wheel import:", stm32nn.__file__)' &&
"$WORK/try/bin/python" demo.py &&
"$WORK/try/bin/python" -m pytest -q tests &&
echo OK
