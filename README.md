# stm32nn — финальный проект FSE4AI

Одна CPU-операция PyTorch на C++: **`stm32nn.round_fp16(x)`**.
Вход — CPU-тензор `float32`. Каждый элемент округляется до IEEE binary16
(FP16, ties to even), затем представляется обратно в `float32`.
Форма сохраняется. Спецификация — одна строка PyTorch:

```python
x.to(torch.float16).to(torch.float32)
```

Поддерживаются batch, скаляр, пустой и неконтинуальный тензор; результат
континуальный. Целочисленный, `float64`, `float16` или не-CPU вход отклоняется
в C++ через `TORCH_CHECK` и вызывает `RuntimeError`. Числовые проверки используют
конечные значения; переполнение FP16 даёт ±inf по спецификации. Это проверка
округления весов: вход, выход и последующие вычисления остаются FP32.

## Быстрый запуск: MacBook или CPU Linux

Нужны Python **3.11 или 3.12**, `make` и C++-компилятор. На macOS установите
Xcode Command Line Tools (`xcode-select --install`), на Debian/Ubuntu:
`sudo apt-get install python3-venv python3-dev build-essential`.
GPU, STM32, данные MNIST и исследовательский репозиторий не нужны.
Команды выполняются из каталога этого проекта:

```bash
make setup
make wheel
make install
make test
make demo
make lint
```

`setup` создаёт `.venv` и устанавливает PyTorch 2.12.1, pytest и Ruff.
На Mac PyTorch берётся из PyPI, на Linux — из официального CPU index.
Для конкретной версии Python: `make setup PYTHON=python3.12`.
`wheel` использует уже установленный PyTorch, без второго изолированного
скачивания. `install` устанавливает wheel из `dist/`.
`test` копирует тесты во временный каталог и проверяет **установленный wheel**.
`demo` проверяет известные значения, миллион случайных чисел и ошибку dtype.
Ожидается **7 passed**, два сообщения `True` и ошибка `input must be float32`.

Полная независимая проверка с двумя свежими окружениями:

```bash
./tools/smoke.sh
```

Скрипт собирает wheel, проверяет его платформенный тег (`linux` или `macosx`),
устанавливает в новое окружение, запускает demo и pytest из нейтрального
каталога. Последняя строка — `OK`. Он работает из любого текущего каталога;
исполняемый бит включён в Git. Нужен интернет для установки зависимостей.

Wheel зависит от Python и ОС: Mac wheel нельзя передать Linux-преподавателю.
Передавайте исходники; на машине преподавателя те же команды создадут её
собственный wheel. На Intel/AMD Linux тег будет `linux_x86_64`.
PyTorch не включён внутрь wheel и устанавливается отдельно.

## Docker: воспроизведение на компьютере преподавателя

Нужен запущенный Docker. Оба этапа используют опубликованный образ курса
`ghcr.io/skolai/fse4ai_2025_02_local_unix:01.10.01`
(из `fse4ai_2025/compose.yaml`). Если преподаватель использует другой образ,
его можно указать через `--build-arg COURSE_IMAGE=...`.

```bash
docker build -t stm32nn:0.1 .
docker run --rm stm32nn:0.1
docker run --rm stm32nn:0.1 python3 /opt/demo.py
```

Или `make docker-test`. Builder устанавливает CPU torch, проверяет компиляцию
для Cortex-M55 и собирает wheel. Финальный этап устанавливает CPU torch,
затем wheel, копирует тесты и запускает pytest. Исходники расширения,
компиляторы builder и локальные `dist/`, `build/`, `.venv`, `.git` в финальный
этап не копируются. На Apple Silicon Docker собирает Linux ARM64;
на машине Intel/AMD — Linux x86_64. Для явной проверки x86_64 на Mac:

```bash
docker build --platform linux/amd64 -t stm32nn:amd64 .
docker run --rm --platform linux/amd64 stm32nn:amd64
```

Эмуляция x86_64 на Mac медленнее нативной сборки. CI выполняет Linux x86_64
без эмуляции; отдельно проверяются macOS wheel и Docker.

## Cortex-M55: что проверяет параметр CPU

В Makefile задано:

```make
CPU = -mcpu=cortex-m55
```

Он применяется **только к проверке компиляции переносимого алгоритма**:

```bash
make cortex-check
```

Нужен Arm GNU Toolchain (`arm-none-eabi-g++`). На Debian/Ubuntu:
`sudo apt-get install gcc-arm-none-eabi libnewlib-dev libstdc++-arm-none-eabi-dev`.
Другой путь: `make cortex-check ARM_CXX=/path/to/arm-none-eabi-g++`.
Результат — `build/cortex-m55/round_fp16.o`, ARM-объект из того же `fp16.h`,
который вызывает PyTorch-расширение. Docker выполняет эту проверку автоматически.
Она не требует платы и не исполняет ARM-код: это не замер скорости M55
и не готовая прошивка. Флаг `-mcpu=cortex-m55` нельзя передавать компилятору
Mac/x86 Linux для сборки загружаемого модуля PyTorch.

## Состав и проверка задания

- `src/stm32nn/fp16.h` — единственный алгоритм округления и преобразования обратно.
- `src/stm32nn/ops.cpp` — CPU-тензоры, `TORCH_CHECK`, `PYBIND11_MODULE`.
- `setup.py`, `pyproject.toml`, `MANIFEST.in` — CppExtension и упаковка.
- `tests/test_op.py` — 7 тестов, пять обязательных видов из ТЗ.
- `tools/smoke.sh`, `Makefile`, `Dockerfile` — сборка и проверка установки.
- `.github/workflows/ci.yml` — Ruff, свежий wheel, pytest и Docker на push/PR.
- `NOTE.md` — реальный баг с коммитом, ограничение и cProfile.
- `CHECKS.md` — проверенные результаты и соответствие пунктам ТЗ/лекциям.

`make clean` удаляет результаты сборки и кэши, оставляя `.venv`.
Работа с прошивками и исследовательские отчёты остаются в исходном проекте;
этот учебный вариант не импортирует их и не пытается воспроизводить нейросеть.

## Git и сдача

История исходного проекта сохранена: merge `14535de`, исправление `a8a5bbf`,
коммиты всех трёх участников с адресами из `AUTHORS`, не менее 10 коммитов
за три даты. Проверка: `git log --graph --oneline --all`, `git log --merges`,
`git shortlog -sne HEAD`. Новые изменения должны быть закоммичены и влиты
в default branch перед сдачей; успешный локальный прогон не означает запуск CI.

Перед отправкой сделайте последний push. ТЗ требует один ZIP с каталогом
проекта **и `.git` внутри**; `git archive` для этого не подходит.
Из родительского каталога, заменив `LOGIN` на логин сдающего:

```bash
zip -r LOGIN.zip fse4ai-stm32nn-course \
  -x '*/.venv/*' '*/dist/*' '*/build/*' '*/__pycache__/*' \
     '*/.pytest_cache/*' '*/.ruff_cache/*' '*/*.egg-info/*'
```

Перед сдачей распакуйте ZIP в новый каталог и повторите `./tools/smoke.sh`
или Docker-команды. Дедлайн и фактическую отправку проверяет команда в LMS.
