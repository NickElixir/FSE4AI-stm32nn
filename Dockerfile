# Same published course image in both stages; override if the course changes it.
ARG COURSE_IMAGE=ghcr.io/skolai/fse4ai_2025_02_local_unix:01.10.01
FROM ${COURSE_IMAGE} AS build
USER root
ENV DEBIAN_FRONTEND=noninteractive MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
        make g++ gcc-arm-none-eabi libnewlib-dev libstdc++-arm-none-eabi-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN python3 -m pip install --break-system-packages 'torch==2.12.1' \
        --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --break-system-packages 'setuptools>=70,<82' wheel
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE AUTHORS Makefile ./
COPY src ./src
COPY tools/cortex_m55_check.cpp ./tools/cortex_m55_check.cpp
RUN make cortex-check
RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels

FROM ${COURSE_IMAGE}
USER root
RUN python3 -m pip install --break-system-packages 'torch==2.12.1' \
        --index-url https://download.pytorch.org/whl/cpu
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages --no-deps /wheels/*.whl \
    && python3 -m pip install --break-system-packages pytest
COPY tests /opt/tests
COPY tools/demo.py /opt/demo.py
WORKDIR /opt
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
