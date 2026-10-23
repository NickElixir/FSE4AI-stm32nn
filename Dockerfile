# Ubuntu 24.04 is the course base confirmed by the team.
ARG COURSE_IMAGE=ubuntu:24.04
FROM ${COURSE_IMAGE} AS build
USER root
ENV DEBIAN_FRONTEND=noninteractive MAX_JOBS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv ca-certificates \
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
RUN python3 -m pip wheel . --no-deps -w /wheels

FROM ${COURSE_IMAGE}
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages 'torch==2.12.1' \
        --index-url https://download.pytorch.org/whl/cpu
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages --no-deps /wheels/*.whl \
    && python3 -m pip install --break-system-packages pytest
COPY tests /opt/tests
COPY tools/demo.py /opt/demo.py
WORKDIR /opt
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
