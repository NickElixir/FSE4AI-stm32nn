FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        g++ \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN python3 -m pip install --break-system-packages torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN python3 -m pip wheel . --no-deps -w /wheels

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages /wheels/*.whl pytest
COPY tests /opt/tests
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
