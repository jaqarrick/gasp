# Stage 1: fetch the prebuilt faial bundle (pico/faial-cost, cu-to-json, libz3.so)
FROM ubuntu:24.04 AS pico

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --yes wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/faial && cd /opt/faial && \
    wget -nv "https://gitlab.com/umb-svl/faial/-/jobs/artifacts/main/raw/bundle/faial.tar.bz2?job=bundle-lin" \
      -O faial.tar.bz2 && \
    tar xjf faial.tar.bz2 && \
    rm faial.tar.bz2

# Stage 2: gasp
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# libgmp10 is a runtime dependency of faial-cost
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgmp10 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=pico /opt/faial /opt/faial
# The binaries have no rpath, so point the loader at the bundled libz3.so
ENV PATH="/opt/faial/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/faial/lib:$LD_LIBRARY_PATH"

WORKDIR /workspace
COPY requirements.txt requirements-dev.txt ./

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
COPY . /workspace

ENV PATH="/workspace/bin:$PATH"

RUN python3 --version \
    && nvcc --version \
    && nvdisasm --version \
    && cuobjdump --version \
    && ncu --version \
    && faial-cost --help > /dev/null \
    && which cu-to-json \
    && echo "All tools OK"
ENV PYTHONPATH="/workspace/src"
ENTRYPOINT ["python", "-m", "gasp"]
