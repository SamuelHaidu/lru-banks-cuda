# Compiles lbc for the pods' GPUs on any machine (no GPU needed): used by `remote-pod build`.
ARG CUDA=12.8.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
