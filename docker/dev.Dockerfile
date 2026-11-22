# The pod image for the edit / build / profile loop: the train image's agent plus the CUDA toolkit (nvcc), CMake,
# Ninja, git and the Nsight profilers (nsys, ncu). Use: remote-pod up --gpu "RTX 4090" --image dev
ARG CUDA=12.8.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04
ARG CUDA
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates zstd git cmake ninja-build \
      python3 cuda-nsight-systems-$(echo ${CUDA} | cut -d. -f1-2 | tr . -) cuda-nsight-compute-$(echo ${CUDA} | cut -d. -f1-2 | tr . -) \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/
COPY tools/remote-pod/dist/remote-pod /usr/local/bin/remote-pod
ENV REMOTE_POD_WORKDIR=/work
WORKDIR /work
EXPOSE 8080
ENTRYPOINT ["remote-pod", "agent"]
