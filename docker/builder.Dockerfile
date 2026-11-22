# The engine's toolchain, on any machine (no GPU needed): compiles lbc for the pods' GPUs (`remote-pod build`) and
# runs the quality gate (engine/scripts/gate.sh: clang-format, clang-tidy, clang as a second CUDA compiler, cppcheck).
ARG CUDA=12.8.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
      clang-20 clang-tidy-20 clang-format-20 cppcheck \
    && rm -rf /var/lib/apt/lists/*
