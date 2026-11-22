#!/usr/bin/env bash
# The engine's quality gate in one command: format check, lint (clang -Werror build, clang-tidy host + device
# profiles, cppcheck), the nvcc release build with -Werror, then the tests. All of it is CMake + Ninja
# (`cmake --workflow --preset gate`); this script only supplies the toolchain when this machine lacks it.
#
#   scripts/gate.sh            the gate
#   scripts/gate.sh format     rewrite the sources with clang-format first, then the gate
#
# Toolchain: nvcc (CUDA 12.8+), clang-20, clang-tidy-20, clang-format-20, cppcheck, cmake, ninja. Without them here,
# everything runs in the builder container (docker/builder.Dockerfile, built on first use; no GPU needed).
set -euo pipefail
cd "$(dirname "$0")/.."

steps='cmake --workflow --preset gate'
if [[ "${1:-}" == format ]]; then
    steps="cmake --preset lint >/dev/null && cmake --build --preset lint --target format && $steps"
fi

have() { command -v "$1" >/dev/null 2>&1; }
if have clang-tidy-20 && have clang-format-20 && have clang++-20 && have cppcheck && have ninja \
    && { have nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; }; then
    exec bash -c "$steps"
fi

cuda=12.8.1
image=remote-pod-builder:$cuda
docker build --network host -q -t "$image" --build-arg CUDA=$cuda -f ../docker/builder.Dockerfile ../docker >/dev/null
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:$PWD" -w "$PWD" "$image" bash -c "$steps"
