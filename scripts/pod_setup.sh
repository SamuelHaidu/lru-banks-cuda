#!/usr/bin/env bash
# Prepares a RunPod container for this clone, then builds and prints the GPU's properties. The container disk is
# wiped on every stop (installed tools go with it) while /workspace is kept, so run this after every pod start:
#
#   cd /workspace/lru-banks-cuda && bash scripts/pod_setup.sh
#
# - installs Ninja if missing (the image has CMake, g++ and nvcc at /usr/local/cuda/bin);
# - points this clone's git at the pod's deploy key (write access to this repository only). The key is kept in
#   /workspace/.ssh, but /workspace is a network filesystem that shows every file as rw-rw-rw- and ignores chmod,
#   and ssh refuses a private key others can read: so a 0600 copy goes to ~/.ssh (container disk) on every setup.
set -euo pipefail
cd "$(dirname "$0")/.."

KEY_SRC=/workspace/.ssh/github_lru_banks_cuda
KEY=$HOME/.ssh/github_lru_banks_cuda
HOSTS=/workspace/.ssh/known_hosts_github

if ! command -v ninja >/dev/null; then
    python3 -m pip install --quiet --break-system-packages --root-user-action=ignore ninja
fi

if [[ -f $KEY_SRC && -f $HOSTS ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    install -m 700 -d "$HOME/.ssh"
    install -m 600 "$KEY_SRC" "$KEY"
    git config core.sshCommand "ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$HOSTS"
fi

cmake --preset release >/dev/null
cmake --build --preset release
./build/release/device_info
