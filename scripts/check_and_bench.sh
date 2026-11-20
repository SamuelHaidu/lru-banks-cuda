#!/usr/bin/env bash
# The protocol behind every row of the research log: the gradient check (2 seeds, the reference's forward rounded to
# bf16 like the C++), then the throughput at 64 x 512 with every gate open (B1) and with every read closed (B2).
#
#   bash scripts/check_and_bench.sh /root/data/pretrain-mix-bpe32768/train.npy [extra train_bench args]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?token file}
shift || true
BIN=build/release/train_bench
cmake --build --preset release >/dev/null
for seed in 1 2; do
    "$BIN" --data "$DATA" --tiny --gate-w 0 --gate-r 0 --seed "$seed" --dump "/tmp/lbc_dump$seed" "$@" >/dev/null
    printf "grad check seed %s: " "$seed"
    python3 tests/check_grads.py "/tmp/lbc_dump$seed" --bf16 2>/dev/null | grep -E "^(PASS|FAIL)" || echo "ERROR"
done
for mode in "B1 all gates open:--gate-w 20 --gate-r 20" "B2 reads closed:--gate-w 20 --gate-r -20"; do
    name=${mode%%:*}
    args=${mode#*:}
    echo "== $name"
    # shellcheck disable=SC2086
    "$BIN" --data "$DATA" --steps 20 --warmup 3 $args "$@" |
        grep -E "timed steps|GPU memory in use:|^  [a-z_+]+ +[0-9.]+ ms"
done
