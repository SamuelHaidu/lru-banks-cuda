#!/usr/bin/env bash
# The protocol behind every row of the research log: the gradient check (2 seeds, the reference's forward rounded to
# bf16 like the C++) of the default model and of each arch option, then the throughput at 64 x 512 with every gate
# open (B1) and with every read closed (B2).
#
#   bash scripts/check_and_bench.sh /root/data/pretrain-mix-bpe32768/train.npy [extra train_bench args]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?token file}
shift || true
BIN=build/release/train_bench
cmake --build --preset release >/dev/null
check() {  # name, then train_bench arguments
    local name=$1
    shift
    for seed in 1 2; do
        "$BIN" --data "$DATA" --tiny --gate-w 0 --gate-r 0 --seed "$seed" --dump "/tmp/lbc_dump$seed" "$@" >/dev/null
        out=$(python3 tests/check_grads.py "/tmp/lbc_dump$seed" --bf16 2>/dev/null | grep -E "^(PASS|FAIL)" || true)
        printf "grad check %-13s seed %s: %s\n" "$name" "$seed" "${out:-ERROR (no verdict)}"
    done
}
check default "$@"
check untied --set arch_params.tied_embeddings=false "$@"
check unfactorized --set arch_params.factorized=false "$@"
check hops-2,1 --set 'arch_params.hops=[2,1]' "$@"
check slots-64,16 --set 'arch_params.bank_slots=[64,16]' "$@"
check head-dim-64 --set arch_params.head_dim=64 "$@"
check all-options --set arch_params.tied_embeddings=false --set arch_params.factorized=false \
    --set 'arch_params.hops=[1,2]' "$@"
# truncated BPTT: the dumped step starts from carried states (one bank chunk per row: final states in place; four)
check carry --set data.stream_chunks=4 --dump-after 2 "$@"
check carry-chunks --set data.stream_chunks=4 --set runtime.bank_chunk=16 --dump-after 2 "$@"
# the carried state is the true continuation: 2 x 512 with the carry against one row of 1024 (bank chunks of 512, 256)
for bc in 512 256; do
    out=$("$BIN" --data "$DATA" --set data.stream_chunks=2 --bank-chunk "$bc" --carry-check "$@" | grep -E "^(PASS|FAIL)" || true)
    printf "carry check   bank chunk %-4s: %s\n" "$bc" "${out:-ERROR (no verdict)}"
done
for mode in "B1 all gates open:--gate-w 20 --gate-r 20" "B2 reads closed:--gate-w 20 --gate-r -20"; do
    name=${mode%%:*}
    args=${mode#*:}
    echo "== $name"
    # shellcheck disable=SC2086
    "$BIN" --data "$DATA" --steps 20 --warmup 3 $args "$@" |
        grep -E "timed steps|GPU memory in use:|^  [a-z0-9_+]+ +[0-9.]+ ms"
done
