#!/usr/bin/env bash
# The planned model at the run's micro-batch shapes (~98k tokens each; 128 rows at context 512), the bank_chunk
# sweep at long contexts, and a tiny staged run with per-stage warmup / decay (the learning rate per epoch).
#
#   remote-pod sync engine :engine && remote-pod sh -d --name bench 'bash engine/scripts/pod_bench.sh'
set -uo pipefail
cd /work/engine
export PATH=/usr/local/cuda/bin:$PATH
DATA=/work/data
step() { printf '\n######## %s\n' "$*"; }

step "build"
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null && cmake --build --preset release 2>&1 | tail -1 || exit 1
BIN=build/release

step "data"
for src in nemotron-opencode:oc nemotron-agentic:ag; do
    name=${src%%:*} dir=$DATA/${src##*:}
    [ -f $dir/metadata.json ] && continue
    rm -rf $dir
    uv run -q --no-project --with numpy --with boto3 python /work/agentic/assemble.py --corpus agentic-v1 \
        --source $name --chunks 0 --out $dir || exit 1
done
TOK=$DATA/oc/train.npy

PLAN=(--layers 8 --set 'arch_params.hops=[2,1,1,1,1,1,1,1]' --set arch_params.num_heads=8 --set arch_params.head_dim=64
      --set arch_params.factorized=false --set arch_params.dim=512 --set arch_params.lru_dim=256
      --set arch_params.hop2_ffn=true --set 'arch_params.slot_half_life_max=[8192,1024,1024,1024,1024,1024,1024,1024]')
bench() {  # context rows [extra --set ...]
    local T=$1 B=$2
    shift 2
    printf "context %4d x %4d rows %-28s " $T $B "$*"
    $BIN/train_bench --data $TOK --seq $T --batch $B --steps 15 --warmup 4 --gate-w 2 --gate-r 2 "${PLAN[@]}" "$@" 2>&1 \
        | grep -E "timed steps|GPU memory in use:|out of memory|error" | sed 's/ (after.*graph://' | tr '\n' ' '
    echo
}

step "the run's micro-batches"
bench 64 1536
bench 128 768
bench 256 384
bench 512 128
bench 512 192
bench 1024 96
bench 2048 48
bench 4096 24

step "bank_chunk at long contexts (default 512)"
for T in 1024 2048 4096; do
    B=$((98304 / T))
    bench $T $B --set runtime.bank_chunk=256
    bench $T $B --set runtime.bank_chunk=1024
done

step "tiny staged run: per-stage warmup and decay (lr at each epoch end)"
RUN=/work/runs/sched-test
rm -rf $RUN
$BIN/lbc create $RUN --dataset $DATA/oc arch_params.num_layers=2 'arch_params.hops=[2,1]' arch_params.hop2_ffn=true \
    data.epoch_tokens=1048576 training.validation_steps=1 >/dev/null || exit 1
# 64 updates per epoch (16384 tokens per update), 3 epochs per stage: warmup 20, decay over the last 60 updates
$BIN/lbc config $RUN training.stages '[
  {"name": "ctx64", "epochs": 3, "set": {"arch_params.window_size": 64, "data.batch_size": 256, "data.micro_batch_size": 128,
     "training.learning_rate": 0.001, "training.warmup_steps": 20, "training.lr_decay_start": 132, "training.lr_decay_steps": 60, "training.lr_min_ratio": 0.3}},
  {"name": "ctx128", "epochs": 3, "set": {"arch_params.window_size": 128, "data.batch_size": 128, "data.micro_batch_size": 64,
     "training.learning_rate": 0.002, "training.warmup_steps": 20, "training.lr_decay_start": 132, "training.lr_decay_steps": 60, "training.lr_min_ratio": 0.1}}]' >/dev/null || exit 1
$BIN/lbc train $RUN 2>&1 | grep -E "^== stage" | cut -c1-140
echo "epoch,lr at the epoch's end (expected: 1e-3, 1e-3, 3e-4, then 2e-3, 2e-3, 2e-4)"
cut -d, -f1,8 $RUN/metrics.csv | tail -n +2
step "done"
