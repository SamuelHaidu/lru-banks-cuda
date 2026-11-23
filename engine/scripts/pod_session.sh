#!/usr/bin/env bash
# One GPU session for the agentic-model work (run on the pod from /work/engine):
#   1. build (release, this GPU)
#   2. data: one corpus chunk from R2 as a dataset (agentic/assemble.py)
#   3. gradient checks: the second FFN, per-layer slot half-lives (tests/check_grads.py, float64 reference)
#   4. a tiny 3-stage lbc train run (context 64 -> 128 -> 256, two datasets with changing weights)
#   5. the planned architecture's step time at every stage shape, and the 512-slot cost proxy
#
#   remote-pod sync engine :engine && remote-pod sh -d --name session 'bash engine/scripts/pod_session.sh'
set -uo pipefail
cd /work/engine
export PATH=/usr/local/cuda/bin:$PATH
DATA=/work/data
step() { printf '\n######## %s\n' "$*"; }

step "1. build"
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null && cmake --build --preset release 2>&1 | tail -1 || exit 1
BIN=build/release

step "2. data"
if [ ! -f $DATA/oc/train.npy ]; then
    uv run -q --no-project --with numpy --with boto3 python /work/agentic/assemble.py --corpus agentic-v1 \
        --source nemotron-opencode --chunks 0 --out $DATA/oc || exit 1
    uv run -q --no-project --with numpy --with boto3 python /work/agentic/assemble.py --corpus agentic-v1 \
        --source nemotron-agentic --chunks 0 --out $DATA/ag || exit 1
fi
TOK=$DATA/oc/train.npy

step "3. gradient checks (PASS / FAIL per seed)"
check() {  # name, train_bench arguments
    local name=$1
    shift
    for seed in 1 2; do
        $BIN/train_bench --data $TOK --tiny --gate-w 0 --gate-r 0 --seed $seed --dump /tmp/dump$seed "$@" >/dev/null 2>&1
        out=$(uv run -q --no-project --with torch --with numpy python tests/check_grads.py /tmp/dump$seed --bf16 2>&1 \
              | grep -E "^(PASS|FAIL)" || true)
        printf "grad check %-22s seed %s: %s\n" "$name" "$seed" "${out:-ERROR (no verdict)}"
    done
}
check default
check hop2-ffn --set arch_params.hop2_ffn=true
check hops-2,1+ffn2 --set 'arch_params.hops=[2,1]' --set arch_params.hop2_ffn=true
check hl-max-per-layer --set 'arch_params.slot_half_life_max=[8192,256]'
check head-dim-64+ffn2 --set arch_params.head_dim=64 --set arch_params.hop2_ffn=true

step "4. tiny 3-stage run"
RUN=/work/runs/stages-test
rm -rf $RUN
$BIN/lbc create $RUN --dataset $DATA/oc arch_params.num_layers=2 'arch_params.hops=[2,1]' arch_params.hop2_ffn=true \
    data.epoch_tokens=2000000 training.validation_steps=2 training.warmup_steps=20 >/dev/null || exit 1
$BIN/lbc config $RUN training.stages '[
  {"name": "ctx64",  "epochs": 2, "set": {"arch_params.window_size": 64,  "data.batch_size": 256, "data.micro_batch_size": 128}},
  {"name": "ctx128", "epochs": 2, "set": {"arch_params.window_size": 128, "data.batch_size": 128, "data.micro_batch_size": 64,
                                          "data.mix_datasets": ["'$DATA'/ag"], "data.mix_fractions": [0.25]}},
  {"name": "ctx256", "epochs": 2, "set": {"arch_params.window_size": 256, "data.batch_size": 64,  "data.micro_batch_size": 32,
                                          "data.mix_datasets": ["'$DATA'/ag"], "data.mix_fractions": [0.5],
                                          "training.learning_rate": 0.001}}]' >/dev/null || exit 1
$BIN/lbc train $RUN --epochs 3 2>&1 | grep -E "^== stage|^Epoch|mixed batches|Resuming|loss:" | sed 's/\r.*\r//' | cut -c1-160
echo "-- resumed after an interruption at epoch 3:"
$BIN/lbc train $RUN 2>&1 | grep -E "^== stage|^Epoch|mixed batches|Resuming|Nothing" | cut -c1-160
cut -d, -f1-3,8 $RUN/metrics.csv

step "5. the planned architecture: step time per stage shape (one micro-batch per step)"
PLAN=(--layers 8 --set 'arch_params.hops=[2,1,1,1,1,1,1,1]' --set arch_params.num_heads=8 --set arch_params.head_dim=64
      --set arch_params.factorized=false --set arch_params.dim=512 --set arch_params.lru_dim=256
      --set arch_params.hop2_ffn=true --set 'arch_params.slot_half_life_max=[8192,1024,1024,1024,1024,1024,1024,1024]')
for shape in "64 512" "128 256" "256 128" "512 64" "1024 32" "2048 16" "4096 8"; do
    set -- $shape
    printf "context %4d x %3d rows: " $1 $2
    $BIN/train_bench --data $TOK --seq $1 --batch $2 --steps 20 --warmup 5 --gate-w 2 --gate-r 2 "${PLAN[@]}" 2>&1 \
        | grep -E "timed steps|GPU memory in use:" | tr '\n' ' '
    echo
done

step "6. the 512-slot cost proxy: one 2-hop layer, 8 heads vs 32 heads (the walk work of 8 heads x 512 slots)"
for heads in 8 32; do
    for shape in "512 64" "4096 8"; do
        set -- $shape
        printf "heads %2d, context %4d x %2d: " $heads $1 $2
        $BIN/train_bench --data $TOK --seq $1 --batch $2 --steps 20 --warmup 5 --gate-w 2 --gate-r 2 --layers 1 \
            --set 'arch_params.hops=[2]' --set arch_params.num_heads=$heads --set arch_params.head_dim=64 \
            --set arch_params.factorized=false --set arch_params.hop2_ffn=true 2>&1 \
            | grep -E "timed steps|^  bank" | tr '\n' ' '
        echo
    done
done
step "done"
