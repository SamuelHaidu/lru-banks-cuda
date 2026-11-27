#!/usr/bin/env bash
# The agentic model's training run, resumable from anywhere: on a fresh or restarted pod the same command
# reinstalls, rebuilds, reassembles the data, pulls the run from R2 and continues (lbc train resumes from
# model.lbc: weights, Adam moments, step, stage, schedule, data position). Checkpoints, metrics and the log go to
# R2 (runs/<run>/) every 10 minutes and at the end.
#
#   remote-pod sync engine :engine && remote-pod cp datasets/agentic :/work/ &&
#   remote-pod sh -d --name train 'bash engine/scripts/run_agentic.sh'     # start, or resume
#   remote-pod logs train -f --epochs                                      # follow
#
# Model: 8 layers (layer 1: 2 hops and a second FFN, slot half-lives to 8192), 8 heads x 64, 128 slots per head,
# dim 512, LRU 256 complex, tied 32k embedding (agentic-bpe32768). Stages: context 64 -> 4096, ~15B tokens.
set -uo pipefail
RUN_NAME=${RUN_NAME:-agentic-47m}
RUN=/work/runs/$RUN_NAME
DATA=/work/data
cd /work/engine
export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH
UV="uv run -q --no-project --with numpy --with boto3 python"

# 1. tools and build (idempotent)
if ! command -v cmake >/dev/null || ! command -v uv >/dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build zstd curl python3 >/dev/null
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
fi
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null && cmake --build --preset release | tail -1 || exit 1
LBC=build/release/lbc
# a host whose GPU is unusable (e.g. no /dev/nvidia-uvm) fails here, before the data is fetched
build/release/device_info >/dev/null || { echo "the GPU is not usable on this host: replace the pod"; exit 3; }

# 2. data: whole chunks of corpora/agentic-v1 per source (~16.8B tokens)
assemble() {  # source, dir, chunks...
    local src=$1 dir=$DATA/$2
    shift 2
    [ -f $dir/metadata.json ] && return 0
    rm -rf $dir
    $UV /work/agentic/assemble.py --corpus agentic-v1 --source $src --chunks "$@" --out $dir || exit 1
}
assemble open-swe-traces oswe 0 1 2 3 4
assemble openmath-tir math 0 1 2 3
assemble toucan toucan 0 1 2
assemble nemotron-swe nswe 0 1
assemble terminal-corpus term 0
assemble nemotron-agentic agentic 0
assemble nemotron-opencode opencode 0

# 3. the run: pulled from R2 when it exists there (a new pod), else created
mkdir -p $RUN
$UV scripts/r2.py pull runs/$RUN_NAME $RUN
if [ ! -f $RUN/config.json ]; then
    $LBC create $RUN --dataset $DATA/oswe arch_params.num_layers=8 'arch_params.hops=[2,1,1,1,1,1,1,1]' \
        arch_params.num_heads=8 arch_params.head_dim=64 arch_params.factorized=false arch_params.dim=512 \
        arch_params.lru_dim=256 arch_params.hop2_ffn=true \
        'arch_params.slot_half_life_max=[8192,1024,1024,1024,1024,1024,1024,1024]' \
        data.epoch_tokens=125000000 training.validation_steps=2 training.checkpoint_every=10 \
        "data.mix_datasets=[\"$DATA/math\", \"$DATA/toucan\", \"$DATA/nswe\", \"$DATA/term\", \"$DATA/agentic\", \"$DATA/opencode\"]" \
        'data.mix_fractions=[0.245, 0.183, 0.088, 0.06, 0.06, 0.06]' >/dev/null || exit 1
    # per stage: rows per update, micro-batch, lr (sqrt of tokens per update from 1e-3 at 256 x 512, capped 2e-3),
    # warmup, then the decay over the stage's last updates (updates per stage = epochs x round(125M / tokens))
    $LBC config $RUN training.stages '[
 {"name": "ctx64", "epochs": 5, "set": {"arch_params.window_size": 64, "data.batch_size": 1536, "data.micro_batch_size": 1536,
  "training.learning_rate": 0.00087, "training.warmup_steps": 1000, "training.lr_decay_start": 5760, "training.lr_decay_steps": 600, "training.lr_min_ratio": 0.3}},
 {"name": "ctx128", "epochs": 5, "set": {"arch_params.window_size": 128, "data.batch_size": 768, "data.micro_batch_size": 768,
  "training.learning_rate": 0.00087, "training.warmup_steps": 300, "training.lr_decay_start": 5760, "training.lr_decay_steps": 600, "training.lr_min_ratio": 0.3}},
 {"name": "ctx256", "epochs": 10, "set": {"arch_params.window_size": 256, "data.batch_size": 384, "data.micro_batch_size": 384,
  "training.learning_rate": 0.00087, "training.warmup_steps": 300, "training.lr_decay_start": 11520, "training.lr_decay_steps": 1200, "training.lr_min_ratio": 0.3}},
 {"name": "ctx512", "epochs": 15, "set": {"arch_params.window_size": 512, "data.batch_size": 256, "data.micro_batch_size": 128,
  "training.learning_rate": 0.001, "training.warmup_steps": 300, "training.lr_decay_start": 12910, "training.lr_decay_steps": 1400, "training.lr_min_ratio": 0.3}},
 {"name": "ctx1024", "epochs": 25, "set": {"arch_params.window_size": 1024, "data.batch_size": 288, "data.micro_batch_size": 96, "runtime.bank_chunk": 1024,
  "training.learning_rate": 0.0015, "training.warmup_steps": 300, "training.lr_decay_start": 9600, "training.lr_decay_steps": 1000, "training.lr_min_ratio": 0.3}},
 {"name": "ctx2048", "epochs": 30, "set": {"arch_params.window_size": 2048, "data.batch_size": 240, "data.micro_batch_size": 48,
  "training.learning_rate": 0.0019, "training.warmup_steps": 300, "training.lr_decay_start": 6870, "training.lr_decay_steps": 750, "training.lr_min_ratio": 0.3}},
 {"name": "ctx4096", "epochs": 30, "set": {"arch_params.window_size": 4096, "data.batch_size": 264, "data.micro_batch_size": 24,
  "training.learning_rate": 0.002, "training.warmup_steps": 300, "training.lr_decay_start": 2380, "training.lr_decay_steps": 1100, "training.lr_min_ratio": 0.1}}]' >/dev/null || exit 1
fi

# 4. train, with the run copied to R2 every 10 minutes and at the end
( while sleep 600; do $UV scripts/r2.py push $RUN runs/$RUN_NAME >/dev/null 2>&1; done ) &
SYNC=$!
$LBC train $RUN 2>&1 | tee -a $RUN/train.log
status=${PIPESTATUS[0]}
kill $SYNC 2>/dev/null
$UV scripts/r2.py push $RUN runs/$RUN_NAME
echo "lbc train exited $status"
exit $status
