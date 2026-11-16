#!/usr/bin/env bash
# Pushes code + one dataset + all tokenizers to a remote pod over rsync/ssh (never runs/),
# then sets up a venv on the pod that uses the pod's system torch (nothing downloads torch).
#
# Usage:
#   ./push.sh -ip 1.2.3.4 -p 2222 -d tinystories-10k [-k ~/.ssh/id_ed25519]
#
#   -k <key>   ssh identity file (default ~/.ssh/id_ed25519)
#
# The venv: /workspace/rnn-research/.venv-pod, --system-site-packages, the project (--no-deps) plus
# keras/tokenizers/numpy; the pod's own torch is used.
#
# Always uploads to /workspace/rnn-research on the remote (local disk; copy runs back with pull_run.sh). Never uploads env files (.env, *.env: tokens)
# or virtual environments.
# On the pod, run `.venv-pod/bin/model ...`, not `uv run` (uv.lock pins torch+xpu).

set -euo pipefail

IP=""
PORT=""
DATASET=""
KEY="$HOME/.ssh/id_ed25519"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ip) IP="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        -d) DATASET="$2"; shift 2 ;;
        -k) KEY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$IP" || -z "$PORT" || -z "$DATASET" ]]; then
    echo "Usage: $0 -ip <address> -p <port> -d <dataset-name> [-k <key>]" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
REMOTE_DIR="/workspace/rnn-research"
SSH="ssh -p $PORT -i $KEY"

if [[ ! -d "$ROOT/datasets/$DATASET" ]]; then
    echo "No such dataset: datasets/$DATASET" >&2
    exit 1
fi
if [[ ! -d "$ROOT/tokenizers/$DATASET" ]]; then
    echo "No such tokenizer: tokenizers/$DATASET" >&2
    exit 1
fi

$SSH "$REMOTE" "mkdir -p $REMOTE_DIR/datasets $REMOTE_DIR/tokenizers"

# Big dataset files (> PART_MB) go as PART_MB pieces over JOBS parallel rsync/ssh streams (one stream
# rarely fills the uplink to a far pod), then are joined on the pod and checked against the local sha256.
# Files already on the pod with the same sha256 are skipped. Everything else goes in parallel too.
JOBS=6          # parallel streams for the pieces (+3 for code/tokenizers/small files; sshd MaxStartups is 10)
PART_MB=32
DS="datasets/$DATASET"
mapfile -t BIG < <(cd "$ROOT" && find "$DS" -type f -size +${PART_MB}M | sort)
declare -A SHA
for rel in "${BIG[@]}"; do SHA[$rel]=$(sha256sum "$ROOT/$rel" | cut -d' ' -f1); done
SEND=()
for rel in "${BIG[@]}"; do
    remote_sha=$($SSH "$REMOTE" "sha256sum $REMOTE_DIR/$rel 2>/dev/null | cut -d' ' -f1" || true)
    if [[ "$remote_sha" == "${SHA[$rel]}" ]]; then echo "same on pod, skip: $rel"; else SEND+=("$rel"); fi
done

# -rlptD instead of -a: skip owner/group preservation (-o -g), which fails
# with "Operation not permitted" when the remote user can't chown to match
# the local UID/GID (e.g. syncing into a container as a different user).
EXCLUDE_BIG=()
for rel in "${BIG[@]}"; do EXCLUDE_BIG+=(--exclude "/$DATASET/$(basename "$rel")"); done
PIDS=()

# Code, excluding local-only/generated directories.
rsync -rlptDz \
    -e "$SSH" \
    --exclude ".venv/" \
    --exclude ".venv-*/" \
    --exclude "venv/" \
    --exclude "env/" \
    --exclude "runs/" \
    --exclude "datasets/" \
    --exclude "tokenizers/" \
    --exclude ".sycl_cache/" \
    --exclude "__pycache__/" \
    --exclude "*.egg-info/" \
    --exclude ".git/" \
    --exclude ".env" \
    --exclude ".env.*" \
    --exclude "*.env" \
    --exclude "credentials*" \
    --exclude "*.pem" \
    --exclude "*.key" \
    --exclude "node_modules/" \
    --exclude "studies/*.pdf" \
    "$ROOT/" "$REMOTE:$REMOTE_DIR/" && echo "done: code" &
PIDS+=($!)

# Requested dataset only, small files (the big ones go as pieces below).
rsync -rlptDz -e "$SSH" "${EXCLUDE_BIG[@]}" \
    "$ROOT/$DS" "$REMOTE:$REMOTE_DIR/datasets/" && echo "done: $DS small files" &
PIDS+=($!)

# All tokenizers.
rsync -rlptDz -e "$SSH" "$ROOT/tokenizers/" "$REMOTE:$REMOTE_DIR/tokenizers/" && echo "done: tokenizers" &
PIDS+=($!)

if (( ${#SEND[@]} )); then
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    for rel in "${SEND[@]}"; do
        mkdir -p "$TMP/.parts/$rel"
        split -b "${PART_MB}M" -d -a 4 "$ROOT/$rel" "$TMP/.parts/$rel/p."
    done
    TOTAL=$(find "$TMP/.parts" -type f | wc -l)
    START=$SECONDS
    export SSH REMOTE REMOTE_DIR TMP TOTAL
    # -R keeps the path after /./, so each piece lands in $REMOTE_DIR/.parts/<file>/ (dirs created by rsync).
    (cd "$TMP" && find .parts -type f | sort) | xargs -P "$JOBS" -I{} sh -c '
        ok=0
        for try in 1 2 3 4 5; do
            rsync -Rlptz -e "$SSH" "$TMP/./{}" "$REMOTE:$REMOTE_DIR/" && { ok=1; break; }
            echo "retry $try {}" >&2; sleep 3
        done
        [ "$ok" = 1 ] || exit 255
        echo sent' | {
        n=0
        while read -r _; do
            n=$((n + 1)); el=$((SECONDS - START))
            echo "  $n/$TOTAL pieces, $((n * PART_MB)) MB, $(( n * PART_MB / (el > 0 ? el : 1) )) MB/s"
        done
        (( n == TOTAL )) || { echo "only $n/$TOTAL pieces sent" >&2; exit 1; }
    }
    echo "pieces sent in $((SECONDS - START)) s, joining on the pod"
    JOIN=""
    for rel in "${SEND[@]}"; do
        JOIN+="cat .parts/$rel/p.* > $rel.tmp && mv $rel.tmp $rel && echo '${SHA[$rel]}  $rel' | sha256sum -c && "
    done
    $SSH "$REMOTE" "cd $REMOTE_DIR && $JOIN rm -rf .parts"  # stops at the first bad join
fi

for pid in "${PIDS[@]}"; do wait "$pid"; done

# A venv on the pod that sees the container's own torch (no torch download).
$SSH "$REMOTE" bash -s <<REMOTE_SETUP
set -euo pipefail
cd $REMOTE_DIR
# The interpreter that has the image's torch (python3 can be an older one, e.g. 3.10 with torch on 3.12).
PY=""
for p in python python3 python3.13 python3.12; do
    if command -v \$p >/dev/null && \$p -c "import torch" 2>/dev/null; then PY=\$(command -v \$p); break; fi
done
[[ -n "\$PY" ]] || { echo "no python with torch on the pod" >&2; exit 1; }
echo "using \$PY (\$(\$PY -V))"
.venv-pod/bin/python -c "import torch" 2>/dev/null || { rm -rf .venv-pod; \$PY -m venv --system-site-packages .venv-pod; }
.venv-pod/bin/pip install -q --no-deps -e .
.venv-pod/bin/pip install -q "keras>=3.15" "keras-hub>=0.19" "tokenizers>=0.20" numpy plotly tqdm
KERAS_BACKEND=torch .venv-pod/bin/python -c "import torch, keras; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'keras', keras.__version__)"
REMOTE_SETUP
