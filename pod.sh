#!/usr/bin/env bash
# One command for the RunPod GPU pod: create it, upload code + one dataset, run commands there as in a local shell,
# train in the background, pull runs back, stop it.
#
#   ./pod.sh up [gpu-id]            create the pod (default "NVIDIA RTX A5000": community cloud, else secure), wait for
#                                   ssh; its id goes to .pod (a running pod is reused)
#   ./pod.sh gpus [--all]           table of the GPUs in stock now (cheapest first): the id to pass to `up`, VRAM, $/hr
#                                   community / secure, stock, the best data centers; --all: unavailable ones too
#   ./pod.sh push [dataset]         rsync the code + datasets/<dataset> + tokenizers/<dataset> (default
#                                   agent-mix-v2-bpe1024) to /root/rnn-research (container disk, fast; wiped when the
#                                   pod stops: push again), runs/ -> /workspace/runs (kept), build the venv if missing
#   ./pod.sh sh [command ...]       run a command in /root/rnn-research with the venv active (no command: a shell)
#                                   e.g. ./pod.sh sh model config --run-name X   |   ./pod.sh sh 'nvidia-smi; ls runs'
#   ./pod.sh train <run> [args]     `model train --run-name <run>` in the background (survives disconnects), log in
#                                   runs/<run>/train.log on the pod
#   ./pod.sh log <run>              follow that log (Ctrl-C stops following, not the training)
#   ./pod.sh pull <run> [local]     runs/<run> back into runs/<local> (default the same name): config, metrics,
#                                   latest and best checkpoints; --all: every checkpoint and the log. The run's
#                                   dataset too when it is missing here (metadata + validation split only)
#   ./pod.sh status | stop | start | rm    the pod's state; stop keeps /workspace (billed storage only); rm deletes all
#
# Never uploads env files, credentials, venvs, runs/, other datasets. Uses runpodctl (its api key) for the pod's
# address; ssh with the key runpodctl set up (in the account, so the pod accepts it).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$ROOT/.pod"
REMOTE_DIR="/root/rnn-research"  # container disk: fast local reads (/workspace is a network volume)
RUNS_DIR="/workspace/runs"  # runs/ links here: kept when the pod stops
IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"  # torch 2.9.1, CUDA 12.8, Ubuntu 24.04 (python 3.12)
DATASET_DEFAULT="agent-mix-v2-bpe1024"

die() { echo "$*" >&2; exit 1; }
pod_id() { [[ -s "$STATE" ]] || die "no pod: ./pod.sh up"; cat "$STATE"; }
json() { python3 -c "import json, sys; d = json.load(sys.stdin); print($1)"; }

connect() {  # sets SSH (command) and HOST for the pod
    local info
    info=$(runpodctl ssh info "$(pod_id)" 2>/dev/null) || die "runpodctl ssh info failed (pod stopped? ./pod.sh start)"
    local ip port key
    ip=$(json 'd.get("ip") or ""' <<<"$info")
    port=$(json 'd.get("port") or ""' <<<"$info")
    key=$(json 'd["ssh_key"]["path"]' <<<"$info")
    [[ -n "$ip" && -n "$port" ]] || die "pod has no ssh address yet (status: ./pod.sh status)"
    SSH="ssh -i $key -p $port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o LogLevel=ERROR"
    HOST="root@$ip"
}

remote() {  # remote "<shell command>": runs in the project dir with the venv when it exists
    connect
    local tty=(); [[ -t 0 ]] && tty=(-t)
    $SSH "${tty[@]}" "$HOST" "cd $REMOTE_DIR 2>/dev/null || cd; [ -f .venv-pod/bin/activate ] && . .venv-pod/bin/activate; $1"
}

cmd_up() {
    local gpu="${1:-NVIDIA RTX A5000}"
    if [[ -s "$STATE" ]] && runpodctl pod get "$(cat "$STATE")" >/dev/null 2>&1; then
        echo "pod $(cat "$STATE") exists (./pod.sh status)"; return
    fi
    local out
    for cloud in COMMUNITY SECURE; do
        local extra=()
        [[ $cloud == COMMUNITY ]] && extra=(--public-ip)  # community pods need it for direct ssh (rsync)
        echo "creating $gpu on $cloud cloud..."
        if out=$(runpodctl pod create --name rnn-research --cloud-type "$cloud" --gpu-id "$gpu" --image "$IMAGE" \
                 --ports 22/tcp --container-disk-in-gb 20 --volume-in-gb 20 "${extra[@]}" --wait --wait-timeout 10m 2>&1); then
            json 'd["id"]' <<<"$out" >"$STATE"
            json '" ".join(map(str, [d["id"], d["machine"]["gpuId"], d["machine"]["dataCenterId"], "ssh", d["ssh"]["ip"], d["ssh"]["port"]]))' <<<"$out"
            return
        fi
        echo "  $(tail -c 300 <<<"$out")"
    done
    die "no pod created"
}

cmd_gpus() {
    local all=0; [[ ${1:-} == --all ]] && all=1
    runpodctl gpu list 2>&1 | ALL=$all python3 -c '
import json, os, sys
try:
    gpus = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit("runpodctl gpu list failed (network, or no api key)")
rank = {"High": 0, "Medium": 1, "Low": 2}
price = lambda g: min(p for p in (g.get("communityPricePerHr"), g.get("securePricePerHr")) if p) if (g.get("communityPricePerHr") or g.get("securePricePerHr")) else 1e9
shown = [g for g in gpus if g["available"] or os.environ["ALL"] == "1"]
shown.sort(key=lambda g: (not g["available"], price(g)))
rows = [("gpu id (for up)", "VRAM", "community", "secure", "stock", "data centers in stock (best 3)")]
for g in shown:
    dcs = sorted((d for d in g.get("dataCenterAvailability") or [] if d["stockStatus"] != "none"),
                 key=lambda d: rank.get(d["stockStatus"], 3))
    money = lambda v: f"${v:.2f}" if v else "-"
    rows.append((g["gpuId"], str(g["memoryInGb"]) + " GB", money(g.get("communityPricePerHr")), money(g.get("securePricePerHr")),
                 g["stockStatus"] if g["available"] else "none",
                 " ".join(d["dataCenterId"] + "(" + d["stockStatus"][0] + ")" for d in dcs[:3]) + (" +%d" % (len(dcs) - 3) if len(dcs) > 3 else "")))
w = [max(len(r[i]) for r in rows) for i in range(6)]
for n, r in enumerate(rows):
    print("  ".join(c.rjust(w[i]) if i in (1, 2, 3) else c.ljust(w[i]) for i, c in enumerate(r)).rstrip())
    if n == 0:
        print("  ".join("-" * x for x in w))
print("\n$/hr; stock H(igh) / M(edium) / L(ow); create one with: ./pod.sh up <gpu id>")
'
}

cmd_push() {
    local ds="${1:-$DATASET_DEFAULT}"
    [[ -d "$ROOT/datasets/$ds" ]] || die "no datasets/$ds"
    connect
    $SSH "$HOST" "mkdir -p $REMOTE_DIR $RUNS_DIR && ln -sfn $RUNS_DIR $REMOTE_DIR/runs"
    cd "$ROOT"
    local code=(model tests ad_hoc_scripts webchat pyproject.toml uv.lock AGENTS.md CLAUDE.md README.md pod.sh)
    local builders=(datasets/*.py)
    # -R keeps the relative paths; -rlptD (not -a): no owner/group (the pod's root can't match our uid anyway)
    rsync -rlptDzR --info=stats1 -e "$SSH" --exclude "__pycache__/" --exclude "*.pyc" \
        "${code[@]}" "${builders[@]}" "datasets/$ds" "tokenizers/$ds" "$HOST:$REMOTE_DIR/"
    remote "[ -x .venv-pod/bin/model ] || bash pod.sh _setup" </dev/null
}

cmd__setup() {  # on the pod: a venv over the image's torch (no torch download), the project and its light deps
    cd "$REMOTE_DIR"
    local py=""
    for p in python3 python python3.12; do
        if command -v "$p" >/dev/null && "$p" -c "import torch" 2>/dev/null; then py=$(command -v "$p"); break; fi
    done
    [[ -n "$py" ]] || die "no python with torch on the pod"
    "$py" -m venv --system-site-packages .venv-pod
    .venv-pod/bin/pip install -q --no-deps -e .
    .venv-pod/bin/pip install -q "keras>=3.15" "keras-hub>=0.19" "tokenizers>=0.20" numpy plotly tqdm "rich>=15"
    # gated-deltanet's Triton kernels (no deps: keep the image's torch / triton)
    .venv-pod/bin/pip install -q --no-deps flash-linear-attention fla-core einops
    .venv-pod/bin/python -c "import torch, keras; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'keras', keras.__version__)"
}

cmd_sh() {
    if [[ $# -eq 0 ]]; then remote "exec bash -l"; else remote "$*"; fi  # as ssh: the words joined, run by bash
}

cmd_train() {
    local run="${1:?run name}"; shift
    local extra=""; [[ $# -gt 0 ]] && extra=$(printf '%q ' "$@")
    # setsid -f: its own session (survives the ssh exit); all three streams redirected, so ssh returns at once
    remote "mkdir -p runs/$run && setsid -f nohup model train --run-name $run $extra >>runs/$run/train.log 2>&1 </dev/null; sleep 5; tail -c 400 runs/$run/train.log; echo; echo started" </dev/null
}

cmd_log() { remote "tail -n 40 -f /workspace/runs/${1:?run name}/train.log"; }  # runs/ links there

cmd_pull() {
    local all=0 args=()
    for a in "$@"; do [[ $a == --all ]] && all=1 || args+=("$a"); done
    local run="${args[0]:?run name}" local_run="${args[1]:-${args[0]}}"
    connect
    mkdir -p "$ROOT/runs/$local_run"
    local only=()  # default: what a run needs here (latest + best checkpoint, config, metrics)
    (( all )) || only=(--include=/config.json --include=/metrics.csv --include=/model.keras --include=/checkpoints/
                       --include=/checkpoints/best.keras --exclude="*")
    rsync -rlptDz --info=progress2 -e "$SSH" "${only[@]}" "$HOST:$RUNS_DIR/$run/" "$ROOT/runs/$local_run/"
    # the run's dataset, if it only exists on the pod: what the web chat and evaluation here need (metadata, the
    # validation split and its category index), not the train split
    local ds
    ds=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['data']['dataset'])" "$ROOT/runs/$local_run/config.json")
    if [[ ! -f "$ROOT/datasets/$ds/metadata.json" ]]; then
        echo "dataset $ds: metadata + validation split from the pod"
        mkdir -p "$ROOT/datasets/$ds"
        rsync -rlptDz --info=progress2 -e "$SSH" --include=/metadata.json --include=/validation.npy \
            --include=/validation.categories.npy --exclude="*" "$HOST:$REMOTE_DIR/datasets/$ds/" "$ROOT/datasets/$ds/"
    fi
}

cmd_status() { runpodctl pod get "$(pod_id)" 2>&1 | json '" ".join(map(str, [d["id"], d.get("desiredStatus"), d.get("costPerHr"), "$/hr  uptime", d.get("uptimeSeconds"), "s  ssh", (d.get("ssh") or {}).get("ip"), (d.get("ssh") or {}).get("port")]))'; }
cmd_stop() { runpodctl pod stop "$(pod_id)"; }
cmd_start() { runpodctl pod start "$(pod_id)"; }
cmd_rm() { runpodctl pod delete "$(pod_id)" && rm -f "$STATE"; }

case "${1:-}" in
    up|gpus|push|sh|train|log|pull|status|stop|start|rm|_setup) c="$1"; shift; "cmd_$c" "$@" ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
esac
