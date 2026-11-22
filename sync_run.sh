#!/usr/bin/env bash
# Mirrors a run directory from the current pod to this machine (for cron): the pod's address is looked up through
# runpodctl every time (it changes when the pod restarts). Nothing is deleted locally; rsync writes each file under a
# temporary name and renames it when complete, so the local copy of a checkpoint is never partial. The pod's own
# temporary files (*.tmp, a checkpoint being written) are skipped.
#
#   ./sync_run.sh RUN [REMOTE_RUNS_DIR]     (default /workspace/runs; local copy in runs/RUN, log runs/RUN/sync.log)
#   cron: */10 * * * * /path/to/rnn-research/sync_run.sh RUN
set -euo pipefail
RUN=${1:?run name}
REMOTE_RUNS=${2:-/workspace/runs}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$ROOT/runs/$RUN"
mkdir -p "$LOCAL"
LOG="$LOCAL/sync.log"
exec 9>"$LOCAL/.sync.lock"
flock -n 9 || { echo "$(date '+%F %T') previous sync still running" >>"$LOG"; exit 0; }
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
POD=$(cat "$ROOT/.pod")
info=""
for i in 1 2 3 4 5; do
    info=$(runpodctl ssh info "$POD" 2>/dev/null) || true
    grep -q '"ip"' <<<"$info" && break
    sleep 5
done
if ! grep -q '"ip"' <<<"$info"; then
    echo "$(date '+%F %T') pod $POD unreachable (no address)" >>"$LOG"
    exit 1
fi
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)" <<<"$info"; }
ip=$(field '["ip"]'); port=$(field '["port"]'); key=$(field '["ssh_key"]["path"]')
start=$(date +%s)
if rsync -rlpt --exclude '*.tmp' --exclude '.sync.lock' --exclude 'sync.log' \
    -e "ssh -i $key -p $port -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30" \
    "root@$ip:$REMOTE_RUNS/$RUN/" "$LOCAL/" 2>>"$LOG"; then
    echo "$(date '+%F %T') synced in $(( $(date +%s) - start ))s: $(ls "$LOCAL/checkpoints" 2>/dev/null | tr '\n' ' ')" >>"$LOG"
else
    echo "$(date '+%F %T') rsync failed" >>"$LOG"
    exit 1
fi
