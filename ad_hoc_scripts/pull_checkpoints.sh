#!/usr/bin/env bash
# Pull a pod run's latest + best checkpoints (cron: `*/5 * * * *`), then check the copies: a pull that met the pod
# mid-save gets a truncated zip, so it pulls once more after 30 s. One line per pull in logs/pull_<run>.log.
#   ad_hoc_scripts/pull_checkpoints.sh <run>
set -uo pipefail
run="${1:?run name}"
cd "$(dirname "$0")/.."
log="logs/pull_${run}.log"
ok() {
    /usr/bin/python3 - "$run" <<'PY'
import sys, zipfile, pathlib
run = pathlib.Path("runs") / sys.argv[1]
for p in (run / "model.keras", run / "checkpoints" / "best.keras"):
    if p.exists():
        try:
            if zipfile.ZipFile(p).testzip() is not None:
                sys.exit(1)
        except zipfile.BadZipFile:
            sys.exit(1)
PY
}
for attempt in 1 2; do
    if ./pod.sh pull "$run" >/dev/null 2>/tmp/pull_err_$$ && ok; then
        echo "$(date '+%F %T') ok: epochs $(($(wc -l < "runs/$run/metrics.csv") - 1)), last $(tail -1 "runs/$run/metrics.csv" | cut -d, -f1,4)" >> "$log"
        rm -f /tmp/pull_err_$$
        exit 0
    fi
    sleep 30
done
echo "$(date '+%F %T') FAILED: $(tail -c 300 /tmp/pull_err_$$ | tr '\n' ' ')" >> "$log"
rm -f /tmp/pull_err_$$
