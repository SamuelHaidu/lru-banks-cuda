#!/usr/bin/env bash
# Trains (or resumes) a run in the background, as the research repository's `pod.sh train` does: the output is
# appended to RUN_DIR/train.log (follow it with `tail -f RUN_DIR/train.log`).
#
#   bash scripts/train.sh RUN_DIR [--epochs N]
set -euo pipefail
RUN=$(realpath -m "${1:?run directory}")
shift
cd "$(dirname "$0")/.."
cmake --build --preset release >/dev/null
[ -f "$RUN/config.json" ] || { echo "no $RUN/config.json: build/release/lbc create $RUN --dataset DIR" >&2; exit 1; }
setsid -f nohup "$PWD/build/release/lbc" train "$RUN" "$@" >>"$RUN/train.log" 2>&1 </dev/null
sleep 5
tail -c 600 "$RUN/train.log"
echo
echo "started: tail -f $RUN/train.log"
