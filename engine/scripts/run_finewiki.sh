#!/usr/bin/env bash
# Pretraining on English FineWiki with truncated BPTT (log P60), on a remote-pod GPU pod: lbc train in the background,
# the run directory pushed to R2 (runs/<run>/) every 10 minutes and once more at the end. The pod's idle timer stops
# it after the last push.
#
#   remote-pod sh -d --name train 'bash /work/scripts/run_finewiki.sh'
set -uo pipefail
RUN=${RUN:-finewiki-4L-512-tbptt16}
DIR=/work/runs/$RUN
push() { python3 -I /work/scripts/r2.py push "$DIR" "runs/$RUN" 2>&1 | tail -1; }
/work/bin/lbc train "$DIR" >>"$DIR/train.log" 2>&1 &
pid=$!
echo "lbc train pid $pid, log $DIR/train.log"
while kill -0 "$pid" 2>/dev/null; do
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 10; done
    push
done
wait "$pid"
status=$?
push
echo "lbc train exited with $status; run pushed to R2 runs/$RUN"
exit "$status"
