#!/usr/bin/env bash
# End-to-end check of remote-pod against a local agent container (no RunPod, no GPU):
#
#   docker run -d --rm --name rp-agent -p 127.0.0.1:18080:8080 -e REMOTE_POD_TOKEN=localtest remote-pod:agent-test
#   bash tools/remote-pod/scripts/e2e_local.sh [path/to/remote-pod]
#
# Each step prints PASS or FAIL; the script exits non-zero on the first failure.
set -uo pipefail
RP=${1:-tools/remote-pod/dist/remote-pod-local}
export REMOTE_POD_URL=http://127.0.0.1:18080 REMOTE_POD_TOKEN=localtest NO_COLOR=1
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; exit 1; }

# sh: output, exit code, stdin, arguments after the command
out=$($RP sh echo hello) && [[ $out == hello ]] && pass "sh prints the output" || fail "sh output: $out"
$RP sh "exit 7"; [[ $? == 7 ]] && pass "sh returns the remote exit code" || fail "exit code"
out=$(echo abc | $RP sh tr a-z A-Z) && [[ $out == ABC ]] && pass "sh forwards piped stdin" || fail "stdin: $out"
out=$($RP sh ls -a /work) && [[ $out == *"."* ]] && pass "sh passes flags to the command" || fail "flags: $out"
err=$($RP sh "echo oops >&2" 2>&1 >/dev/null) && [[ $err == oops ]] && pass "sh keeps stderr separate" || fail "stderr: $err"

# cp: a directory up and back, byte for byte
mkdir -p "$tmp/src/sub" && head -c 5000000 /dev/urandom > "$tmp/src/sub/blob.bin" && echo hi > "$tmp/src/a.txt"
$RP cp "$tmp/src" :data/ >/dev/null 2>&1 || fail "cp up"
$RP cp :data/src "$tmp/back" >/dev/null 2>&1 || fail "cp down"
cmp -s "$tmp/src/sub/blob.bin" "$tmp/back/sub/blob.bin" && pass "cp round trip (5 MB, directory)" || fail "cp content"

# sync: the second run sends only the edited file
mkdir -p "$tmp/engine" && for i in 1 2 3; do echo "kernel $i" > "$tmp/engine/k$i.cu"; done
$RP sync "$tmp/engine" :engine >/dev/null 2>&1 || fail "sync 1"
echo "kernel 2, faster" > "$tmp/engine/k2.cu"
n=$($RP sync "$tmp/engine" :engine --json 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["sent"]))')
[[ $n == 1 ]] && pass "sync sends only the changed file" || fail "sync sent $n"
out=$($RP sh cat engine/k2.cu) && [[ $out == "kernel 2, faster" ]] && pass "sync content" || fail "sync content: $out"

# a detached job, followed to its end, then read again from an offset
$RP sh -d --name e2e "for i in 1 2 3; do echo step \$i; sleep 0.3; done; exit 4" >/dev/null 2>&1 || fail "detach"
out=$($RP logs e2e -f 2>/dev/null); code=$?
[[ $out == $'step 1\nstep 2\nstep 3' && $code == 4 ]] && pass "logs -f follows a job to its exit code" || fail "logs: $out ($code)"
out=$($RP logs e2e --offset 7 2>/dev/null) && [[ $out == $'step 2\nstep 3' ]] && pass "logs resumes at a byte offset" || fail "offset: $out"
$RP jobs --json 2>/dev/null | python3 -c 'import json,sys; j=json.load(sys.stdin); assert j[0]["name"]=="e2e" and j[0]["exit_code"]==4' \
  && pass "jobs lists it" || fail "jobs"

# Ctrl-C reaches a remote command (INT to its process group: exit 130)
( sleep 1; kill -INT $(pgrep -f "remote-pod-local sh sleep 30" | head -1) ) &
$RP sh sleep 30 >/dev/null 2>&1; code=$?
[[ $code == 130 ]] && pass "Ctrl-C stops the remote command (exit 130)" || fail "interrupt: exit $code"

# a wrong token is refused with a hint
out=$(REMOTE_POD_TOKEN=wrong $RP sh true 2>&1); code=$?
[[ $code == 1 && $out == *"refused the token"* ]] && pass "a wrong token is refused" || fail "token: $out"
echo "all passed"
