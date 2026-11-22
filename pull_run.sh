#!/usr/bin/env bash
# Pulls a run directory from a remote pod's /workspace/rnn-research/runs/
# into this repo's own runs/, over rsync/ssh.
#
# Usage:
#   ./pull_run.sh -ip 1.2.3.4 -p 2222 -r retriver-tokenid -l gpu-retriver-tokenid [-k ~/.ssh/id_ed25519]
#
# Remote: /workspace/rnn-research/runs/<remote-run-name>/
# Local:  ./runs/<local-run-name>/

set -euo pipefail

IP=""
PORT=""
REMOTE_RUN=""
LOCAL_RUN=""
KEY="$HOME/.ssh/id_ed25519"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ip) IP="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        -r) REMOTE_RUN="$2"; shift 2 ;;
        -l) LOCAL_RUN="$2"; shift 2 ;;
        -k) KEY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$IP" || -z "$PORT" || -z "$REMOTE_RUN" || -z "$LOCAL_RUN" ]]; then
    echo "Usage: $0 -ip <address> -p <port> -r <remote-run-name> -l <local-run-name>" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="root@${IP}"
REMOTE_DIR="/workspace/rnn-research/runs/${REMOTE_RUN}"
LOCAL_DIR="$ROOT/runs/${LOCAL_RUN}"

mkdir -p "$LOCAL_DIR"

# -rlptD instead of -a: skip owner/group preservation (-o -g), which fails
# with "Operation not permitted" when the local user can't chown to match
# the remote UID/GID.
rsync -rlptDvz --progress \
    -e "ssh -p $PORT -i $KEY" \
    "$REMOTE:$REMOTE_DIR/" "$LOCAL_DIR/"
