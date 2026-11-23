#!/usr/bin/env bash
# Downloads the raw agentic sources (parquet / jsonl, as published) into $RAW, about 100 GB. Run on a data pod:
#   remote-pod cp datasets/agentic :/work/ && remote-pod sh -d --name download 'bash agentic/download.sh'
# Every source is public (no gated terms): CC BY 4.0 (NVIDIA) or Apache 2.0 (Toucan).
set -euo pipefail
RAW=${RAW:-/work/raw}
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$RAW"

get() {  # repo, local name, include patterns...
    local repo=$1 name=$2
    shift 2
    local inc=()
    for p in "$@"; do inc+=(--include "$p"); done
    local t0=$SECONDS
    hf download "$repo" --repo-type dataset --local-dir "$RAW/$name" "${inc[@]}" --quiet >/dev/null
    printf '%-28s %8s  %4ds\n' "$name" "$(du -sh "$RAW/$name" | cut -f1)" $((SECONDS - t0))
}

get nvidia/Nemotron-Terminal-Corpus terminal-corpus '*'
get nvidia/Nemotron-Agentic-v1 nemotron-agentic '*'
get nvidia/Nemotron-SWE-v1 nemotron-swe '*'
get Agent-Ark/Toucan-1.5M toucan '*'
# one model per agent framework, for diversity (~25 GB of 98)
get nvidia/Open-SWE-Traces open-swe-traces 'README.md' \
    'data/claudecode/qwen38_27b/*' 'data/minisweagent/qwen36_27b/*' 'data/opencode/qwen38_27b/*' \
    'data/openhands/minimax_m25/*' 'data/openhands/deepseek_v4_flash_0731/*' 'data/sweagent/minimax_m25/*'
get nvidia/Nemotron-SFT-OpenCode-v1 nemotron-opencode 'README.md' 'bash_only_tool/*' 'general/*'
# math with a Python tool: the tool-integrated reasoning split only
get nvidia/OpenMathReasoning openmath-tir 'README.md' 'data/tir-*'
echo "done: $(du -sh "$RAW" | cut -f1) in $RAW"
