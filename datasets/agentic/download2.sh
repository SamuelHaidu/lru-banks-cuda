#!/usr/bin/env bash
# The second pass: the parts of Open-SWE-Traces and Nemotron-SFT-OpenCode-v1 that download.sh left out (~94 GB), into
# $RAW (a separate directory, converted with: convert.py --raw $RAW --out /work/text2 --sources open-swe-traces
# nemotron-opencode). Together the two passes are the whole of both datasets.
set -euo pipefail
RAW=${RAW:-/work/raw2}
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$RAW"

get() {  # repo, local name, include patterns... ; --exclude patterns after "--"
    local repo=$1 name=$2
    shift 2
    local args=()
    local mode=--include
    for p in "$@"; do
        if [[ $p == -- ]]; then mode=--exclude; continue; fi
        args+=("$mode" "$p")
    done
    local t0=$SECONDS
    hf download "$repo" --repo-type dataset --local-dir "$RAW/$name" "${args[@]}" --quiet >/dev/null
    printf '%-28s %8s  %4ds\n' "$name" "$(du -sh "$RAW/$name" | cut -f1)" $((SECONDS - t0))
}

get nvidia/Open-SWE-Traces open-swe-traces 'data/*' -- \
    'data/claudecode/qwen38_27b/*' 'data/minisweagent/qwen36_27b/*' 'data/opencode/qwen38_27b/*' \
    'data/openhands/minimax_m25/*' 'data/openhands/deepseek_v4_flash_0731/*' 'data/sweagent/minimax_m25/*'
get nvidia/Nemotron-SFT-OpenCode-v1 nemotron-opencode 'agent_skills/*' 'bash_only_tool_skills/*' 'question_tool/*' \
    'agent_skills_question_tool/*'
echo "done: $(du -sh "$RAW" | cut -f1) in $RAW"
