#!/usr/bin/env bash
# Generation from rsm-8L-768-tbptt in the training format (agent-mix-nemotron-bpe1024): system, tools, user, then the
# model writes the assistant turn. It stops at </assistant> (the end of what the loss covers); past it the training
# text continues with a <result> the model was never asked to predict.
#
#   bash ad_hoc_scripts/generate_tbptt.sh [-u "task text"] [run-name] [max-new-tokens]
#
# -u / --user sets the task the user asks for (default: create hello.txt); the terminal state line stays.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK='Create a file named hello.txt in the current directory containing the single line "hello world", then print its contents.'
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -u|--user) TASK="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
RUN="${ARGS[0]:-rsm-8L-768-tbptt}"
MAX="${ARGS[1]:-300}"

SYSTEM='You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.'
TOOLS='terminal(keystrokes: str, duration: float): send keystrokes to the terminal verbatim (end a command with \n; C-c, C-d for Ctrl+C, Ctrl+D), wait duration seconds, return the screen
task_complete(): mark the task as complete (it is then graded)'
USER="## Task
${TASK}

Current terminal state:
root@host:/app# "

PROMPT="<bos><system>${SYSTEM}</system><tools>
${TOOLS}
</tools><user>${USER}</user><assistant><think>"

uv run model generate --run-name "$RUN" --checkpoint best --prompt "$PROMPT" \
  --max-new-tokens "$MAX" --temperature 0.7 --top-k 40 --repetition-penalty 1.1 \
  --stop-tokens "</assistant>" "<eos>" --pretty
