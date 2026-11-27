"""A generation smoke test on a run's checkpoint: tokenizes a few agentic prompts, runs `lbc generate`, decodes.

    uv run --no-project --with tokenizers python scripts/smoke_generate.py RUN_DIR TOKENIZER_JSON [--checkpoint latest] [--direct | --simple] [--seed N]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

run, tok_path = sys.argv[1], sys.argv[2]
ckpt = sys.argv[sys.argv.index("--checkpoint") + 1] if "--checkpoint" in sys.argv else "latest"
tok = Tokenizer.from_file(tok_path)
PROMPTS = [
    "<bos><tools>\npython(code: str): run Python 3 code (sympy, numpy available); returns what it prints\n</tools>"
    "<user>What is the sum of the squares of the first 20 positive integers?</user><assistant><think>",
    "<bos><system>You are a Claude agent, built on Anthropic's Claude Agent SDK.CWD: /testbed</system><tools>\n"
    "Bash(command: string, description?: string): execute shell commands\n"
    "Read(file_path: string): read a file\n</tools>"
    "<user>Find the Python files in /testbed that import yaml and show the first one.</user><assistant>",
    "<bos><user>Write a Python function that checks whether a number is prime.</user><assistant>",
    "<bos>def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    # 64+ token dependencies: a made-up tool defined ~90 tokens before the answer must call it; a variable name
    "<bos><tools>\nfetch_weather_report(city: string, days: integer): the weather forecast for a city\n</tools>"
    "<user>Hi! I am planning a trip with my family next week and we still have not decided where to go. We like "
    "the beach, but my kids hate the rain, and my wife wants a quiet place. We are thinking about Lisbon, maybe "
    "for four or five days. Could you check how the weather will be there?</user><assistant>",
    "<bos>import numpy as np\n\nzebra_matrix_total = np.zeros((4, 4))\n# fill the diagonal, then add a constant "
    "to every cell, then normalize the rows so each sums to one; the steps below are written one per line so that "
    "they are easy to read and to test separately in the notebook\nfor i in range(4):\n    ",
]
if "--direct" in sys.argv:  # no reasoning: the turn pre-filled with an empty <think></think>, or with <call>
    weather = PROMPTS[4]
    PROMPTS = [weather + "<think></think>", weather + "<think></think><call>",
               PROMPTS[2] + "<think></think>", PROMPTS[1] + "<think></think><call>"]
if "--simple" in sys.argv:  # one tool, a one-line request; each also with the reasoning skipped
    simple = [
        "<bos><tools>\nget_weather(city: string): the current weather in a city\n</tools>"
        "<user>What is the weather in Paris?</user><assistant>",
        "<bos><tools>\nBash(command: string): execute shell commands\n</tools>"
        "<user>Check if the file notes.txt has abc written in it.</user><assistant>",
        "<bos><tools>\npython(code: str): run Python 3 code; returns what it prints\n</tools>"
        "<user>What is 17 * 23?</user><assistant>",
    ]
    PROMPTS = simple + [x + "<think></think>" for x in simple]
seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else "1"
with tempfile.TemporaryDirectory() as d:
    ids, out = Path(d) / "ids.txt", Path(d) / "out.txt"
    ids.write_text("\n".join(" ".join(map(str, tok.encode(p, add_special_tokens=False).ids)) for p in PROMPTS) + "\n")
    subprocess.run(["build/release/lbc", "generate", run, "--checkpoint", ckpt, "--ids", str(ids), "--out", str(out),
                    "--tokens", "160", "--temperature", "0.7", "--top-k", "40", "--seed", seed], check=True)
    for p, line in zip(PROMPTS, out.read_text().splitlines()):
        gen = tok.decode([int(x) for x in line.split()], skip_special_tokens=False)
        print(f"===== prompt: {p[-90:]!r}\n----- generated:\n{gen}\n")
