"""0-shot multiple-choice benchmarks for an lbc model, scored like lm-eval-harness: each choice's continuation
log-probability given the prompt (acc), and divided by the continuation's length in characters (acc_norm).

    uv run python ad_hoc_scripts/lbc_bench.py make arc_easy /tmp/lbc/arc_easy arc_easy_test.json   (pairs.txt, meta.json)
    lbc score RUN --pairs pairs.txt --out scores.txt                                     (on the pod)
    uv run python ad_hoc_scripts/lbc_bench.py report /tmp/lbc/arc_easy scores.txt

Prompts and tokenization follow lm-eval: "Question: {question}\nAnswer:" + " {choice}", the pair encoded together and
split at the context's length; the context starts with <bos> (id 0), as a training document does.
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

TOK = Tokenizer.from_file("tokenizers/pretrain-mix-bpe32768/tokenizer.json")


def encode_pair(context, continuation):
    n = len(context) - len(context.rstrip())  # trailing spaces move to the continuation (lm-eval)
    if n:
        continuation, context = context[-n:] + continuation, context[:-n]
    whole = TOK.encode(context + continuation).ids
    ctx = TOK.encode(context).ids
    return [0] + ctx, whole[len(ctx):]


def load(task, path):
    """`path`: the test split as JSON rows (from the Hugging Face dataset viewer API: datasets-server.huggingface.co)."""
    if task == "arc_easy":
        for d in json.load(open(path)):
            yield (f"Question: {d['question']}\nAnswer:", [" " + c for c in d["choices"]["text"]],
                   d["choices"]["label"].index(d["answerKey"]))
    else:
        raise SystemExit(f"unknown task {task}")


def make(task, out, path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    with open(out / "pairs.txt", "w") as f:
        for ctx, choices, gold in load(task, path):
            items.append({"gold": gold, "lens": [len(c) for c in choices]})
            for c in choices:
                a, b = encode_pair(ctx, c)
                f.write(" ".join(map(str, a)) + " | " + " ".join(map(str, b)) + "\n")
    json.dump({"task": task, "items": items}, open(out / "meta.json", "w"))
    print(f"{task}: {len(items)} questions, {sum(len(i['lens']) for i in items)} pairs -> {out}/pairs.txt")


def report(d, scores):
    meta = json.load(open(Path(d) / "meta.json"))
    ll = [float(l.split()[0]) for l in open(scores) if l.strip()]
    i = acc = accn = 0
    for it in meta["items"]:
        k = len(it["lens"])
        s = ll[i:i + k]
        i += k
        acc += max(range(k), key=lambda j: s[j]) == it["gold"]
        accn += max(range(k), key=lambda j: s[j] / it["lens"][j]) == it["gold"]
    n = len(meta["items"])
    print(f"{meta['task']}: {n} questions  acc {100 * acc / n:.2f}  acc_norm {100 * accn / n:.2f}")


if __name__ == "__main__":
    {"make": lambda: make(sys.argv[2], sys.argv[3], sys.argv[4]), "report": lambda: report(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
