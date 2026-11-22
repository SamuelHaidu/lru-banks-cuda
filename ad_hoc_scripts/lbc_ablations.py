"""Per-token effects of silencing parts of an lbc model, from `lbc dump --silence ...` directories (each with
row_loss.bin): the loss increase overall, on tokens repeated from earlier in the document vs new ones, by token type,
and the tokens hurt most (decoded). The directory `none` is the intact model.

    uv run python ad_hoc_scripts/lbc_ablations.py /tmp/abl [--tokenizer tokenizers/pretrain-mix-bpe32768]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

FUNCTION = set("the a an of to in and or is are was were be been it its this that for on with as at by from not but "
               "if then so we you he she they i".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--tokenizer", default="tokenizers/pretrain-mix-bpe32768")
    a = ap.parse_args()
    root = Path(a.root)
    base = root / "none"
    cfg = {l.split()[0]: l.split()[1] for l in (base / "config.txt").read_text().split("\n") if l.strip()}
    B, T = int(cfg["B"]), int(cfg["T"])
    ids = np.fromfile(base / "ids.bin", dtype=np.int32).reshape(B, T)
    y = np.fromfile(base / "y.bin", dtype=np.int32).reshape(B, T)
    reset = np.fromfile(base / "reset.bin", dtype=np.uint8).reshape(B, T).astype(bool)
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    text = {i: tok.decode([i]) for i in np.unique(np.concatenate([ids.ravel(), y[y >= 0]])).tolist()}

    # target classes: the predicted token y[b, t] seen earlier in its document (repeat) or not (new)
    scored = y >= 0
    repeat = np.zeros_like(scored)
    for b in range(B):
        seen = set()
        for t in range(T):
            if reset[b, t]:
                seen = set()
            seen.add(int(ids[b, t]))
            if scored[b, t]:
                repeat[b, t] = int(y[b, t]) in seen

    def cat(i):
        s = text[i]
        core = s.strip()
        if "\n" in s:
            return "newline"
        if re.fullmatch(r"[^\w\s]+", core or "x"):
            return "punct"
        if s.startswith(" ") and core.lower() in FUNCTION:
            return "function"
        if s.startswith(" "):
            return "word start"
        return "continuation"

    cats = np.full((B, T), "", dtype=object)
    for b in range(B):
        for t in range(T):
            if scored[b, t]:
                cats[b, t] = cat(int(y[b, t]))
    names = ["word start", "continuation", "function", "punct", "newline"]

    lb = np.fromfile(base / "row_loss.bin", dtype=np.float32).reshape(B, T)
    print(f"intact model: loss {lb[scored].mean():.3f} over {scored.sum()} targets "
          f"(repeat {lb[scored & repeat].mean():.3f}, {repeat[scored].mean():.0%} of targets; "
          f"new {lb[scored & ~repeat].mean():.3f})")
    print(f"\n{'silenced':22s} {'all':>6s} {'repeat':>7s} {'new':>6s}  " + " ".join(f"{n:>12s}" for n in names))
    order = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "none")
    hurt = {}
    for name in order:
        la = np.fromfile(root / name / "row_loss.bin", dtype=np.float32).reshape(B, T)
        dl = la - lb
        row = [dl[scored].mean(), dl[scored & repeat].mean(), dl[scored & ~repeat].mean()]
        bycat = [dl[scored & (cats == c)].mean() for c in names]
        label = name.replace("_", ":").replace("-", ",")
        print(f"{label:22s} " + " ".join(f"{x:+6.3f}" for x in row[:1]) + f" {row[1]:+7.3f} {row[2]:+6.3f}  " +
              " ".join(f"{x:+12.3f}" for x in bycat))
        per = defaultdict(list)
        for b, t in zip(*np.nonzero(scored)):
            per[int(y[b, t])].append(dl[b, t])
        hurt[label] = sorted(((np.mean(v), len(v), i) for i, v in per.items() if len(v) >= 6), reverse=True)[:10]
    print("\ntokens hurt most (mean loss increase, count >= 6):")
    for label, lst in hurt.items():
        print(f"  {label:20s} " + ", ".join(f"{text[i]!r} {m:+.2f}" for m, c, i in lst[:8]))


if __name__ == "__main__":
    main()
