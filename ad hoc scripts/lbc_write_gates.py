"""What the trained write gates of an lbc (C++) model write and skip, from `lbc dump` decisions: per token, the
fraction of heads whose write gate is open, per layer; broken down by token type and position; plus documents rendered
with the skipped tokens marked.

    build/release/lbc dump RUN --checkpoint latest --rows 16 --out /tmp/lbc_writes   (on the pod; copy the dir here)
    uv run python "ad hoc scripts/lbc_write_gates.py" /tmp/lbc_writes [--docs 3] [--tokenizer tokenizers/pretrain-mix-bpe32768]
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
    ap.add_argument("dump")
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--tokenizer", default="tokenizers/pretrain-mix-bpe32768")
    ap.add_argument("--layer", type=int, default=-1, help="render one layer's write rate (default: mean over layers)")
    a = ap.parse_args()
    d = Path(a.dump)
    cfg = {l.split()[0]: l.split()[1] for l in (d / "config.txt").read_text().split("\n") if l.strip()}
    B, T, H, L = (int(cfg[k]) for k in ("B", "T", "H", "L"))
    ids = np.fromfile(d / "ids.bin", dtype=np.int32).reshape(B, T)
    reset = np.fromfile(d / "reset.bin", dtype=np.uint8).reshape(B, T).astype(bool)
    gw = np.stack([np.fromfile(d / f"l{n}_gw.bin", dtype=np.uint8).reshape(B, T, H) for n in range(L)])  # L B T H
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    pieces = {i: tok.id_to_token(i) for i in np.unique(ids).tolist()}
    text = {i: tok.decode([i]) for i in np.unique(ids).tolist()}

    # eligible: a write is possible (not a reset, not a row's first token)
    elig = ~reset.copy()
    elig[:, 0] = False
    rate = gw.mean(-1)  # L B T: fraction of heads writing

    print(f"model epoch {cfg.get('epoch')}, {B} rows x {T}, {L} layers x {H} heads; eligible tokens {elig.sum()}")
    print("\n1. write rate per layer (fraction of (token, head) writes), and per head:")
    for n in range(L):
        per_head = gw[n][elig].mean(0)
        print(f"   layer {n}: {rate[n][elig].mean():.3f}   heads " + " ".join(f"{x:.2f}" for x in per_head))

    def category(i):
        s, p = text[i], pieces[i]
        word_start = p.startswith(("Ġ", "▁")) or s.startswith(" ")
        core = s.strip()
        if "\n" in s:
            return "newline"
        if core == "":
            return "whitespace"
        if re.fullmatch(r"\d+", core):
            return "digits"
        if re.fullmatch(r"[^\w\s]+", core):
            return "punctuation/symbols"
        if word_start and core.lower() in FUNCTION:
            return "function word"
        if word_start:
            return "word start (content)"
        return "word continuation (subword)"

    cats = defaultdict(list)
    for b in range(B):
        for t in range(T):
            if elig[b, t]:
                cats[category(int(ids[b, t]))].append((b, t))
    print("\n2. write rate by token type (layers 0 / 1 / 2 / 3), share of tokens:")
    total = elig.sum()
    for c, pos in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        bs, ts = np.array(pos).T
        r = [rate[n][bs, ts].mean() for n in range(L)]
        print(f"   {c:28s} {len(pos) / total:5.1%}   " + " / ".join(f"{x:.2f}" for x in r))

    print("\n3. write rate by position in the row (all layers):")
    for lo, hi in ((1, 16), (16, 64), (64, 128), (128, 256), (256, 512)):
        m = elig[:, lo:hi]
        print(f"   [{lo:3d}, {hi:3d}): {rate[:, :, lo:hi].mean(0)[m].mean():.3f}")

    per_tok = defaultdict(list)
    for b in range(B):
        for t in range(T):
            if elig[b, t]:
                per_tok[int(ids[b, t])].append(rate[:, b, t].mean())
    common = [(np.mean(v), len(v), i) for i, v in per_tok.items() if len(v) >= 8]
    common.sort()
    show = lambda i: repr(text[i])
    print("\n4. tokens seen >= 8 times, least written (mean over layers and heads):")
    print("   " + ", ".join(f"{show(i)} {m:.2f} (x{c})" for m, c, i in common[:15]))
    print("   most written:")
    print("   " + ", ".join(f"{show(i)} {m:.2f} (x{c})" for m, c, i in common[-15:][::-1]))

    print("\n4b. first occurrence of a token in its document vs a repeat (layers 0 / 1 / 2 / 3):")
    first, repeat = [], []
    for b in range(B):
        seen = set()
        for t in range(T):
            i = int(ids[b, t])
            if reset[b, t]:
                seen = set()
            if elig[b, t]:
                (repeat if i in seen else first).append((b, t))
            seen.add(i)
    for name, pos in (("first occurrence", first), ("repeat", repeat)):
        bs, ts = np.array(pos).T
        print(f"   {name:17s} {len(pos) / total:5.1%}   " + " / ".join(f"{rate[n][bs, ts].mean():.2f}" for n in range(L)))

    print(f"\n5. {a.docs} documents: each token with its write rate (mean over layers and heads);"
          " [token] = skipped by most heads (< 0.35), plain = written by most (> 0.65), {token} = in between")
    shown = 0
    for b in range(B):
        if shown == a.docs:
            break
        out = []
        for t in range(1, min(T, 220)):
            if reset[b, t] and t > 1:
                out.append(" <|doc|> ")
                break
            s = text[int(ids[b, t])]
            r = rate[:, b, t].mean() if a.layer < 0 else rate[a.layer, b, t]
            out.append(f"[{s}]" if r < 0.35 else (s if r > 0.65 else "{" + s + "}"))
        print(f"\n--- document {shown + 1} (row {b}) ---\n" + "".join(out))
        shown += 1


if __name__ == "__main__":
    main()
