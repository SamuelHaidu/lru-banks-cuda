"""Interpretability study of an lbc (C++) language model from `lbc dump` directories (see the run's report): ablations,
depth (early exit), memory vs repeat distance, induction on repeated random sequences, failures, context use, write
gates. Prints a summary and writes ROOT/results.json for the HTML report.

    uv run python "ad hoc scripts/lbc_study.py" /tmp/study [--tokenizer tokenizers/pretrain-mix-bpe32768]
        [--freq datasets/pretrain-mix-bpe32768/validation.npy]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

FUNCTION = set("the a an of to in and or is are was were be been it its this that for on with as at by from not but "
               "if then so we you he she they i".split())
CATS = ["word start", "continuation", "function", "punct", "newline", "digits"]


def load(d):
    cfg = {l.split()[0]: l.split()[1] for l in (d / "config.txt").read_text().split("\n") if l.strip()}
    B, T = int(cfg["B"]), int(cfg["T"])
    r = {"cfg": cfg, "B": B, "T": T}
    for name, dt in (("ids", np.int32), ("y", np.int32), ("reset", np.uint8), ("row_loss", np.float32)):
        r[name] = np.fromfile(d / f"{name}.bin", dtype=dt).reshape(B, T)
    r["reset"] = r["reset"].astype(bool)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--tokenizer", default="tokenizers/pretrain-mix-bpe32768")
    ap.add_argument("--freq", default="datasets/pretrain-mix-bpe32768/validation.npy")
    a = ap.parse_args()
    root = Path(a.root)
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    base = load(root / "none")
    B, T, ids, y, reset, lb = base["B"], base["T"], base["ids"], base["y"], base["reset"], base["row_loss"]
    scored = y >= 0
    vocab_ids = np.unique(np.concatenate([ids.ravel(), y[scored]])).tolist()
    text = {i: tok.decode([i]) for i in vocab_ids}
    out = {"epoch": int(base["cfg"].get("epoch", 0)), "targets": int(scored.sum())}

    def cat(i):
        s = text[i]
        core = s.strip()
        if "\n" in s:
            return "newline"
        if re.fullmatch(r"\d+", core or "x"):
            return "digits"
        if re.fullmatch(r"[^\w\s]+", core or "x"):
            return "punct"
        if s.startswith(" ") and core.lower() in FUNCTION:
            return "function"
        if s.startswith(" "):
            return "word start"
        return "continuation"

    # per target: category, repeat (seen earlier in the document) and distance to its last occurrence, position
    cats = np.full((B, T), "", dtype=object)
    dist = np.zeros((B, T), dtype=np.int64)  # 0: new
    pos = np.zeros((B, T), dtype=np.int64)  # tokens since the document start (or row start)
    for b in range(B):
        last, start = {}, 0
        for t in range(T):
            if reset[b, t]:
                last, start = {}, t
            last[int(ids[b, t])] = t
            pos[b, t] = t - start
            if scored[b, t]:
                cats[b, t] = cat(int(y[b, t]))
                p = last.get(int(y[b, t]))
                dist[b, t] = t + 1 - p if p is not None else 0
    repeat = dist > 0
    out["intact"] = {"loss": float(lb[scored].mean()), "repeat": float(lb[scored & repeat].mean()),
                     "new": float(lb[scored & ~repeat].mean()), "repeat_share": float(repeat[scored].mean()),
                     "by_cat": {c: [float(lb[scored & (cats == c)].mean()), float((scored & (cats == c)).mean() / scored.mean())] for c in CATS}}
    print(f"epoch {out['epoch']}, {out['targets']} targets; intact loss {out['intact']['loss']:.3f} "
          f"(repeat {out['intact']['repeat']:.3f}, new {out['intact']['new']:.3f})")

    # 1. ablations
    out["ablations"] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("none", "induction")):
        la = load(d)["row_loss"]
        dl = la - lb
        label = d.name.replace("_", ":").replace("-", ",")
        per = defaultdict(list)
        for bb, tt in zip(*np.nonzero(scored)):
            per[int(y[bb, tt])].append(dl[bb, tt])
        top = sorted(((float(np.mean(v)), len(v), i) for i, v in per.items() if len(v) >= 6), reverse=True)[:10]
        out["ablations"][label] = {
            "all": float(dl[scored].mean()), "repeat": float(dl[scored & repeat].mean()),
            "new": float(dl[scored & ~repeat].mean()), "loss": float(la[scored].mean()),
            "by_cat": {c: float(dl[scored & (cats == c)].mean()) for c in CATS},
            "hurt": [[text[i], m, c] for m, c, i in top]}
    print("\nablations (loss increase all / repeat / new):")
    for k, v in out["ablations"].items():
        print(f"  {k:20s} {v['all']:+.3f} / {v['repeat']:+.3f} / {v['new']:+.3f}   hurt: " +
              ", ".join(f"{t!r} {m:+.1f}" for t, m, c in v["hurt"][:5]))

    # 2. memory vs distance of the repeat
    bins = [(1, 4), (5, 16), (17, 64), (65, 256), (257, 512)]
    out["distance"] = []
    variants = {"intact": lb}
    for k in ("0:1", "0:2,1:2,2:2,3:2"):
        d = root / k.replace(":", "_").replace(",", "-")
        if d.exists():
            variants[k] = load(d)["row_loss"]
    for lo, hi in bins:
        m = scored & (dist >= lo) & (dist <= hi)
        out["distance"].append({"range": [lo, hi], "count": int(m.sum()),
                                **{k: float(v[m].mean()) for k, v in variants.items()}})
    out["distance_new"] = {k: float(v[scored & ~repeat].mean()) for k, v in variants.items()}
    print("\nloss of repeated tokens by distance to their previous occurrence:")
    for r in out["distance"]:
        print(f"  {r['range']}: n={r['count']:5d}  " + "  ".join(f"{k} {r[k]:.2f}" for k in variants))

    # 3. induction: documents <bos> s s <eos>, |s| = 100
    ind = load(root / "induction")
    il, iy, ir = ind["row_loss"], ind["y"], ind["reset"]
    first, second = defaultdict(list), defaultdict(list)
    for b in range(ind["B"]):
        starts = [t for t in range(ind["T"]) if ir[b, t]] + [ind["T"]]
        for s0, s1 in zip(starts, starts[1:]):
            for t in range(s0, s1):
                k = t - s0  # input index in the document: 1..100 first copy, 101..200 second
                if iy[b, t] < 0 or t + 1 >= ind["T"]:
                    continue
                if 1 <= k <= 99:
                    first[k].append(il[b, t])
                elif 100 <= k <= 199:
                    second[k - 100].append(il[b, t])
    prof = lambda dd, ks: [float(np.mean(dd[k])) if dd[k] else None for k in ks]
    out["induction"] = {"first": prof(first, range(1, 100)), "second": prof(second, range(0, 100)),
                        "first_mean": float(np.mean([x for v in first.values() for x in v])),
                        "second_mean": float(np.mean([x for v in second.values() for x in v])),
                        "uniform": float(np.log(32768))}
    print(f"\ninduction (random 100-token sequence repeated): first copy {out['induction']['first_mean']:.2f}, "
          f"second copy {out['induction']['second_mean']:.2f} (uniform {np.log(32768):.2f})")

    # 4. failures
    counts = Counter()
    try:
        raw = open(a.freq, "rb").read()
        hl = int.from_bytes(raw[8:10], "little")
        arr = np.frombuffer(raw[10 + hl:], dtype=np.uint16)
        counts = Counter(dict(zip(*np.unique(arr, return_counts=True))))
        total = len(arr)
    except OSError:
        total = 1
    freq_bins = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1)]
    out["by_freq"] = []
    fr = np.vectorize(lambda i: counts.get(i, 0) / total)(np.where(scored, y, 0))
    for lo, hi in freq_bins:
        m = scored & (fr >= lo) & (fr < hi)
        if m.sum():
            out["by_freq"].append({"range": [lo, hi], "count": int(m.sum()), "loss": float(lb[m].mean()),
                                   "repeat": float(lb[m & repeat].mean()) if (m & repeat).sum() else None,
                                   "new": float(lb[m & ~repeat].mean()) if (m & ~repeat).sum() else None})
    print("\nloss by target frequency (share of the validation split):")
    for r in out["by_freq"]:
        print(f"  {r['range']}: n={r['count']:5d} loss {r['loss']:.2f}  repeat {r['repeat']}  new {r['new']}")
    per = defaultdict(list)
    for bb, tt in zip(*np.nonzero(scored)):
        per[int(y[bb, tt])].append(lb[bb, tt])
    hard = sorted(((float(np.mean(v)), len(v), i) for i, v in per.items() if len(v) >= 8), reverse=True)[:20]
    easy = sorted(((float(np.mean(v)), len(v), i) for i, v in per.items() if len(v) >= 30))[:12]
    out["hard_tokens"] = [[text[i], m, c] for m, c, i in hard]
    out["easy_tokens"] = [[text[i], m, c] for m, c, i in easy]
    # worst single predictions with context (new tokens and repeats)
    examples = []
    order = np.argsort(-np.where(scored, lb, -1).ravel())
    used_rows = Counter()
    for flat in order:
        bb, tt = divmod(int(flat), T)
        if used_rows[bb] >= 2 or not scored[bb, tt]:
            continue
        lo = max(0, tt - 14)
        while lo < tt and reset[bb, lo] and lo + 1 <= tt:
            lo += 1
        ctx = tok.decode(ids[bb, lo:tt + 1].tolist())
        examples.append({"context": ctx[-90:], "target": text[int(y[bb, tt])], "loss": float(lb[bb, tt]),
                         "repeat_distance": int(dist[bb, tt])})
        used_rows[bb] += 1
        if len(examples) == 14:
            break
    out["worst_examples"] = examples
    # repeats the model still misses: repeated within 64 tokens but loss > 5
    misses = []
    for bb, tt in zip(*np.nonzero(scored & (dist > 0) & (dist <= 64) & (lb > 5))):
        lo = max(0, tt - 12)
        misses.append({"context": tok.decode(ids[bb, lo:tt + 1].tolist())[-80:], "target": text[int(y[bb, tt])],
                       "loss": float(lb[bb, tt]), "repeat_distance": int(dist[bb, tt])})
    misses.sort(key=lambda e: -e["loss"])
    out["missed_repeats"] = misses[:10]
    out["missed_repeat_share"] = float(((dist > 0) & (dist <= 64) & (lb > 5) & scored).sum() /
                                       max(1, ((dist > 0) & (dist <= 64) & scored).sum()))
    print("\nhardest tokens:", ", ".join(f"{t!r} {m:.1f}" for t, m, c in out["hard_tokens"][:12]))
    print("worst predictions:")
    for e in examples[:8]:
        print(f"  {e['loss']:5.1f}  ...{e['context'][-60:]!r} -> {e['target']!r}  (repeat distance {e['repeat_distance']})")
    print(f"repeats within 64 tokens still missed (loss > 5): {out['missed_repeat_share']:.1%}")

    # 5. context use: loss by position in the document
    pbins = [(0, 8), (8, 32), (32, 128), (128, 256), (256, 512)]
    out["by_position"] = [{"range": [lo, hi], "loss": float(lb[scored & (pos >= lo) & (pos < hi)].mean()),
                           "count": int((scored & (pos >= lo) & (pos < hi)).sum())} for lo, hi in pbins]
    print("\nloss by position in the document:", ", ".join(f"{r['range']}: {r['loss']:.2f}" for r in out["by_position"]))

    # 6. write gates (intact dump)
    gwd = root / "none"
    if (gwd / "l0_gw.bin").exists():
        H, L = int(base["cfg"]["H"]), int(base["cfg"]["L"])
        gw = np.stack([np.fromfile(gwd / f"l{n}_gw.bin", dtype=np.uint8).reshape(B, T, H) for n in range(L)])
        rate = gw.mean(-1)
        elig = ~reset.copy()
        elig[:, 0] = False
        incat = np.full((B, T), "", dtype=object)
        inrep = np.zeros((B, T), dtype=bool)
        for b in range(B):
            seen = set()
            for t in range(T):
                if reset[b, t]:
                    seen = set()
                i = int(ids[b, t])
                incat[b, t] = cat(i) if i in text else ""
                inrep[b, t] = i in seen
                seen.add(i)
        out["writes"] = {
            "per_layer": [float(rate[n][elig].mean()) for n in range(L)],
            "per_head": [[float(x) for x in gw[n][elig].mean(0)] for n in range(L)],
            "by_cat": {c: [float(rate[n][elig & (incat == c)].mean()) for n in range(L)] for c in CATS},
            "first": [float(rate[n][elig & ~inrep].mean()) for n in range(L)],
            "repeat": [float(rate[n][elig & inrep].mean()) for n in range(L)]}
        # a sample document rendered with layer-0 write rates
        b = next(b for b in range(B) if len(re.findall(r" the | of | and ", tok.decode(ids[b, :200].tolist()))) >= 6)
        sample = []
        for t in range(1, 160):
            if reset[b, t]:
                break
            sample.append([text.get(int(ids[b, t]), tok.decode([int(ids[b, t])])), float(rate[0, b, t])])
        out["writes"]["sample"] = sample
        print("\nwrites per layer:", " ".join(f"{x:.2f}" for x in out["writes"]["per_layer"]))

    (root / "results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {root / 'results.json'}")


if __name__ == "__main__":
    main()
