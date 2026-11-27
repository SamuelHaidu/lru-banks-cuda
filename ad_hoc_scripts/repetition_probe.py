"""Why does an lbc model repeat itself? A probe on real documents (teacher-forced) and on its own generations, with the
banks' reads ablated per layer and hop. Torch inference (`model/lbc_torch.py`), local (XPU / CPU).

    .venv/bin/python ad_hoc_scripts/repetition_probe.py runs/finewiki-47m-tbptt16/model.lbc \\
        --tokens /tmp/finewiki-val/validation.npy [--docs 24] [--device xpu] [--phases calib,gen,ablate]

calib   teacher-forced on real documents: the probability mass the model puts on (a) repeating the current token,
        (b) the induction candidate (the token after the previous occurrence of the current token), (c) any token
        already in the document, against how often the data does each. Ratio > 1: the model over-predicts copies on
        real text. Also the banks' read behaviour: the read weight on the slots written at this very step (the
        token's own key / value) and on slots written in the last 4 steps, and each read's direct logit effect on the
        current token and on the induction candidate (the "copy push" per layer / hop).
gen     continuations of the documents' first 32 tokens, sampled (temperature 0.8 top-k 50, and 1.0), against the
        documents' real continuations: tokens already in the last 128, immediate repeats, repeated 4-grams.
ablate  the same generations with the reads of chosen layers / hops zeroed (o = 0): which reads make the loops.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.lbc_torch import BOS, Decoder, LbcModel, trace  # noqa: E402

import torch  # noqa: E402


def sample(logits, temperature, top_k, top_p, rep, seen, rng) -> int:
    """Sampling on the host (no keras): repetition penalty `rep` on the `seen` tokens (CTRL: positive logits / rep,
    negative x rep), then temperature, top-k, top-p (nucleus)."""
    x = logits.float().cpu().numpy().astype(np.float64)
    if rep and rep != 1.0 and len(seen):
        s = np.asarray(sorted(set(seen)))
        x[s] = np.where(x[s] > 0, x[s] / rep, x[s] * rep)
    if temperature <= 0:
        return int(np.argmax(x))
    x = x / temperature
    if top_k:
        kth = np.partition(x, -top_k)[-top_k]
        x = np.where(x >= kth, x, -np.inf)
    q = np.exp(x - x.max())
    q /= q.sum()
    if top_p and top_p < 1.0:
        order = np.argsort(-q)
        cut = int(np.searchsorted(np.cumsum(q[order]), top_p)) + 1
        keep = np.zeros_like(q, dtype=bool)
        keep[order[:cut]] = True
        q = np.where(keep, q, 0.0)
        q /= q.sum()
    return int(rng.choice(len(q), p=q))

EOS = 1


def documents(path: str, n: int, min_len: int, max_len: int) -> list[list[int]]:
    a = np.load(path, mmap_mode="r")
    starts = np.flatnonzero(np.asarray(a[: min(len(a), 20_000_000)]) == BOS)
    docs = []
    for s, e in zip(starts, list(starts[1:]) + [len(a)]):
        d = [int(x) for x in a[s:e]]
        if len(d) >= min_len:
            docs.append(d[:max_len])
        if len(docs) >= n:
            break
    return docs


class AblDecoder(Decoder):
    """A decoder whose reads are scaled per (layer, hop) by `scale` (a device tensor: graph-safe to change)."""

    def __init__(self, model, scale: torch.Tensor, **kw):
        self.scale = scale
        super().__init__(model, **kw)

    def _step(self) -> None:
        self._hop = {}
        super()._step()

    def _read(self, li, q, tau, gate):
        hop = self._hop.get(li, 0)
        self._hop[li] = hop + 1
        p, o = super()._read(li, q, tau, gate)
        return p, o * self.scale[li, hop]


def rep_stats(ids: list[int], start: int) -> dict[str, float]:
    """Repetition of ids[start:] given everything before: in the last 128 tokens, immediate repeat, repeated 4-gram."""
    n = len(ids) - start
    if n <= 0:
        return {}
    in128 = sum(ids[t] in set(ids[max(0, t - 128):t]) for t in range(start, len(ids)))
    imm = sum(ids[t] == ids[t - 1] for t in range(start, len(ids)))
    seen, dup, tot = set(), 0, 0
    for t in range(3, len(ids)):
        g = tuple(ids[t - 3:t + 1])
        if t >= start + 3:
            tot += 1
            dup += g in seen
        seen.add(g)
    return {"in_last128": in128 / n, "immediate": imm / n, "dup_4gram": dup / max(tot, 1)}


def calibration(model: LbcModel, docs: list[list[int]]) -> None:
    L, H = model.L, model.H
    G, E = model.lens_weights()  # logit effect of a stream addition x: (x G) E^T / rms(h_final)
    Gt, Et = torch.from_numpy(G), torch.from_numpy(E)
    Ws = [[model.layers[li][k].float().cpu() for k in (("Wo", "Wo2") if model.hops[li] == 2 else ("Wo",))]
          for li in range(L)]
    sums = {k: [0.0, 0.0] for k in ("repeat current", "induction", "any seen")}
    n_pos = n_ind = 0
    acc = {k: np.zeros((L, 2)) for k in ("self", "recent", "ent", "push_cur", "push_y", "push_ind")}
    for di, ids in enumerate(docs):
        tr = trace(model, ids, max_tokens=len(ids))
        T = len(ids)
        probs = torch.softmax(torch.from_numpy(tr["logits"]), -1)
        rstd = torch.rsqrt(torch.from_numpy(tr["h"][:, -1]).pow(2).mean(-1) + 1e-6)  # the final norm's scale
        seen = torch.zeros(model.V, dtype=torch.bool)
        last: dict[int, int] = {}
        keep, xs, ys, cands = [], [], [], []  # scored positions: x -> y, induction candidate (-1: none)
        for t in range(T - 1):
            x, y = ids[t], ids[t + 1]
            if x != BOS:
                seen[x] = True
            if y != EOS and x != BOS:
                pr = probs[t]
                cand = ids[last[x] + 1] if x in last else -1
                sums["repeat current"][0] += float(pr[x]); sums["repeat current"][1] += float(y == x)
                sums["any seen"][0] += float(pr[seen].sum()); sums["any seen"][1] += float(bool(seen[y]))
                if cand >= 0:
                    sums["induction"][0] += float(pr[cand]); sums["induction"][1] += float(y == cand)
                keep.append(t); xs.append(x); ys.append(y); cands.append(cand)
            last[x] = t
        n_pos += len(keep)
        kt = np.asarray(keep)
        has = np.asarray(cands) >= 0
        n_ind += int(has.sum())
        written = tr["w"] > 0  # (T, L, H, M)
        Ex, Ey = Et[torch.tensor(xs)], Et[torch.tensor(ys)]
        Ec = Et[torch.tensor([c if c >= 0 else 0 for c in cands])]
        for li in range(L):
            rec = written[:, li].copy()
            for k in range(1, 5):  # written in this step or the 4 before
                rec[k:] |= written[:-k, li]
            for hop in range(model.hops[li]):
                p = tr["p"][kt, li, hop]  # (n, H, M)
                acc["self"][li, hop] += float((p * written[kt, li]).sum()) / H
                acc["recent"][li, hop] += float((p * rec[kt]).sum()) / H
                acc["ent"][li, hop] += float(-(p * np.log(np.where(p > 0, p, 1.0))).sum()) / H
                o = torch.from_numpy(tr["o"][kt, li, hop].reshape(len(kt), -1))
                lens = ((o @ Ws[li][hop]) @ Gt) * rstd[kt].unsqueeze(1)  # (n, E0)
                acc["push_cur"][li, hop] += float((lens * Ex).sum())
                acc["push_y"][li, hop] += float((lens * Ey).sum())
                acc["push_ind"][li, hop] += float(((lens * Ec).sum(1) * torch.from_numpy(has)).sum())
        print(f"  doc {di + 1}/{len(docs)}: {T} tokens", flush=True, end="\r")
    print()
    self_w, recent_w, ent = acc["self"], acc["recent"], acc["ent"]
    push_cur, push_ind, push_y = acc["push_cur"], acc["push_ind"], acc["push_y"]
    n_read = np.full((L, 2), float(n_pos))
    print(f"\n== calibration on real text: {n_pos} positions in {len(docs)} documents ({n_ind} with an induction "
          f"candidate)")
    print(f"{'event':16s} {'model mass':>10s} {'data rate':>10s} {'ratio':>7s}")
    for k, (pm, emp) in sums.items():
        d = n_ind if k == "induction" else n_pos
        print(f"{k:16s} {pm / d:10.4f} {emp / d:10.4f} {pm / max(emp, 1e-9):7.2f}")
    print("\n== bank reads per layer / hop (teacher-forced): read weight on the slots written at this step and in the "
          "last 5 steps,\n   read entropy (nats, over 128 slots: ln 128 = 4.85), direct logit push (mean) on the current "
          "token / induction candidate / true next token")
    print(f"{'layer hop':10s} {'self w':>7s} {'last5 w':>8s} {'entropy':>8s} {'push cur':>9s} {'push ind':>9s} "
          f"{'push y':>8s}")
    for li in range(L):
        for hop in range(model.hops[li]):
            n = max(n_read[li, hop], 1)
            print(f"L{li} hop{hop + 1}   {self_w[li, hop] / n:7.3f} {recent_w[li, hop] / n:8.3f} {ent[li, hop] / n:8.3f} "
                  f"{push_cur[li, hop] / n:9.3f} {push_ind[li, hop] / max(n_ind, 1):9.3f} "
                  f"{push_y[li, hop] / n:8.3f}")


def generate(model: LbcModel, dec: Decoder, prompt: list[int], n: int, setting, rng) -> list[int]:
    temperature, top_k, top_p, rep = setting
    dec.reset()
    logits = None
    for t in prompt:
        logits = dec.feed(t)
    out = list(prompt)
    for _ in range(n):
        nxt = sample(logits, temperature, top_k, top_p, rep, out[1:], rng)
        out.append(nxt)
        if nxt == EOS:
            break
        logits = dec.feed(nxt)
    return out


def parse_setting(x: str):
    f = x.split(":")  # T, top-k (0: off), top-p (0 or 1: off), repetition penalty (1: off)
    f += ["", "0", "0", "1"][len(f):]
    t, k, p, r = float(f[0]), int(f[1] or 0), float(f[2] or 0), float(f[3] or 1)
    return t, (k or None), (p if 0 < p < 1 else None), r


def setting_name(st) -> str:
    t, k, p, r = st
    return f"T {t}" + (f" top-k {k}" if k else "") + (f" top-p {p}" if p else "") + (f" rep {r}" if r != 1 else "")


def gen_table(model, docs, settings, ablations, n_new, prompt_len, seed, tok, show):
    L = model.L
    scale = torch.ones(L, 2, device=model.device)
    dec = AblDecoder(model, scale)
    real = [rep_stats(d[:prompt_len + n_new], prompt_len) for d in docs if len(d) >= prompt_len + 32]
    keys = ("in_last128", "immediate", "dup_4gram")
    mean = lambda rows: {k: float(np.mean([r[k] for r in rows if r])) for k in keys}  # noqa: E731
    print(f"\n== generation: {len(docs)} prompts of {prompt_len} tokens, up to {n_new} new tokens")
    print(f"{'source':44s} {'in last 128':>11s} {'immediate':>10s} {'dup 4-gram':>11s} {'mean len':>9s}")
    r = mean(real)
    print(f"{'real continuation':44s} {r['in_last128']:11.3f} {r['immediate']:10.4f} {r['dup_4gram']:11.3f}")
    for name, cells in ablations:
        scale.fill_(1.0)
        for li, hop in cells:
            scale[li, hop] = 0.0
        for st in settings:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            outs = [generate(model, dec, d[:prompt_len], n_new, st, rng) for d in docs]
            g = mean([rep_stats(o, prompt_len) for o in outs])
            ml = np.mean([len(o) - prompt_len for o in outs])
            label = f"{name}, {setting_name(st)}"
            print(f"{label:44s} {g['in_last128']:11.3f} {g['immediate']:10.4f} {g['dup_4gram']:11.3f} {ml:9.1f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
            if show and tok is not None:
                print("      e.g. " + repr(tok.decode(outs[0][prompt_len:prompt_len + 80])[:300]))


def loop_test(model: LbcModel, docs: list[list[int]], span: int = 24, reps: int = 8) -> None:
    """Self-reinforcement (Xu et al. 2022, "Learning to Break the Loop"): a real span repeated `reps` times after
    <bos>; per repetition k, the mean probability of the span's tokens and how often the argmax is the span's token.
    With the reads of all layers zeroed too: whether the banks carry the loop."""
    L = model.L
    spans = []
    for d in docs:
        if len(d) >= 64 + span:
            spans.append(d[64:64 + span])
    scale = torch.ones(L, 2, device=model.device)
    T = 1 + span * reps
    dec = AblDecoder(model, scale, record=T)
    print(f"\n== loop test: {len(spans)} real spans of {span} tokens repeated {reps} times; mean p(span token) / "
          f"argmax = span token, per repetition (the first: the span itself, no repeat yet)")
    for name, off in (("model", False), ("no reads", True)):
        scale.fill_(0.0 if off else 1.0)
        pk, ak = np.zeros(reps), np.zeros(reps)
        for s in spans:
            ids = [BOS] + s * reps
            dec.reset()
            for t in ids:
                dec.feed(t)
            h = dec.rec["h"][:T, -1]
            pr = torch.softmax(model.logits(h), -1).float().cpu()
            for t in range(1, T - 1):  # position t predicts ids[t + 1], of repetition t // span (0: the first)
                k = t // span
                y = ids[t + 1]
                pk[k] += float(pr[t, y])
                ak[k] += float(int(pr[t].argmax()) == y)
        n = len(spans) * span
        print(f"{name:10s} p:      " + " ".join(f"{x / n:.3f}" for x in pk))
        print(f"{name:10s} argmax: " + " ".join(f"{x / n:.3f}" for x in ak))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", default="/tmp/finewiki-val/validation.npy")
    ap.add_argument("--tokenizer", default="tokenizers/agentic-bpe32768/tokenizer.json")
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--new", type=int, default=192)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--phases", default="calib,gen,ablate")
    ap.add_argument("--settings", default="0.8:50,1.0",
                    help="comma list of T[:top_k[:top_p[:rep_penalty]]], e.g. 0.8:50,1.0,1.0:0:0.95,0.8:50:1:1.2")
    ap.add_argument("--show", action="store_true", help="print one sample per generation setting")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    model = LbcModel(a.checkpoint, a.device)
    print(f"{a.checkpoint}: {model.num_params / 1e6:.1f}M params, epoch {model.epoch}, {model.L} layers, hops "
          f"{model.hops}", flush=True)
    phases = set(a.phases.split(","))
    if "loop" in phases:
        loop_test(model, documents(a.tokens, a.docs, 96, 512))
    if "calib" in phases:
        calibration(model, documents(a.tokens, a.docs, 256, a.calib_len))
    docs = documents(a.tokens, a.docs, a.prompt + a.new, a.prompt + a.new)
    settings = [parse_setting(x) for x in a.settings.split(",")]
    if "gen" in phases:
        gen_table(model, docs, settings, [("model", [])], a.new, a.prompt, a.seed, tok, a.show)
    if "ablate" in phases:
        L = model.L
        cells = [(li, hop) for li in range(L) for hop in range(model.hops[li])]
        abl = [("no reads at all", cells)]
        abl += [(f"no reads in L{li}", [(li, h) for h in range(model.hops[li])]) for li in range(L)]
        gen_table(model, docs, [(0.8, 50, None, 1.0)], abl, a.new, a.prompt, a.seed, tok, a.show)


if __name__ == "__main__":
    main()
