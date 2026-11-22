"""Does the token-kernel model lock onto recent tokens and lose far context it needs?

Observation (stepper, story "Roxy"): after "...she wanted to climb it.\\n\\nRoxy tried", almost
all energy sits on "tried" (then on "tried to"): selective, Markov-1/2-like context. Good
for most steps, but when the next token is a word seen ~9 tokens back ("climb"), the
recent tokens hold the state and the far occurrence gets almost no energy.

This script measures it over many validation stories, for `rope-token-kernel-osc8`
(prior/forgetting/saturation, normalized) against `abl-exp-osc8` (same shared oscillator
kernel, no per-token numbers, same training budget):

1. how often the state is locked on the last two tokens (energy at k <= 2 above 50 / 80%);
2. targets that are a content word already in the window ("copy" targets), by distance of
   its latest occurrence: energy the selective model puts on those occurrences, and
   p(target) for both models, split by locked / not locked.

Run: .venv/bin/python studies/selective_context_check.py [--stories 200]
"""

import argparse
import sys

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv  # keep model imports from parsing our flags
from model.scripts._model_io import load_full_model, load_tokenizer  # noqa: E402
sys.argv = _argv

FUNCTION = {"the", "a", "an", "to", "and", "was", "has", "had", "of", "in", "on", "at", "it", "is", "that", "with",
            "for", "he", "she", "they", "his", "her", "but", "so", "then", "said", "were", "you", "i", "we", "not",
            "very", "too", "be", "up", "all", "one", "day", "there", "this", "what", "can", "did", "do", "him", "them"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rope-token-kernel-osc8")
    ap.add_argument("--baseline", default="abl-exp-osc8")
    ap.add_argument("--stories", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=150)
    args = ap.parse_args()

    import torch
    from keras import ops

    from model import config as cfg_mod
    from model.data import _document_bounds, load_tokens

    m, config, _ = load_full_model(args.run)
    base, _, _ = load_full_model(args.baseline)
    tok = load_tokenizer(config)
    T = m.window_size
    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "validation"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, int(meta["doc_token_id"]), True)
    V = m.vocabulary_size
    word = [(tok.id_to_token(i) or "").replace("Ġ", " ") for i in range(V)]
    content = np.array([w.strip().isalpha() and len(w.strip()) > 2 and w.strip().lower() not in FUNCTION for w in word])
    dev = m.token_embedding.embeddings.value.device
    to_np = lambda t: np.asarray(ops.convert_to_numpy(t), dtype=np.float64)
    tril = ops.convert_to_tensor(np.tril(np.ones((T, T), "float32")))
    log_k = ops.convert_to_tensor(m._log_distance)[None]

    rows = []  # per step: k2 share, top1 share, neff, logp model, logp base, copy distance, energy on target occurrences
    used = 0
    for s, e in zip(starts, ends):
        if e - s < args.min_tokens:
            continue
        ids = np.asarray(tokens[s:e], dtype=np.int64)
        n = len(ids) - 1
        x = np.zeros((n, T), np.int32)
        for i in range(n):
            ctx = ids[max(0, i + 1 - T): i + 1]
            x[i, T - len(ctx):] = ctx
        y = ids[1:]
        xt = torch.as_tensor(x, device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(m(xt).float(), -1)[torch.arange(n), torch.as_tensor(y, device=dev)].cpu().numpy()
            lpb = torch.log_softmax(base(xt).float(), -1)[torch.arange(n), torch.as_tensor(y, device=dev)].cpu().numpy()
            idsx = ops.cast(xt, "int32")
            same = ops.cast(ops.equal(idsx[:, :, None], idsx[:, None, :]), "float32")
            mention = ops.sum(same * tril[None], axis=-1)
            w = np.abs(to_np(m._weights(idsx, log_k, mention)))
        energy = w / w.sum(-1, keepdims=True)
        for i in range(n):
            en = energy[i]
            k2 = en[-2:].sum()
            # latest earlier occurrence of the target inside the real window (k >= 3)
            occ = np.where(x[i, :-2] == y[i])[0]
            n_real = min(i + 1, T)
            occ = occ[occ >= T - n_real]
            dist = (T - occ.max()) if (len(occ) and content[y[i]]) else 0
            e_tgt = en[occ].sum() if dist else 0.0
            rows.append((k2, en.max(), 1.0 / (en ** 2).sum(), lp[i], lpb[i], dist, e_tgt))
        used += 1
        if used >= args.stories:
            break

    r = np.array(rows)
    k2, top1, neff, lp, lpb, dist, etgt = r.T
    print(f"{used} stories, {len(r)} steps (window {T}); model {args.run} vs baseline {args.baseline}")
    print(f"\nall steps: loss model {-lp.mean():.3f}  baseline {-lpb.mean():.3f}")
    print(f"state concentration: median effective tokens {np.median(neff):.1f}; energy at k<=2 > 50% in "
          f"{(k2 > .5).mean() * 100:.0f}% of steps, > 80% in {(k2 > .8).mean() * 100:.0f}%; median top-1 share {np.median(top1) * 100:.0f}%")
    locked = k2 > 0.5
    for name, sel in (("locked (k<=2 > 50%)", locked), ("not locked", ~locked)):
        print(f"  {name:20s} {sel.mean() * 100:4.0f}% of steps   loss model {-lp[sel].mean():.3f}  baseline {-lpb[sel].mean():.3f}")

    cp = dist > 0
    print(f"\ncopy targets (content word already in the window at k >= 3): {cp.mean() * 100:.1f}% of steps")
    print("  distance   n       energy on its occurrences   p(target) model   p(target) baseline   model - baseline loss")
    for lo, hi in ((3, 8), (9, 32), (33, 128)):
        sel = cp & (dist >= lo) & (dist <= hi)
        if sel.sum():
            print(f"  {lo:3d}-{hi:<4d} {sel.sum():6d}   {etgt[sel].mean() * 100:8.2f}%                 {np.exp(lp[sel]).mean():.3f}             "
                  f"{np.exp(lpb[sel]).mean():.3f}               {(-lp[sel] + lpb[sel]).mean():+.3f} nats")
    q = np.quantile(k2[cp], [1 / 3, 2 / 3]) if cp.any() else [0, 0]
    print(f"  copy targets split by recent concentration (energy at k<=2, tertiles {q[0] * 100:.0f}% / {q[1] * 100:.0f}%):")
    for name, sel in (("low k<=2", cp & (k2 <= q[0])), ("mid k<=2", cp & (k2 > q[0]) & (k2 <= q[1])), ("high k<=2", cp & (k2 > q[1]))):
        print(f"  copy & {name:11s} n={sel.sum():6d}  energy on target {etgt[sel].mean() * 100:5.2f}%  "
              f"p model {np.exp(lp[sel]).mean():.3f}  p baseline {np.exp(lpb[sel]).mean():.3f}  "
              f"model - baseline {(-lp[sel] + lpb[sel]).mean():+.3f} nats")
    other = ~cp
    print(f"  non-copy steps: model - baseline {(-lp[other] + lpb[other]).mean():+.3f} nats")


if __name__ == "__main__":
    main()
