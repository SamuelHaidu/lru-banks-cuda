"""Has a `gated-deltanet` run's mixture of experts collapsed? Validation rows per category through the model, the MoE
replaced by an instrumented copy (same math as `_moe`). Per layer:

- load: share of tokens per expert with the router bias (what training does) and without it (the router alone):
  balanced only with the bias means the bias, not the router, spreads the tokens;
- bias: how many choices the bias changes, its size vs the router's top-1 / top-2 logit margin;
- router: mean probability of the chosen expert (gate), routing entropy / ln E (1: uniform, the router can't tell);
- specialization: I(expert; category) and I(expert; token id) / H(expert), the top tokens per expert;
- distinct experts: all experts run on the same rows, cosine between their outputs (1: clones);
- ablations (all layers at once): loss with each token's second choice, a random expert, the gate set to 1 / E.
  Interchangeable experts cost nothing when scrambled.

    uv run python "ad hoc scripts/gdn_moe_health.py" --run-name X [--checkpoint latest] [--docs 4] [--device xpu]
        [--per-layer] [--show-layer 8] [--dataset D]   # --dataset: compare checkpoints on the same rows
"""

import argparse
import json
import math
import re

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--docs", type=int, default=4, help="validation documents per category")
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--device", default="xpu")
ap.add_argument("--dataset", default=None, help="validation rows of this dataset (default: the run's); same tokenizer")
ap.add_argument("--per-layer", action="store_true", help="also the random-routing ablation layer by layer")
ap.add_argument("--show-layer", type=int, default=None, help="top tokens per expert for this layer")
a = ap.parse_args()
setup_device(a.device)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_info(x, y, nx, ny):
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    return entropy(joint.sum(1)) + entropy(joint.sum(0)) - entropy(joint.ravel())


def main():
    config = cfg_mod.load(a.run_name)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    m = load_checkpoint(a.run_name, a.checkpoint)
    E = m.num_experts
    ds = a.dataset or config["data"]["dataset"]
    meta = json.loads((cfg_mod.DATASETS_DIR / ds / "metadata.json").read_text())
    tokens = np.load(cfg_mod.DATASETS_DIR / ds / meta["validation"]["file"], mmap_mode="r")
    index = np.load(cfg_mod.DATASETS_DIR / ds / "validation.categories.npy")
    names = {v: k for k, v in meta["category_ids"].items()}
    rng = np.random.default_rng(0)
    rows, cats = [], []
    for c in sorted(names):
        docs = index[(index[:, 2] == c) & (index[:, 1] > a.window)]
        for s, _, _ in docs[rng.choice(len(docs), min(a.docs, len(docs)), replace=False)]:
            rows.append(np.asarray(tokens[s:s + a.window + 1]).astype(np.int64))
            cats.append(c)
    rows = np.stack(rows)
    cats = np.repeat(np.asarray(cats), a.window)
    x_ids, y_ids = rows[:, :-1], rows[:, 1:]
    flat_ids = x_ids.reshape(-1)
    print(f"run {a.run_name} checkpoint {a.checkpoint}, {ds}: {len(rows)} rows x {a.window} ({a.docs} docs x "
          f"{len(names)} categories), {m.num_layers} layers x {E} experts, top-1")

    mode = {"route": "normal", "layer": None}
    rec = {}

    def moe(p, h, bias_var):
        n = int(re.search(r"l(\d+)_router_bias", bias_var.path).group(1))
        logits = h.float() @ p["router"].float()
        probs = torch.softmax(logits, -1)
        bias = p["router_bias"].float()
        choice = torch.argmax(logits + bias, -1)
        if mode["route"] == "normal":
            top2 = logits.topk(2, -1).values
            d = {"choice": choice.cpu().numpy(), "raw": logits.argmax(-1).cpu().numpy(),
                      "gate": probs.gather(1, choice[:, None])[:, 0].cpu().numpy(),
                      "ent": (-(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(E)).cpu().numpy(),
                      "margin": (top2[:, 0] - top2[:, 1]).cpu().numpy(), "bias": bias.cpu().numpy()}
            sub = h[torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(n))[:2048].to(h.device)]
            outs = torch.stack([F.gelu(sub @ p["w1"][e] + p["b1"][e], approximate="tanh") @ p["w2"][e] + p["b2"][e]
                                for e in range(E)])  # (E, R, D)
            outs = F.normalize(outs.float(), dim=-1)
            d["cos"] = torch.einsum("erd,frd->efr", outs, outs).mean(-1).cpu().numpy()
            rec.setdefault(n, []).append(d)
        gate = probs.gather(1, choice[:, None])
        if mode["layer"] is None or mode["layer"] == n:
            if mode["route"] == "second":
                masked = (logits + bias).scatter(1, choice[:, None], float("-inf"))
                choice = masked.argmax(-1)
                gate = probs.gather(1, choice[:, None])
            elif mode["route"] == "random":
                g = torch.Generator().manual_seed(1000 + n)
                choice = torch.randint(0, E, (h.shape[0],), generator=g).to(h.device)
                gate = probs.gather(1, choice[:, None])
            elif mode["route"] == "flat_gate":
                gate = torch.full_like(gate, 1.0 / E)
        y = torch.zeros(h.shape[0], p["w2"].shape[-1], device=h.device, dtype=torch.float32)
        for e in range(E):
            sel = (choice == e).nonzero().squeeze(1)
            if sel.numel():
                r = h.index_select(0, sel)
                y.index_copy_(0, sel, (F.gelu(r @ p["w1"][e] + p["b1"][e], approximate="tanh") @ p["w2"][e]
                                       + p["b2"][e]).float())
        return gate * y

    m._moe = moe
    y_t = torch.as_tensor(y_ids.reshape(-1))

    def loss(route, layer=None):
        mode.update(route=route, layer=layer)
        out = []
        with torch.no_grad():
            for b in range(0, len(x_ids), 8):
                h = m.dense_hidden(x_ids[b:b + 8].astype(np.int32))
                logits = m.readout(m.readout_input(h.reshape(-1, h.shape[-1])))
                out.append(torch.as_tensor(logits).float().cpu())
        return float(F.cross_entropy(torch.cat(out), y_t))

    base = loss("normal")
    print(f"\nloss {base:.4f}\n")
    print("layer | load % per expert (with bias)     | max/min | without bias: max/min  | bias changes | bias range"
          "   | gate  | entropy | I(e;cat) I(e;tok) | output cos max/mean")
    for n in range(m.num_layers):
        rec[n] = {k: (np.mean([d[k] for d in rec[n]], 0) if k in ("cos", "bias") else
                      np.concatenate([d[k] for d in rec[n]])) for k in rec[n][0]}
        r = rec[n]
        load = np.bincount(r["choice"], minlength=E) / len(r["choice"])
        raw = np.bincount(r["raw"], minlength=E) / len(r["raw"])
        He = entropy(load)
        mi_cat = mutual_info(r["choice"], cats, E, len(names)) / He
        mi_tok = mutual_info(r["choice"], flat_ids, E, m.vocabulary_size) / He
        cos = r["cos"][~np.eye(E, dtype=bool)]
        rawr = f"{raw.max() / max(raw.min(), 1e-9):.1f}" if raw.min() > 0 else "inf"
        print(f"{n:5d} | {' '.join(f'{100 * v:4.1f}' for v in load)} | {load.max() / load.min():7.2f} | "
              f"{' '.join(f'{100 * v:2.0f}' for v in raw)} {rawr:>4} | {100 * (r['choice'] != r['raw']).mean():10.1f}% | "
              f"{r['bias'].min():+.2f}..{r['bias'].max():+.2f} | {r['gate'].mean():.3f} | {r['ent'].mean():7.3f} | "
              f"{mi_cat:8.3f} {mi_tok:8.3f} | {cos.max():.2f} {cos.mean():.2f}")
    margin = np.concatenate([rec[n]["margin"] for n in rec])
    print(f"\nrouter top-1 / top-2 logit margin: median {np.median(margin):.2f}, p10 {np.percentile(margin, 10):.2f}")

    if a.show_layer is not None:
        r = rec[a.show_layer]
        print(f"\nlayer {a.show_layer}: categories and top tokens per expert")
        for e in range(E):
            sel = r["choice"] == e
            cat_share = np.bincount(cats[sel], minlength=len(names)) / max(1, sel.sum())
            ids, cnt = np.unique(flat_ids[sel], return_counts=True)
            # tokens most specific to the expert: share of the token's occurrences routed here, among frequent ones
            tot = np.bincount(flat_ids, minlength=m.vocabulary_size)
            spec = [(cnt[i] / tot[ids[i]], ids[i]) for i in range(len(ids)) if tot[ids[i]] >= 20]
            top = ", ".join(repr(tok.id_to_token(int(t)).replace("Ġ", " ").replace("Ċ", "\\n"))
                            + f" {s:.0%}" for s, t in sorted(spec, reverse=True)[:8])
            cs = " ".join(f"{names[c][:6]} {100 * v:.0f}" for c, v in enumerate(cat_share))
            print(f"  expert {e} ({100 * sel.mean():.1f}%): {cs}\n      {top}")

    print("\nablations (all layers):")
    for route, what in (("second", "second choice"), ("random", "random expert"), ("flat_gate", "gate 1/E")):
        lo = loss(route)
        print(f"  {what:14s} loss {lo:.4f}  ({lo - base:+.4f})")
    if a.per_layer:
        print("\nrandom expert, one layer at a time:")
        for n in range(m.num_layers):
            lo = loss("random", n)
            print(f"  layer {n:2d}: {lo:.4f} ({lo - base:+.4f})")


if __name__ == "__main__":
    main()
