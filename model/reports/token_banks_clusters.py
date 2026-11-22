"""`model report token-banks-clusters`: k-means over each token's learned (lambda, prior) in every bank.

Each token has, per bank b, a forgetting exponent lambda_b (sigmoid, 0 = flat weight that never
decays, 1 = the bank's plain kernel decay) and a prior s_b (softplus). Features per token:
`[lambda_1..lambda_n, log s_1..log s_n]`, z-scored per feature (standard scaler). Only tokens seen
at least `--min-count` times in the training split are clustered: rare tokens barely move from init.

k-means++ with restarts for every k in `--k-range` (the page switches between them); the initial k
is the best silhouette unless `--k` is given. Retention per bank is `|w_b(64^lambda)| / |w_b(1)|`:
how much weight a token keeps 64 tokens later. All array work is torch (CPU).

The page is a single offline HTML file: Tailwind CSS (`templates/report_tailwind.css`, prebuilt by `tools/report-css`,
`npm run build`) and plotly.js (bundled with the `plotly` Python package) are inlined.
"""

ARCHS = ('rope-token-banks',)

import argparse
import argparse as _ap
import html as _html
import json
from pathlib import Path
from typing import List

import torch

from model.reports._model_io import load_full_model, load_tokenizer
from model.reports.token_kernel_stepper import _story


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model report token-banks-clusters",
                                description="k-means of per-token (lambda, prior) across the banks of a rope-token-banks run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--min-count", type=int, default=100, help="Minimum training-split occurrences for a token to be clustered.")
    p.add_argument("--k", type=int, default=None, help="Number of clusters (default: best silhouette over --k-range; the curve is always computed).")
    p.add_argument("--k-range", type=int, nargs=2, default=(4, 12))
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stories", type=int, default=4, help="Validation stories to color by cluster.")
    p.add_argument("--min-tokens", type=int, default=180)
    p.add_argument("--max-tokens", type=int, default=400)
    return p


def _kmeans(x: torch.Tensor, k: int, restarts: int, gen: torch.Generator, iters: int = 100):
    """k-means++ init, Lloyd iterations; best of `restarts` by inertia. Returns (labels, centers, inertia)."""
    best = None
    n = x.shape[0]
    for _ in range(restarts):
        c = x[torch.randint(n, (1,), generator=gen)]
        for _ in range(1, k):  # k-means++: next center with probability ~ squared distance
            d2 = torch.cdist(x, c).pow(2).min(1).values
            c = torch.cat([c, x[torch.multinomial(d2 / d2.sum(), 1, generator=gen)]])
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)
            new = torch.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if torch.allclose(new, c, atol=1e-6):
                break
            c = new
        lab = torch.cdist(x, c).argmin(1)
        inertia = (x - c[lab]).pow(2).sum().item()
        if best is None or inertia < best[2]:
            best = (lab, c, inertia)
    return best


def _silhouette(dist: torch.Tensor, lab: torch.Tensor, k: int) -> float:
    """Mean silhouette from a precomputed (n, n) distance matrix."""
    onehot = torch.nn.functional.one_hot(lab, k).float()  # (n, k)
    sums = dist @ onehot  # (n, k): total distance to each cluster
    counts = onehot.sum(0)
    own = counts[lab]
    a = sums.gather(1, lab[:, None])[:, 0] / (own - 1).clamp_min(1)
    other = sums / counts[None]
    other.scatter_(1, lab[:, None], float("inf"))
    b = other.min(1).values
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
    s[own <= 1] = 0.0
    return s.mean().item()


def _group(text: str, function_words) -> str:
    t = text.strip()
    if not t:
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if not text.startswith(" "):
        return "subword / no space"
    if t.lower() in function_words:
        return "function word"
    if t[0].isupper() and t[1:].islower():
        return "capitalized (names, openers)"
    return "other word"


TEMPLATES = Path(__file__).resolve().parent / "templates"


def main(argv: List[str]) -> None:
    args = build_parser().parse_args(argv)
    from plotly.offline import get_plotlyjs

    from model import config as cfg_mod
    from model.data import load_tokens
    from model.reports.token_kernel_report import FUNCTION_WORDS

    model, config, ckpt = load_full_model(args.run_name, args.checkpoint)
    if config["arch"] != "rope-token-banks":
        raise SystemExit(f"Run {args.run_name!r} is {config['arch']!r}, not rope-token-banks.")
    tok = load_tokenizer(config)
    V, nb = model.vocabulary_size, model.num_banks
    text = [(tok.id_to_token(i) or f"<{i}>").replace("Ġ", " ").replace("Ċ", "\\n") for i in range(V)]

    tokens = load_tokens(str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train")))
    counts = torch.zeros(V, dtype=torch.long)
    for a in range(0, len(tokens), 50_000_000):  # bincount in chunks over the memory-mapped ids
        counts += torch.bincount(torch.as_tensor(tokens[a:a + 50_000_000].astype("int64")), minlength=V)[:V]

    with torch.no_grad():
        lam = torch.sigmoid(model.bank_lambda.value.detach().float().cpu())  # (nb, V)
        prior = torch.nn.functional.softplus(model.bank_prior.value.detach().float().cpu())
        dev = model.bank_lambda.value.device
        w64 = model.bank_kernel(torch.exp(lam * torch.log(torch.tensor(64.0))).to(dev)).detach().float().cpu().abs()
        w1 = model.bank_kernel(torch.ones(nb, 1, device=dev)).detach().float().cpu().abs()
        retention = w64 / (w1 + 1e-12)
        sl = slice(None) if model.general_pair_kernels else slice(1, None)
        osc = {name: getattr(model, f"osc_{name}").value.detach().float().cpu()[sl] for name in ("rate", "freq", "amp", "phase")}
        osc["rate"] = torch.nn.functional.softplus(osc["rate"])

    keep = torch.nonzero(counts >= args.min_count)[:, 0]
    if keep.numel() < 50:
        raise SystemExit(f"Only {keep.numel()} tokens with >= {args.min_count} occurrences.")
    feats = torch.cat([lam[:, keep].T, torch.log(prior[:, keep]).T], dim=1)  # (n, 2 nb)
    z = (feats - feats.mean(0)) / feats.std(0).clamp_min(1e-9)  # standard scaler

    gen = torch.Generator().manual_seed(args.seed)
    dist = torch.cdist(z, z)
    ks = sorted(set(range(args.k_range[0], args.k_range[1] + 1)) | ({args.k} if args.k else set()))
    labels, sil = {}, {}
    share = counts[keep].double()
    for k in ks:
        lab, _, inertia = _kmeans(z, k, args.restarts, gen)
        sil[k] = round(_silhouette(dist, lab, k), 4)
        # renumber clusters by their share of training text (cluster 0 = most text)
        mass = torch.zeros(k, dtype=torch.float64).index_add_(0, lab, share)
        rank = torch.empty(k, dtype=torch.long)
        rank[torch.argsort(-mass)] = torch.arange(k)
        labels[k] = rank[lab].tolist()
        print(f"k={k:2d}  silhouette {sil[k]:.4f}  inertia {inertia:.1f}")
    k0 = args.k or max(sil, key=sil.get)

    zc = z - z.mean(0)
    _, Sv, Vt = torch.linalg.svd(zc, full_matrices=False)
    pc = zc @ Vt[:2].T
    pcvar = (Sv[:2] ** 2 / (Sv ** 2).sum()).tolist()

    group_names = ["function word", "punctuation", "capitalized (names, openers)", "other word", "subword / no space", "whitespace"]
    gi = {g: i for i, g in enumerate(group_names)}
    r2 = lambda v: round(float(v), 3)
    toks = {
        "id": keep.tolist(), "t": [text[i] for i in keep.tolist()], "n": counts[keep].tolist(),
        "g": [gi[_group(text[i], FUNCTION_WORDS)] for i in keep.tolist()],
        "lam": [[r2(lam[b, i]) for b in range(nb)] for i in keep.tolist()],
        "pr": [[r2(prior[b, i]) for b in range(nb)] for i in keep.tolist()],
        "ret": [[r2(retention[b, i]) for b in range(nb)] for i in keep.tolist()],
        "z": [[round(float(v), 2) for v in row] for row in z],
        "pc": [[r2(a), r2(b)] for a, b in pc.tolist()],
    }
    pos = {int(i): j for j, i in enumerate(keep.tolist())}
    stories = []
    for skip in range(args.stories):
        ns = _ap.Namespace(text=None, min_tokens=args.min_tokens, max_tokens=args.max_tokens, doc_skip=skip)
        try:
            ids = _story(config, tok, ns)
        except SystemExit:
            break
        stories.append([[text[i], pos.get(int(i), -1)] for i in ids])

    data = {
        "run": args.run_name, "ckpt": ckpt.name, "nb": nb, "bank_lambdas": model.bank_lambdas, "window": model.window_size,
        "min_count": args.min_count, "k0": k0, "sil": sil, "labels": labels, "tokens": toks, "groups": group_names,
        "pcvar": pcvar, "osc": {k_: [[round(float(v), 5) for v in row] for row in t] for k_, t in osc.items()},
        "stories": stories, "total": int(counts.sum()),
    }
    page = (TEMPLATES / "token_banks_clusters.html").read_text()
    css = (TEMPLATES / "report_tailwind.css").read_text()
    page = (page.replace("/*__CSS__*/", css)
                .replace("/*__PLOTLY__*/", get_plotlyjs())
                .replace("__TITLE__", _html.escape(f"{args.run_name} / {ckpt.name}"))
                .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    out = ckpt.parent.parent / "reports" / "token_banks_clusters.html" if ckpt.parent.name == "checkpoints" \
        else ckpt.parent / "reports" / "token_banks_clusters.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"Wrote {out}  ({keep.numel()} tokens, k {ks[0]}..{ks[-1]}, initial k = {k0}, {len(page) / 1e6:.1f} MB)")
