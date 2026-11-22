"""`model report log-gaussian-kernel-trace`: what distances does the adaptive kernel pick?

For `rope-log-gaussian-sum`, samples validation windows and reads the two
distance kernels the model applies to each one:

    static   a_k            (one learned vector, same for every input)
    adaptive w_k(x)         (sum of 8 log-Gaussians, amplitudes from the MLP)
    combined a_k + w_k(x)   (what actually weights the context)

and reports:

- the mean kernels and the spread of the combined kernel across inputs;
- the amplitude distribution per Gaussian center;
- a per-input "long-context share": the fraction of combined kernel mass at
  distance k > `--long-k`, broken down by prefix length, by the n-1 token,
  and by whether the target token appears far back in the context;
- the loss with per-input amplitudes vs. dataset-mean amplitudes vs. no
  adaptive pass, i.e. what the input dependence is actually worth.
"""

ARCHS = ('rope-log-gaussian-sum',)

import argparse
import datetime as dt
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from model.device import setup_backend

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.archs.rope_log_gaussian_sum import RopeLogGaussianSumLM  # noqa: E402
from model.data import _document_bounds, _fill_windows, load_tokens  # noqa: E402
from model.reports._model_io import load_full_model, load_tokenizer  # noqa: E402
from model.reports._report import plotly_div, render_report, token_label  # noqa: E402


def sample_windows(config: Dict[str, Any], window: int, n: int, seed: int):
    """Validation windows exactly as the sampler builds them, plus prefix length."""
    name = config["data"]["dataset"]
    meta = cfg_mod.load_dataset_metadata(name)
    path = str(cfg_mod.dataset_split_path(name, "validation"))
    tokens = load_tokens(path)
    starts, ends = _document_bounds(path, tokens, int(meta["doc_token_id"]), True)
    rng = np.random.default_rng(seed)
    docs = rng.integers(0, len(starts), size=n)
    targets = rng.integers(starts[docs] + 1, ends[docs])
    x, valid = _fill_windows(tokens, targets, starts[docs], window)
    y = np.asarray(tokens[targets]).astype(np.int64)
    return x, y, valid.sum(axis=1)


def _nll(logits, y):
    """Per-sample cross-entropy. Written out because torch's `reduction="none"`
    kernel exceeds the work-group limit on the Iris Xe XPU."""
    import torch

    return torch.logsumexp(logits, dim=-1) - logits.gather(1, y[:, None])[:, 0]


def run(run_name: str, checkpoint: str, num_samples: int, batch_size: int, long_k: int, min_count: int, seed: int):
    import torch
    import plotly.graph_objects as go

    model, config, ckpt_path = load_full_model(run_name, checkpoint)
    if not isinstance(model, RopeLogGaussianSumLM):
        raise SystemExit(f"Run {run_name!r} is {config['arch']!r}; this script needs 'rope-log-gaussian-sum'.")
    tokenizer = load_tokenizer(config)
    T = model.window_size
    labels = model.basis_labels()
    signed = model.kernel_basis != "log-gaussian"  # oscillator amplitudes/weights can be negative
    ytype = "linear" if signed else "log"

    x_np, y_np, prefix_len = sample_windows(config, T, num_samples, seed)
    device = model.distance_weights.value.device

    alphas, adaptive, losses_input, h_parts = [], [], [], []
    with torch.no_grad():
        for s in range(0, num_samples, batch_size):
            xb = torch.as_tensor(x_np[s : s + batch_size], device=device)
            yb = torch.as_tensor(y_np[s : s + batch_size], device=device)
            rotated, h0, alpha, w = model.kernels(xb)
            h = torch.einsum("bt,btd->bd", w, rotated)
            losses_input.append(_nll(model.readout(model.readout_input(h0 + h)), yb).cpu())
            alphas.append(alpha.cpu())
            adaptive.append(w.cpu())
            h_parts.append((xb, yb))
        alpha_all = torch.cat(alphas)
        alpha_mean = alpha_all.mean(0, keepdim=True).to(device)
        basis = torch.as_tensor(model._basis, device=device)
        losses_mean, losses_static = [], []
        for xb, yb in h_parts:
            rotated, h0, _, _ = model.kernels(xb)
            w_mean = (alpha_mean @ basis.T).expand(xb.shape[0], -1)
            h_mean = torch.einsum("bt,btd->bd", w_mean, rotated)
            losses_mean.append(_nll(model.readout(model.readout_input(h0 + h_mean)), yb).cpu())
            losses_static.append(_nll(model.readout(model.readout_input(h0)), yb).cpu())
        static = model.distance_weights.value.detach().cpu().numpy()

    alpha_np = alpha_all.numpy()
    # Reorder the context axis to distance order: column 0 = k=1 (n-1).
    static_k = static[::-1]
    adaptive_k = torch.cat(adaptive).numpy()[:, ::-1]
    combined_k = static_k[None, :] + adaptive_k
    k = np.arange(1, T + 1)

    mass = combined_k / combined_k.sum(axis=1, keepdims=True)
    long_share = mass[:, k > long_k].sum(axis=1)
    k50 = 1 + (np.cumsum(mass, axis=1) < 0.5).sum(axis=1)

    loss = {
        "per_input": float(torch.cat(losses_input).mean()),
        "mean_alpha": float(torch.cat(losses_mean).mean()),
        "static_only": float(torch.cat(losses_static).mean()),
    }

    # -- plots ---------------------------------------------------------------
    figs = {}
    p10, p50, p90 = np.percentile(combined_k, [10, 50, 90], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k, y=p10, fill="tonexty", line=dict(width=0), name="combined p10-p90", fillcolor="rgba(47,95,163,0.2)"))
    fig.add_trace(go.Scatter(x=k, y=p50, name="combined median", line=dict(color="#2f5fa3")))
    fig.add_trace(go.Scatter(x=k, y=static_k, name="static a_k", line=dict(dash="dash", color="#8a97a3")))
    fig.add_trace(go.Scatter(x=k, y=adaptive_k.mean(0), name="adaptive w_k (mean)", line=dict(color="#2f7d4f")))
    fig.update_layout(xaxis=dict(title="distance k (1 = n-1)", type="log"), yaxis=dict(title="weight", type=ytype), height=460, margin=dict(t=20))
    figs["kernels"] = plotly_div(fig, "kernels", include_js=True)

    fig = go.Figure()
    for j, name in enumerate(labels):
        fig.add_trace(go.Box(y=alpha_np[:, j], name=name, boxpoints=False))
    fig.update_layout(yaxis=dict(title="amplitude alpha_j", type=ytype), showlegend=False, height=380, margin=dict(t=20))
    figs["alpha"] = plotly_div(fig, "alpha")

    fig = go.Figure(go.Histogram(x=long_share, nbinsx=60))
    fig.update_layout(xaxis=dict(title=f"long-context share (mass at k > {long_k})"), yaxis=dict(title="inputs"), height=320, margin=dict(t=20))
    figs["hist"] = plotly_div(fig, "hist")

    bins = [(1, 4), (5, 16), (17, 64), (65, T - 1), (T, T)]
    by_prefix = []
    for lo, hi in bins:
        sel = (prefix_len >= lo) & (prefix_len <= hi)
        if sel.any():
            by_prefix.append({"label": f"{lo}" if lo == hi else f"{lo}-{hi}", "n": int(sel.sum()),
                              "long": float(long_share[sel].mean()), "k50": float(np.median(k50[sel]))})
    fig = go.Figure(go.Bar(x=[b["label"] for b in by_prefix], y=[b["long"] for b in by_prefix]))
    fig.update_layout(xaxis=dict(title="real (non-padding) context tokens"), yaxis=dict(title="mean long-context share"), height=320, margin=dict(t=20))
    figs["prefix"] = plotly_div(fig, "prefix")

    # -- groupings -------------------------------------------------------------
    full = prefix_len >= T  # compare tokens only on full windows, so padding does not confound
    last_tok = x_np[:, -1]
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in np.where(full)[0]:
        groups[int(last_tok[i])].append(i)
    rows = [(tid, len(ix), float(long_share[ix].mean()), float(np.median(k50[ix])))
            for tid, ix in groups.items() if len(ix) >= min_count]
    rows.sort(key=lambda r: r[2])
    label = (lambda t: token_label(tokenizer, t)) if tokenizer else str
    fmt = lambda rs: [{"token": label(t), "n": n, "long": l, "k50": q} for t, n, l, q in rs]
    shortest, longest = fmt(rows[:15]), fmt(rows[::-1][:15])

    # Does the kernel reach further when the target is copyable from far back?
    far = x_np[:, : T - long_k]  # context indices at distance > long_k
    target_far = (far == y_np[:, None]).any(axis=1) & full
    target_none = ~(x_np == y_np[:, None]).any(axis=1) & full
    copy_rows = [
        {"label": f"target appears at k > {long_k}", "n": int(target_far.sum()), "long": float(long_share[target_far].mean()) if target_far.any() else float("nan")},
        {"label": "target absent from context", "n": int(target_none.sum()), "long": float(long_share[target_none].mean()) if target_none.any() else float("nan")},
    ]

    # Example contexts: extremes of long-context share among full windows.
    def example(i: int) -> Dict[str, Any]:
        tail = x_np[i, -32:]
        text = tokenizer.decode([int(t) for t in tail]) if tokenizer else " ".join(map(str, tail))
        return {"long": float(long_share[i]), "k50": int(k50[i]), "text": text, "target": label(int(y_np[i]))}
    full_ix = np.where(full)[0]
    order = full_ix[np.argsort(long_share[full_ix])]
    examples_short = [example(i) for i in order[:4]]
    examples_long = [example(i) for i in order[::-1][:4]]
    fig = go.Figure()
    for i, color in [(i, "#b23b3b") for i in order[::-1][:4]] + [(i, "#2f5fa3") for i in order[:4]]:
        fig.add_trace(go.Scatter(x=k, y=mass[i], line=dict(color=color, width=1), showlegend=False,
                                 hovertemplate=f"long share {long_share[i]:.2f}<br>k %{{x}}<br>mass %{{y:.4f}}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="distance k", type="log"), yaxis=dict(title="normalized kernel mass", type=ytype), height=380, margin=dict(t=20))
    figs["examples"] = plotly_div(fig, "examples")

    cv = alpha_np.std(0) / alpha_np.mean(0)
    context = {
        "run_name": run_name,
        "checkpoint": ckpt_path.name,
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_samples": num_samples,
        "num_full": int(full.sum()),
        "long_k": long_k,
        "min_count": min_count,
        "loss": loss,
        "figs": figs,
        "kernel_basis": model.kernel_basis,
        "cv": [{"center": c, "mean": float(m_), "cv": float(v)} for c, m_, v in zip(labels, alpha_np.mean(0), cv)],
        "long_share_mean": float(long_share.mean()),
        "long_share_p10": float(np.percentile(long_share, 10)),
        "long_share_p90": float(np.percentile(long_share, 90)),
        "k50_median": float(np.median(k50)),
        "adaptive_share": float((adaptive_k.sum(1) / combined_k.sum(1)).mean()),
        "by_prefix": by_prefix,
        "shortest": shortest,
        "longest": longest,
        "copy_rows": copy_rows,
        "examples_short": examples_short,
        "examples_long": examples_long,
    }
    out_path = cfg_mod.run_dir(run_name) / "reports" / "log_gaussian_kernel_trace.html"
    render_report("log_gaussian_kernel_trace.html", context, out_path)
    return out_path, context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model report log-gaussian-kernel-trace",
        description="Which context distances the adaptive log-Gaussian kernel weights, per input.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    parser.add_argument("--num-samples", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--long-k", type=int, default=16, help="Distances above this count as long context.")
    parser.add_argument("--min-count", type=int, default=30, help="Min occurrences for an n-1 token to be ranked.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_path, ctx = run(args.run_name, args.checkpoint, args.num_samples, args.batch_size,
                        args.long_k, args.min_count, args.seed)
    l = ctx["loss"]
    print(f"loss per-input {l['per_input']:.4f} | mean-alpha {l['mean_alpha']:.4f} | static-only {l['static_only']:.4f}")
    print(f"long-context share mean {ctx['long_share_mean']:.3f} (p10 {ctx['long_share_p10']:.3f}, p90 {ctx['long_share_p90']:.3f}), "
          f"median k50 {ctx['k50_median']:.0f}, adaptive share of kernel mass {ctx['adaptive_share']:.3f}")
    print("by prefix:", ", ".join(f"{b['label']}: {b['long']:.3f}" for b in ctx["by_prefix"]))
    print("copy:", ", ".join(f"{c['label']} (n={c['n']}): {c['long']:.3f}" for c in ctx["copy_rows"]))
    print("shortest n-1:", ", ".join(f"{r['token']!r} {r['long']:.3f}" for r in ctx["shortest"][:8]))
    print("longest n-1:", ", ".join(f"{r['token']!r} {r['long']:.3f}" for r in ctx["longest"][:8]))
    print("alpha CV per basis function:", ", ".join(f"{c['center']}: {c['cv']:.2f}" for c in ctx["cv"]))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
