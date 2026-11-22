"""Ad hoc: pairwise (Pearson + Spearman) cross-correlation over `dense1`'s
512 channels, for a single fixed operation `a * b = ...` with `b` fixed and
`a` swept over its full range.

Cosine similarity between two channels' activation vectors tells you whether
they point in a similar direction (same relative pattern of highs/lows,
same relative magnitudes) -- it does NOT tell you whether two channels move
together / are correlated AS A FUNCTION OF `a` (e.g. two channels could be
perfectly linearly correlated -- always rising and falling together -- while
still having low cosine similarity if their raw magnitudes/offsets differ).
Pearson/Spearman correlation across the swept axis is the right tool for
"do these two channels co-vary with `a`", independent of scale/offset
(Pearson: linear co-variation; Spearman: monotonic co-variation, robust to
nonlinearity such as saturating GELU curvature).

Setup: fix `b`, build `<bos> a1 a0 * b1 b0 =` for every `a` in `[0, 97)`,
collect `dense1` activations -> a `(97, 512)` matrix (97 samples, one per
`a`; 512 channels). Computes the full `512x512` Pearson and Spearman
correlation matrices across those 97 samples and plots both as heatmaps.

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_dense1_cross_correlation.py
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_dense1_cross_correlation.py --b 48
"""

import argparse
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

RUN_NAME = "modarith-baseline-v2"
CHECKPOINT = 24  # epoch 24 -- run diverged to NaN starting epoch 25, see modular-arithmetic-findings.md
MODULUS = 97
BOS, STAR, EQUALS = 10, 14, 15


def digits(n: int) -> tuple:
    return (n // 10, n % 10)


def build_prefixes_fixed_b(b: int, window: int) -> np.ndarray:
    """`<bos> a1 a0 * b1 b0 =` for every `a` in `[0, MODULUS)`, `b` fixed."""
    b0, b1 = digits(b)
    prefixes = []
    for a in range(MODULUS):
        a0, a1 = digits(a)
        prefixes.append((BOS, a0, a1, STAR, b0, b1, EQUALS))
    prefixes = np.array(prefixes, dtype=np.int32)
    pad = np.zeros((prefixes.shape[0], window - 7), dtype=np.int32)
    return np.concatenate([pad, prefixes], axis=1)


def dense1_activations(model, token_ids: np.ndarray) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        ids = torch.as_tensor(token_ids, dtype=torch.int32).to(device)
        norm_in = model.normalized_input_embeddings()
        flat = norm_in[ids.reshape(-1)]
        u = flat.reshape(ids.shape[0], model.window_size * model.dim_in)
        u = torch.nn.functional.normalize(u, dim=-1)
        h1 = model.dense1(u)
    return h1.detach().cpu().numpy()


def pearson_matrix(x: np.ndarray) -> np.ndarray:
    """`x`: (samples, channels). Returns the (channels, channels) Pearson
    correlation matrix."""
    return np.corrcoef(x, rowvar=False)


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Spearman = Pearson correlation of the rank-transformed data."""
    from scipy.stats import rankdata

    ranks = np.apply_along_axis(rankdata, 0, x)
    return np.corrcoef(ranks, rowvar=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=48, help="Fixed `b` operand (default 48).")
    args = parser.parse_args()

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes_fixed_b(args.b, window)
    h1 = dense1_activations(model, token_ids)  # (97, 512): one row per `a`, `b` fixed

    print(f"{RUN_NAME} @ {CHECKPOINT} -- dense1 512x512 cross-correlation, b={args.b} fixed, a=0..{MODULUS-1} swept\n")

    pearson = pearson_matrix(h1)
    spearman = spearman_matrix(h1)

    # Off-diagonal summary stats -- the diagonal is trivially 1.0 for both.
    n = pearson.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    pear_off = pearson[off_diag_mask]
    spear_off = spearman[off_diag_mask]

    print(f"Pearson  off-diagonal: mean={pear_off.mean():.4f}  |corr|>0.9 fraction={np.mean(np.abs(pear_off) > 0.9):.4f}  "
          f"|corr|>0.99 fraction={np.mean(np.abs(pear_off) > 0.99):.4f}")
    print(f"Spearman off-diagonal: mean={spear_off.mean():.4f}  |corr|>0.9 fraction={np.mean(np.abs(spear_off) > 0.9):.4f}  "
          f"|corr|>0.99 fraction={np.mean(np.abs(spear_off) > 0.99):.4f}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pearson correlation (512x512)", "Spearman correlation (512x512)"))
    fig.add_trace(go.Heatmap(z=pearson, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="Pearson r", x=0.46)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=spearman, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="Spearman rho")), row=1, col=2)
    fig.update_xaxes(title_text="dense1 channel", row=1, col=1)
    fig.update_yaxes(title_text="dense1 channel", row=1, col=1)
    fig.update_xaxes(title_text="dense1 channel", row=1, col=2)
    fig.update_yaxes(title_text="dense1 channel", row=1, col=2)
    fig.update_layout(
        title=f"{RUN_NAME}: dense1 512x512 cross-correlation across a=0..96 (b={args.b} fixed)",
        height=800, width=1600,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multiplication_dense1_cross_correlation_b{args.b}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
