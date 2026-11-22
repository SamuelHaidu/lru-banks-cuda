"""Ad hoc: full MSSA decomposition of `y0` -- the single highest-variance
`dense1` unit (rank 0), the same one first plotted as a raw line in
`modarith_multiplication_dense1_unit0_line.py` -- across the WHOLE
flattened multiplication grid (same x-axis as that original plot:
`a*97 + b`, `a` outer loop 0..96, `b` inner loop 0..96).

"Channel" = one of the 512 neurons in `dense1` (the first hidden layer,
80->512, GELU); `y0` is channel/unit #{whichever index has the highest
variance}, picked as "the" example unit throughout this investigation.
`a` is the first operand of the equation `a * b mod 97`; for each fixed `a`
the model was run on every `b` in 0..96, so one full sweep covers every
possible multiplication.

Decomposition method: MSSA needs a per-`a` trajectory matrix (see
`modarith_multiplication_mssa.py`), so this script runs that per-`a`
decomposition for every `a` in 0..96, keeps ONLY unit y0's reconstructed
components (RC1, RC2, RC3) each time, and concatenates the 97 per-`a`
segments back into one length-9409 series -- reproducing y0's original
flattened line, now split into:

  - RC1: the single leading MSSA mode (the "shared driver" already shown to
    dominate cross-channel variance in `modarith_multiplication_mssa.py`)
  - RC2, RC3: the next two modes
  - residual: raw signal minus (RC1+RC2+RC3) -- whatever those 3 modes don't
    explain

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_y0_decomposition.py
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_y0_decomposition.py --num-modes 5
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


def build_prefixes(window: int) -> np.ndarray:
    prefixes = []
    for a in range(MODULUS):
        a0, a1 = digits(a)
        for b in range(MODULUS):
            b0, b1 = digits(b)
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


def hankel_trajectory(series: np.ndarray, L: int) -> np.ndarray:
    n = len(series) - L + 1
    return np.lib.stride_tricks.sliding_window_view(series, L)[:n]


def diagonal_average(matrix: np.ndarray, series_len: int) -> np.ndarray:
    rows, L = matrix.shape
    out = np.zeros(series_len)
    counts = np.zeros(series_len)
    for i in range(rows):
        out[i : i + L] += matrix[i]
        counts[i : i + L] += 1
    return out / counts


def y0_components_for_row(row_activations: np.ndarray, target_channel: int, L: int, num_modes: int) -> np.ndarray:
    """Returns `(num_modes, 97)`: unit `target_channel`'s reconstructed
    component waveforms for one fixed `a`, mean-centered (channel means are
    NOT added back here -- the caller adds y0's own single mean back once,
    after concatenating all 97 `a` segments, so segment boundaries don't
    show a spurious per-`a` DC jump)."""
    from scipy.linalg import svd as scipy_svd

    series_len, num_channels = row_activations.shape
    channel_means = row_activations.mean(axis=0)
    centered = row_activations - channel_means
    blocks = [hankel_trajectory(centered[:, c], L) for c in range(num_channels)]
    trajectory = np.concatenate(blocks, axis=1)

    u, sv, vt = scipy_svd(trajectory, full_matrices=False, lapack_driver="gesvd")
    u, sv, vt = u[:, :num_modes], sv[:num_modes], vt[:num_modes]

    rc = np.zeros((num_modes, series_len))
    col_start = target_channel * L
    for m in range(num_modes):
        rank1_block = sv[m] * np.outer(u[:, m], vt[m, col_start : col_start + L])
        rc[m] = diagonal_average(rank1_block, series_len)
    return rc, channel_means[target_channel]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-length", type=int, default=48)
    parser.add_argument("--num-modes", type=int, default=3, help="How many MSSA modes to extract per `a` row.")
    args = parser.parse_args()
    L = args.window_length

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1 = dense1_activations(model, token_ids)  # (9409, 512), a outer / b inner
    grid = h1.reshape(MODULUS, MODULUS, h1.shape[1])  # (a, b, channel)

    order = np.argsort(-h1.var(axis=0))
    y0_channel = int(order[0])
    print(f"y0 = dense1 unit #{y0_channel} (highest variance)\n"
          f"Decomposing across all {MODULUS} values of `a`, window length L={L}, top {args.num_modes} modes...\n")

    raw_full = grid[:, :, y0_channel].reshape(-1)  # (9409,), matches original y0 flattened line
    rc_full = np.zeros((args.num_modes, MODULUS * MODULUS))
    means = np.zeros(MODULUS)

    for a in range(MODULUS):
        rc, mean = y0_components_for_row(grid[a], y0_channel, L, args.num_modes)
        rc_full[:, a * MODULUS : (a + 1) * MODULUS] = rc
        means[a] = mean
        if a % 20 == 0:
            print(f"  a={a}/{MODULUS} done")

    # Add y0's OWN global mean back once (not each row's local mean -- rows'
    # local means differ slightly by `a`, and re-adding them per-segment
    # would reintroduce the exact per-`a` DC jump the decomposition is
    # trying to separate out from the oscillatory modes).
    global_mean = raw_full.mean()
    reconstruction = rc_full.sum(axis=0) + global_mean
    residual = raw_full - reconstruction

    corr = np.corrcoef(raw_full, reconstruction)[0, 1]
    print(f"\nraw vs. (RC1..RC{args.num_modes} + mean) correlation across the full grid: {corr:.4f}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = list(range(MODULUS * MODULUS))
    tickvals = list(range(0, MODULUS * MODULUS, MODULUS * 10))
    ticktext = [str(a) for a in range(0, MODULUS, 10)]

    n_panels = 2 + args.num_modes  # raw+reconstruction, residual, then each RC alone
    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True,
        subplot_titles=(
            ["y0 raw signal vs. RC1.." + str(args.num_modes) + " + mean (reconstruction)", "residual (raw - reconstruction)"]
            + [f"RC{m+1} alone (mode {m+1})" for m in range(args.num_modes)]
        ),
    )

    fig.add_trace(go.Scatter(x=x, y=raw_full, mode="lines", name="raw y0", line=dict(width=1), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=reconstruction, mode="lines", name="reconstruction", line=dict(width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=residual, mode="lines", name="residual", line=dict(width=1, color="firebrick")), row=2, col=1)
    for m in range(args.num_modes):
        fig.add_trace(go.Scatter(x=x, y=rc_full[m], mode="lines", name=f"RC{m+1}", line=dict(width=1)), row=3 + m, col=1)

    for r in range(1, n_panels + 1):
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=r, col=1)
        fig.update_yaxes(title_text="activation", row=r, col=1)
    fig.update_xaxes(title_text="a (tick label); b sweeps 0..96 within each a-block", row=n_panels, col=1)

    fig.update_layout(
        title=f"{RUN_NAME}: y0 (dense1 unit #{y0_channel}) MSSA decomposition across the full a*b grid "
              f"(raw-vs-reconstruction correlation={corr:.3f})",
        height=280 * n_panels, width=1400,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_y0_decomposition.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
