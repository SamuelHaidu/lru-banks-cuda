"""Ad hoc: extract and plot the actual WAVEFORM of the shared MSSA mode(s)
found in `modarith_multiplication_mssa.py` -- that script only reported how
much variance each mode explains; this one reconstructs what the mode
actually looks like as a function of `b`, via standard SSA/MSSA
"reconstructed components" (diagonal-averaging the rank-1 (or rank-k)
reprojection of the trajectory matrix back into a length-97 time series).

For a fixed `a`:
  1. Build the same stacked multichannel trajectory matrix as
     `modarith_multiplication_mssa.py` (all 512 dense1 channels, window
     length L).
  2. SVD it, keeping singular VECTORS this time (not just values).
  3. For each of the top-k modes, reconstruct its per-channel contribution:
     `U[:, i] @ diag(sv[i]) @ Vt[i, :]` gives a rank-1 matrix the same shape
     as the trajectory; diagonal-averaging any one channel's L-column block
     of that rank-1 matrix recovers a length-97 "reconstructed component"
     (RC) -- the classic SSA reconstruction step. This is the actual
     WAVEFORM the shared mode traces out in `b`, not just an energy number.
  4. Plot RC1 (and RC2, RC3 if requested) overlaid on the raw signal of a
     couple of real channels, so the extracted wave can be checked visually
     against what the real units are doing.

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_mssa_waveforms.py"
    .venv/bin/python "ad hoc scripts/modarith_multiplication_mssa_waveforms.py" --a-values 0 48 96 --num-modes 3
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
    """Inverts a (series_len - L + 1, L) Hankel-shaped matrix back into a
    length-`series_len` series by averaging each anti-diagonal -- the
    standard SSA reconstruction step. `matrix` need not be exactly Hankel
    (a rank-1 reprojection isn't), hence the averaging."""
    rows, L = matrix.shape
    out = np.zeros(series_len)
    counts = np.zeros(series_len)
    for i in range(rows):
        out[i : i + L] += matrix[i]
        counts[i : i + L] += 1
    return out / counts


def reconstructed_components(row_activations: np.ndarray, L: int, num_modes: int):
    """Returns `(singular_values, rc_per_channel, channel_means)` where
    `rc_per_channel` has shape `(num_modes, 512, 97)`: for each of the top
    `num_modes` MSSA modes, the reconstructed length-97 waveform of every one
    of the 512 channels.

    Each channel is MEAN-CENTERED before building its Hankel block (standard
    SSA practice -- you decompose the fluctuation around the mean, not the
    raw signal with each channel's own offset baked in). This matters a lot
    here: without centering, RC1 mostly captures the shared DC/offset level
    across channels rather than the shared oscillation, and different
    channels' oscillations are independently phase-shifted, so averaging
    raw (off-centered) per-channel RCs across all 512 channels for display
    washes the signal down to near-zero amplitude even though every
    individual channel's own RC is a strong, real wave -- exactly what
    produced the earlier flat-looking, "I don't see anything" plot.
    `channel_means` (length 512) lets a caller add each channel's own mean
    back for a real like-for-like overlay against its raw (uncentered)
    signal, or leave it off to compare pure fluctuations.

    Uses `scipy.linalg.svd(..., lapack_driver="gesvd")` rather than
    `np.linalg.svd`/`scipy.sparse.linalg.svds` -- a genuine Hankel
    trajectory matrix has heavily correlated (overlapping-window) rows,
    which is numerically pathological for the default divide-and-conquer
    driver: both `np.linalg.svd` (`LinAlgError: SVD did not converge`) and
    `scipy.sparse.linalg.svds`/ARPACK (`Could not build an Arnoldi
    factorization`) fail on this exact trajectory shape, while the slower,
    classical QR-based `gesvd` driver handles it without issue -- unlike the
    energy-only sweep in `modarith_multiplication_mssa.py` (97 rows, needs
    speed), this script only processes a handful of `a` rows, so `gesvd`'s
    extra cost is fine.
    """
    from scipy.linalg import svd as scipy_svd

    series_len, num_channels = row_activations.shape
    channel_means = row_activations.mean(axis=0)  # (num_channels,)
    centered = row_activations - channel_means
    blocks = [hankel_trajectory(centered[:, c], L) for c in range(num_channels)]
    trajectory = np.concatenate(blocks, axis=1)  # (series_len-L+1, L*num_channels)

    u, sv, vt = scipy_svd(trajectory, full_matrices=False, lapack_driver="gesvd")
    u, sv, vt = u[:, :num_modes], sv[:num_modes], vt[:num_modes]

    rc = np.zeros((num_modes, num_channels, series_len))
    for m in range(num_modes):
        rank1 = sv[m] * np.outer(u[:, m], vt[m])  # (series_len-L+1, L*num_channels)
        for c in range(num_channels):
            block = rank1[:, c * L : (c + 1) * L]
            rc[m, c] = diagonal_average(block, series_len)
    return sv, rc, channel_means


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-length", type=int, default=48)
    parser.add_argument("--num-modes", type=int, default=3, help="How many leading MSSA modes to reconstruct and plot.")
    parser.add_argument("--a-values", type=int, nargs="+", default=[0, 48, 96],
                         help="Which `a` rows to reconstruct waveforms for (default: 0, 48, 96).")
    parser.add_argument("--example-channels", type=int, nargs="+", default=None,
                         help="Which channel indices to overlay as raw examples (default: top-3 variance channels).")
    args = parser.parse_args()

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1 = dense1_activations(model, token_ids)  # (9409, 512), a outer / b inner
    grid = h1.reshape(MODULUS, MODULUS, h1.shape[1])  # (a, b, channel)

    example_channels = args.example_channels
    if example_channels is None:
        example_channels = np.argsort(-h1.var(axis=0))[:3].tolist()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # One row of subplots per `a` value, one column per example channel: each
    # panel overlays that channel's RAW (uncentered) signal against its own
    # RC1+RC2+...+RCk reconstruction (mode contributions summed, channel mean
    # added back) -- a real like-for-like comparison, unlike averaging RCs
    # across channels with different offsets/phases (see docstring above).
    n_rows = len(args.a_values)
    n_cols = len(example_channels)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"a={a}, channel #{c}" for a in args.a_values for c in example_channels],
    )

    for i, a in enumerate(args.a_values, start=1):
        sv, rc, channel_means = reconstructed_components(grid[a], args.window_length, args.num_modes)
        b_axis = list(range(MODULUS))

        for j, c in enumerate(example_channels, start=1):
            raw = grid[a, :, c]
            reconstruction = rc[:, c, :].sum(axis=0) + channel_means[c]  # sum of top-k modes + mean back

            fig.add_trace(
                go.Scatter(x=b_axis, y=raw, mode="lines", name=f"a={a} c#{c} raw",
                           line=dict(width=1), opacity=0.6, showlegend=(i == 1 and j == 1),
                           legendgroup="raw"),
                row=i, col=j,
            )
            fig.add_trace(
                go.Scatter(x=b_axis, y=reconstruction, mode="lines",
                           name=f"a={a} c#{c} RC1..{args.num_modes} (sv1={sv[0]:.1f})",
                           line=dict(width=2, dash="dash"), showlegend=(i == 1 and j == 1),
                           legendgroup="reconstruction"),
                row=i, col=j,
            )
            fig.update_yaxes(title_text="activation", row=i, col=j)
            fig.update_xaxes(title_text="b", row=i, col=j)

    fig.update_layout(
        title=f"{RUN_NAME}: raw dense1 signal vs. top-{args.num_modes} MSSA-mode reconstruction, per channel",
        height=350 * n_rows, width=450 * n_cols,
        legend=dict(title="raw vs. reconstruction (shared legend across all panels)"),
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_mssa_waveforms.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
