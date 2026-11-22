"""Ad hoc: Multichannel Singular Spectrum Analysis (MSSA) over `dense1`'s 512
units, to check whether the per-channel periodicity in `b` already confirmed
statistically (`modarith_multiplication_pattern_significance.py`) reflects a
small number of SHARED oscillatory modes across many channels, or 512
independently-phased/frequenced coincidences.

Scope: per fixed `a`, each of the 512 dense1 units gives a length-97 series
in `b`. This is Option A from the discussion with the user -- not full 2D
(a,b) MSSA, which would need a much bigger trajectory matrix for little
extra signal given the 2D-FFT test already showed the periodicity lives in
`b` alone (`ka=0` for every unit checked).

Method (standard MSSA): for a fixed `a`,
  1. For each of the 512 channels, build a length-(97-L+1) x L Hankel
     (trajectory) sub-matrix via sliding windows of length `L` over the
     channel's 97-point series in `b`.
  2. Horizontally stack all 512 channels' sub-matrices into one trajectory
     matrix of shape (97-L+1, L*512).
  3. SVD it. The top singular vectors/values reveal modes shared ACROSS
     channels: if many channels share a periodic driver in `b`, a small
     number of singular values will dominate (fast-decaying spectrum), and
     the leading right-singular vectors, reshaped to (L, 512) and inspected
     per-channel, will show which channels load onto that shared mode.
  4. Repeat for several `a` values and report how consistent the leading
     singular value's dominance and its dominant per-window frequency are.

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_mssa.py"
    .venv/bin/python "ad hoc scripts/modarith_multiplication_mssa.py" --window-length 48 --a-values 0 25 50 75
"""

import argparse

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
    """`(N,)` series -> `(N-L+1, L)` trajectory (Hankel) matrix."""
    n = len(series) - L + 1
    return np.lib.stride_tricks.sliding_window_view(series, L)[:n]


def mssa_for_row(row_activations: np.ndarray, L: int, k: int = 8) -> tuple:
    """`row_activations`: (97, 512) -- one `a`'s worth of dense1 activations,
    indexed by `b`. Returns `(top_k_singular_values, total_energy)`, where
    `total_energy` is the trajectory matrix's squared Frobenius norm (i.e.
    the sum of ALL singular values squared, cheap to get directly without a
    full SVD -- `sum(M**2)` equals `sum(singular_values**2)`).

    Uses `scipy.sparse.linalg.svds` (top-k only, via ARPACK) rather than a
    full `np.linalg.svd` -- the full SVD computes and discards ~40 unused
    singular values/vectors per call, and doing that for all 97 `a` rows
    (as opposed to the original handful) is the difference between ~2
    minutes and ~15 seconds.
    """
    from scipy.sparse.linalg import svds

    num_channels = row_activations.shape[1]
    blocks = [hankel_trajectory(row_activations[:, c], L) for c in range(num_channels)]
    trajectory = np.concatenate(blocks, axis=1)  # (97-L+1, L*num_channels)
    total_energy = float((trajectory ** 2).sum())
    # svds returns singular values ascending; flip to descending like np.linalg.svd.
    singular_values = svds(trajectory, k=k, return_singular_vectors=False)
    return np.sort(singular_values)[::-1], total_energy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-length", type=int, default=48, help="MSSA embedding window L (default 48, ~half of 97).")
    parser.add_argument("--a-values", type=int, nargs="+", default=None,
                         help="Which `a` rows to run MSSA on. Default: every a in 0..96 (light per-row via top-k svds).")
    args = parser.parse_args()
    L = args.window_length
    a_values = args.a_values if args.a_values is not None else list(range(MODULUS))

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1 = dense1_activations(model, token_ids)  # (9409, 512), a outer / b inner
    grid = h1.reshape(MODULUS, MODULUS, h1.shape[1])  # (a, b, channel)

    print(f"{RUN_NAME} @ {CHECKPOINT} -- per-`a` MSSA over all 512 dense1 channels "
          f"(window length L={L}, series length 97 in `b`, {len(a_values)} rows)\n")

    top1_fracs, top5_fracs = [], []
    for a in a_values:
        sv, total_energy = mssa_for_row(grid[a], L)
        top1_frac = (sv[0] ** 2) / total_energy if total_energy > 0 else float("nan")
        top5_frac = (sv[:5] ** 2).sum() / total_energy if total_energy > 0 else float("nan")
        top1_fracs.append(top1_frac)
        top5_fracs.append(top5_frac)
        print(f"a={a:3d}: top singular value fraction={top1_frac:.4f}  top-5 fraction={top5_frac:.4f}  "
              f"first 8 singular values={np.round(sv, 3).tolist()}")

    print(f"\nAcross all {len(a_values)} rows: top-1 fraction mean={np.mean(top1_fracs):.4f} "
          f"(min={np.min(top1_fracs):.4f}, max={np.max(top1_fracs):.4f}); "
          f"top-5 fraction mean={np.mean(top5_fracs):.4f} "
          f"(min={np.min(top5_fracs):.4f}, max={np.max(top5_fracs):.4f})")
    print("\nInterpretation: a fast-decaying spectrum (top-1 or top-5 fraction close to 1,")
    print("far above 1/num_channels or 1/rank baselines) means a SMALL NUMBER of shared")
    print("oscillatory modes explain most cross-channel variation in `b` -- consistent with")
    print("a genuine shared periodic mechanism rather than 512 independent coincidences.")
    print("A flat/slowly-decaying spectrum would argue against a shared mode.")

    if len(a_values) > 1:
        import plotly.graph_objects as go
        from pathlib import Path

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a_values, y=top1_fracs, mode="lines+markers", name="top-1 singular value fraction"))
        fig.add_trace(go.Scatter(x=a_values, y=top5_fracs, mode="lines+markers", name="top-5 singular value fraction"))
        fig.update_layout(
            title=f"{RUN_NAME}: MSSA shared-mode strength across dense1's 512 channels, per `a` row",
            xaxis_title="a",
            yaxis_title="fraction of total spectral energy",
            height=500, width=1200,
        )
        out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "multiplication_mssa_per_row.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
