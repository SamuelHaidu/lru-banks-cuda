"""Ad hoc: phase correlation across `dense1`'s 512 channels, to properly test
the "shared cyclic pattern with per-channel phase shift" picture that
`modarith_multiplication_dense1_cross_correlation.py` left open -- ordinary
Pearson/Spearman correlation at zero lag scores ~0 for two signals sharing a
frequency but differing in phase (e.g. cos vs. sin of the same angle), which
is exactly the kind of relationship a Fourier-style basis would produce.

Setup: same as the cross-correlation script -- fix `b`, sweep `a` over
`[0, 97)`, collect the `(97, 512)` dense1 activations.

Method (proper CIRCULAR statistics, not linear correlation of raw angles --
angles wrap around, so plain Pearson on phase values is invalid at the
wraparound boundary):
  1. For each channel, take the FFT of its length-97 signal over `a` and
     extract the complex coefficient at a chosen frequency `k` (default: the
     single frequency with the most total power summed across all 512
     channels -- i.e. the population's shared dominant frequency, found
     empirically here rather than assumed).
  2. That complex coefficient's angle is the channel's PHASE at that
     frequency. Two channels sharing the same oscillation but shifted (e.g.
     cos vs. sin) will have a highly consistent phase DIFFERENCE across
     repeated measurements -- but with only one measurement per channel
     here (one fixed `b`), "phase correlation between channel j and k" is
     simply their phase difference `phase_j - phase_k`, and a population of
     512 CONSISTENT (non-uniformly-scattered) phase differences relative to
     e.g. the population mean phase is itself the signature of a shared
     oscillator.
  3. Reports: (a) a polar scatter of all 512 phases at the dominant
     frequency -- clustering means a shared oscillator, uniform scatter
     means independent phases; (b) the population Phase-Locking Value (PLV,
     also called the mean resultant length) `|mean(exp(i*phase))|`, which is
     exactly "angular correlation" done correctly for circular data (1.0 =
     perfectly phase-aligned, 0.0 = uniformly scattered, i.e. the circular
     analogue of Pearson r for a single population rather than a pairwise
     matrix); (c) the full 512x512 pairwise phase-DIFFERENCE matrix, and its
     own PLV-style consistency measure (Rayleigh-test-style circular
     concentration) to see if specific channel pairs lock at a fixed
     relative phase (e.g. exactly 90 degrees apart, i.e. cos/sin pairs).

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_dense1_phase_correlation.py"
    .venv/bin/python "ad hoc scripts/modarith_multiplication_dense1_phase_correlation.py" --b 48 --frequency 10
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


def dominant_shared_frequency(x: np.ndarray) -> int:
    """`x`: (samples, channels). Returns the FFT frequency bin (excluding DC)
    with the most POOLED power summed across all channels -- the
    population's shared dominant frequency."""
    spectrum = np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0)
    power = (np.abs(spectrum) ** 2).sum(axis=1)
    power[0] = 0.0
    return int(np.argmax(power))


def circular_mean_resultant_length(angles: np.ndarray) -> float:
    """Phase-Locking Value / mean resultant length: 1.0 = all angles
    identical, 0.0 = uniformly scattered around the circle. The circular
    analogue of "how correlated/concentrated is this population", done
    correctly for wraparound data."""
    return float(np.abs(np.mean(np.exp(1j * angles))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=48, help="Fixed `b` operand (default 48).")
    parser.add_argument("--frequency", type=int, default=None,
                         help="FFT frequency bin to extract phase at. Default: the population's shared dominant frequency.")
    args = parser.parse_args()

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes_fixed_b(args.b, window)
    h1 = dense1_activations(model, token_ids)  # (97, 512): one row per `a`, `b` fixed

    k = args.frequency if args.frequency is not None else dominant_shared_frequency(h1)
    print(f"{RUN_NAME} @ {CHECKPOINT} -- dense1 phase correlation, b={args.b} fixed, a=0..{MODULUS-1} swept, "
          f"frequency bin k={k} (period ~{MODULUS / k:.1f} in `a`)\n")

    spectrum = np.fft.rfft(h1 - h1.mean(axis=0, keepdims=True), axis=0)  # (freqs, 512)
    coeff = spectrum[k]  # (512,) complex
    phases = np.angle(coeff)  # (512,) in [-pi, pi]
    magnitudes = np.abs(coeff)  # how strongly each channel actually carries this frequency

    plv_all = circular_mean_resultant_length(phases)
    # Magnitude-weighted PLV: channels that barely carry this frequency
    # shouldn't count as much toward "is there a shared phase" as channels
    # that carry it strongly.
    weights = magnitudes / magnitudes.sum()
    plv_weighted = float(np.abs(np.sum(weights * np.exp(1j * phases))))

    print(f"Population Phase-Locking Value (unweighted): {plv_all:.4f}")
    print(f"Population Phase-Locking Value (magnitude-weighted): {plv_weighted:.4f}")
    print("(1.0 = every channel phase-aligned at this frequency; 0.0 = phases uniformly scattered)\n")

    # Pairwise phase-difference matrix + its own concentration.
    diff_matrix = phases[:, None] - phases[None, :]  # (512, 512), wrapped implicitly by angle math below
    diff_matrix = np.angle(np.exp(1j * diff_matrix))  # wrap to [-pi, pi]

    # Cluster channels by phase to check for discrete phase groups (e.g. two
    # clusters ~90 degrees apart would suggest cos/sin-style pairing).
    sort_order = np.argsort(phases)

    print(f"Phase range across channels: [{phases.min():.3f}, {phases.max():.3f}] radians "
          f"([{np.degrees(phases.min()):.1f}, {np.degrees(phases.max()):.1f}] degrees)")
    print(f"Magnitude-weighted circular std of phases: "
          f"{np.sqrt(-2 * np.log(plv_weighted)) if plv_weighted > 0 else float('nan'):.4f} radians\n")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        subplot_titles=(
            f"512 channel phases at k={k} (marker size ~ magnitude), PLV={plv_weighted:.3f}",
            "512x512 pairwise phase difference (radians, sorted by phase)",
        ),
    )
    fig.add_trace(
        go.Scatterpolar(
            r=magnitudes, theta=np.degrees(phases), mode="markers",
            marker=dict(size=5, color=magnitudes, colorscale="Viridis", showscale=True, colorbar=dict(title="|FFT coeff|", x=0.42)),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=diff_matrix[sort_order][:, sort_order], colorscale="Twilight", zmid=0, zmin=-np.pi, zmax=np.pi,
                   colorbar=dict(title="phase diff (rad)")),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="channel (sorted by phase)", row=1, col=2)
    fig.update_yaxes(title_text="channel (sorted by phase)", row=1, col=2)
    fig.update_layout(
        title=f"{RUN_NAME}: dense1 phase correlation at shared frequency k={k} (b={args.b} fixed)",
        height=700, width=1500,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multiplication_dense1_phase_correlation_b{args.b}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
