"""Ad hoc: is the visual periodicity in `dense1` units (see
`modarith_multiplication_dense1_top_units_line.py`) real structure, or an
eyeballing illusion the way strong-looking "patterns" can appear in filtered
Gaussian noise?

Three checks, each comparing the REAL unit against matched-noise controls:

  1. **2D FFT on the genuine (97, 97) (a, b) grid** (not the flattened 1D
     line -- flattening itself can manufacture a spurious period-97 artifact
     from the a-block boundaries baked into the x-axis). Reports the peak
     frequency pair and what fraction of total spectral energy it holds.

  2. **Permutation null.** Shuffles which (a, b) label is attached to each
     already-computed activation value (activations themselves untouched,
     only their assignment to grid cells is randomized) 200 times, recomputes
     the same "peak spectral energy fraction" statistic each time, and
     reports the real unit's percentile against that null distribution. A
     real periodic dependency on (a, b) must vanish under this shuffle
     (nothing left to be periodic in); noise that only looks periodic by eye
     will already score similarly to its own shuffles.

  3. **Matched-variance Gaussian-noise control.** Generates random noise with
     the same variance as the real unit, runs it through the identical 2D
     FFT + peak-energy-fraction pipeline, so the real unit's score has a
     concrete, non-eyeballed noise baseline to beat.

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_pattern_significance.py
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_pattern_significance.py --ranks 0 2 5
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
NUM_SHUFFLES = 200
SEED = 0


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


def peak_energy_fraction(grid_2d: np.ndarray) -> tuple:
    """2D FFT of a (97, 97) real-valued grid. Returns (peak (ka, kb) excluding
    the DC term, fraction of total non-DC spectral energy at that peak)."""
    spec = np.abs(np.fft.fft2(grid_2d - grid_2d.mean()))
    power = spec ** 2
    power[0, 0] = 0.0  # exclude DC
    total = power.sum()
    if total == 0:
        return (0, 0), 0.0
    peak_flat = np.argmax(power)
    ka, kb = np.unravel_index(peak_flat, power.shape)
    return (int(ka), int(kb)), float(power[ka, kb] / total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 2, 5])
    args = parser.parse_args()

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1 = dense1_activations(model, token_ids)  # (9409, 512), a outer / b inner

    order = np.argsort(-h1.var(axis=0))
    rng = np.random.default_rng(SEED)

    print(f"{RUN_NAME} @ {CHECKPOINT} -- 2D-FFT periodicity significance "
          f"({NUM_SHUFFLES} label-shuffles + matched-noise control per unit)\n")

    for rank in args.ranks:
        unit = int(order[rank])
        y = h1[:, unit]
        grid = y.reshape(MODULUS, MODULUS)  # (a, b), matches build order

        real_peak, real_frac = peak_energy_fraction(grid)

        # Permutation null: shuffle which (a,b) cell each activation value
        # lands in, recompute the same statistic. A real (a,b)-dependent
        # signal cannot survive this; positional/index artifacts might.
        null_fracs = np.empty(NUM_SHUFFLES)
        flat = y.copy()
        for i in range(NUM_SHUFFLES):
            shuffled = rng.permutation(flat).reshape(MODULUS, MODULUS)
            _, null_fracs[i] = peak_energy_fraction(shuffled)
        percentile = float((null_fracs < real_frac).mean() * 100)

        # Matched-variance Gaussian noise control, same pipeline.
        noise = rng.normal(loc=y.mean(), scale=y.std(), size=y.shape)
        noise_peak, noise_frac = peak_energy_fraction(noise.reshape(MODULUS, MODULUS))

        print(f"y{rank} = dense1 unit #{unit}")
        print(f"  real:   peak (ka,kb)={real_peak}  energy_fraction={real_frac:.4f}")
        print(f"  null (shuffled labels): mean={null_fracs.mean():.4f}  "
              f"max={null_fracs.max():.4f}  real is at percentile {percentile:.1f} of the null distribution")
        print(f"  matched-variance Gaussian noise: peak={noise_peak}  energy_fraction={noise_frac:.4f}")
        verdict = "REAL STRUCTURE" if percentile >= 99.0 and real_frac > noise_frac * 3 else "INCONCLUSIVE / LIKELY NOISE-LIKE"
        print(f"  verdict: {verdict}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
