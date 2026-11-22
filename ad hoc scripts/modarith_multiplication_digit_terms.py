"""Ad hoc: does `modarith-baseline-v2` solve `(a * b) mod 97` via digit-level
partial products, rather than the Fourier-circle (addition) or discrete-log
mechanisms already ruled out by `modarith_multiplication_mechanism.py`?

`a = 10*a1 + a0`, `b = 10*b1 + b0` (a0/a1/b0/b1 the four digit tokens the
network actually reads), so:

    a*b = 100*(a1*b1) + 10*(a1*b0 + a0*b1) + a0*b0

every term of which is available to a bilinear function of the four digit
embeddings without any modular-inverse or logarithm trick -- exactly the kind
of thing a 3-layer FFN over concatenated embeddings can build directly
(cross terms come for free from products of hidden units, which GELU-gated
dense layers can approximate). This probe checks whether the query / hidden
units collapse onto any of:

    a0*b0 mod 97, a1*b0 mod 97, a0*b1 mod 97, a1*b1 mod 97, (any of the four
    digits alone), and the two "combined" terms a1*b0 + a0*b1 (mod 97) and
    the full weighted sum 100*a1*b1 + 10*(a1*b0+a0*b1) + a0*b0 (mod 97, i.e.
    a*b mod 97 itself, included again here for direct comparison under the
    same collapse-ratio code as the other candidate indices).

A single dominant term with a low collapse ratio would mean the network
picked out that one partial product as its primary feature; a case where NO
single term collapses well suggests the true representation is a genuine
multi-term interaction that no single 1-D index captures (in which case a
linear probe cannot fully diagnose it, and behavior-only ablations, e.g.
zeroing one digit's embedding contribution, are the next step).

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_digit_terms.py"
"""

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


def forward_with_activations(model, token_ids: np.ndarray):
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        ids = torch.as_tensor(token_ids, dtype=torch.int32).to(device)
        norm_in = model.normalized_input_embeddings()
        flat = norm_in[ids.reshape(-1)]
        u = flat.reshape(ids.shape[0], model.window_size * model.dim_in)
        u = torch.nn.functional.normalize(u, dim=-1)
        h1 = model.dense1(u)
        h2 = model.dense2(h1)
        z = model.dense3(h2)
        q = torch.nn.functional.normalize(z, dim=-1)
    return (t.detach().cpu().numpy() for t in (h1, h2, q))


def dominant_frequency(signal: np.ndarray) -> tuple:
    spec = np.abs(np.fft.rfft(signal - signal.mean()))
    if spec[1:].sum() == 0:
        return 0, 0.0
    k = 1 + int(np.argmax(spec[1:]))
    frac = spec[k] ** 2 / (spec[1:] ** 2).sum()
    return k, float(frac)


def collapse_ratio(activations: np.ndarray, index: np.ndarray, num_bins: int) -> float:
    within = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    for r in range(num_bins):
        mask = index == r
        if mask.any():
            within[r] = activations[mask].std(axis=0).mean()
            counts[r] = mask.sum()
    overall = activations.std(axis=0).mean()
    valid = counts > 1
    if not valid.any() or overall == 0:
        return float("nan")
    return float(within[valid].mean() / overall)


def clean_unit_fraction(activations: np.ndarray, index: np.ndarray, num_bins: int) -> float:
    profile = np.zeros((num_bins, activations.shape[1]))
    for r in range(num_bins):
        mask = index == r
        if mask.any():
            profile[r] = activations[mask].mean(axis=0)
    fracs = np.array([dominant_frequency(profile[:, u])[1] for u in range(profile.shape[1])])
    return float((fracs > 0.5).mean())


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1, h2, q = forward_with_activations(model, token_ids)

    a_grid, b_grid = np.meshgrid(np.arange(MODULUS), np.arange(MODULUS), indexing="ij")
    a_flat, b_flat = a_grid.reshape(-1), b_grid.reshape(-1)
    a0, a1 = a_flat % 10, a_flat // 10
    b0, b1 = b_flat % 10, b_flat // 10

    candidates = {
        "a0 alone": a0 % MODULUS,
        "a1 alone": a1 % MODULUS,
        "b0 alone": b0 % MODULUS,
        "b1 alone": b1 % MODULUS,
        "a0*b0 mod 97": (a0 * b0) % MODULUS,
        "a1*b0 mod 97": (a1 * b0) % MODULUS,
        "a0*b1 mod 97": (a0 * b1) % MODULUS,
        "a1*b1 mod 97": (a1 * b1) % MODULUS,
        "a1*b0 + a0*b1 mod 97": (a1 * b0 + a0 * b1) % MODULUS,
        "a*b mod 97 (full product)": (a_flat * b_flat) % MODULUS,
        "a0+b0 mod 97": (a0 + b0) % MODULUS,
        "a1+b1 mod 97": (a1 + b1) % MODULUS,
    }

    print(f"{RUN_NAME} @ {CHECKPOINT} -- collapse ratio & query clean-fraction per candidate index\n"
          f"(lower collapse = more that the query is a function of this index alone;\n"
          f" higher clean-fraction = more units show a single dominant FFT frequency over it)\n")
    print(f"{'index':<28} {'q_collapse':>11} {'q_clean':>8} {'h1_clean':>9} {'h2_clean':>9}")
    for name, idx in candidates.items():
        q_collapse = collapse_ratio(q, idx, MODULUS)
        q_clean = clean_unit_fraction(q, idx, MODULUS)
        h1_clean = clean_unit_fraction(h1, idx, MODULUS)
        h2_clean = clean_unit_fraction(h2, idx, MODULUS)
        print(f"{name:<28} {q_collapse:>11.4f} {q_clean:>8.4f} {h1_clean:>9.4f} {h2_clean:>9.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
