"""Ad hoc: curve-fits every `dense1` unit's activation-vs-generation-step
trace (from self-generated text) against a small family of candidate
functional forms, to check whether the "exponential decay" units spotted by
eye (e.g. #218, #766 on `distill-gpt-to-baseline`) are actually
exponential, or just smoothly monotonic in a way that could equally be
linear/logistic/power-law -- eyeballing a smooth curve and calling it
"exponential" isn't enough; this fits and compares.

Candidate families (all fit by nonlinear least squares against `activation
value ~ f(token position)`):
  - constant:    a
  - linear:      a + b*t
  - exponential: a + b*exp(-c*t)      (decay if c>0, growth if c<0)
  - logistic:    a + b / (1 + exp(-c*(t - d)))
  - power law:   a + b*(t+1)**c       (t+1 to avoid t=0 singularities)

Model selection via AIC (Akaike Information Criterion, penalizes extra
parameters) on the residual sum of squares -- not just raw R^2, which always
improves with more free parameters and would unfairly favor logistic/power
law over exponential just because they have one more knob.

Usage:
    .venv/bin/python "ad hoc scripts/dense1_activation_curve_fitting.py"
    .venv/bin/python "ad hoc scripts/dense1_activation_curve_fitting.py" --run-name distill-gpt-to-baseline --seed 0
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

DEFAULT_RUN_NAME = "distill-gpt-to-baseline"
CHECKPOINT = "latest"


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


def left_pad(ids: list, window: int) -> np.ndarray:
    ids = ids[-window:]
    pad = [0] * (window - len(ids))
    return np.array(pad + ids, dtype=np.int32)


def generate_ids(model, bos_id: int, eos_id: int, window: int, max_new_tokens: int,
                  temperature: float, seed: int) -> list:
    import torch

    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    ids = [bos_id]

    for _ in range(max_new_tokens):
        x = left_pad(ids, window)[None, :]
        with torch.no_grad():
            logits = model(torch.as_tensor(x, dtype=torch.int32).to(device), training=False)
            logits = logits[0].detach().cpu().numpy().astype(np.float64)

        if temperature <= 0.0:
            next_id = int(np.argmax(logits))
        else:
            logits = logits / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            next_id = int(rng.choice(probs.size, p=probs))

        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids


# -- candidate curve families ------------------------------------------------

def f_constant(t, a):
    return np.full_like(t, a, dtype=np.float64)


def f_linear(t, a, b):
    return a + b * t


def f_exponential(t, a, b, c):
    return a + b * np.exp(-c * t)


def f_logistic(t, a, b, c, d):
    # Clip the exponent, not just suppress the warning -- an optimizer
    # excursion into a huge |c*(t-d)| would otherwise silently produce inf/nan
    # through exp() and corrupt that trial's residual rather than just being
    # numerically noisy, since np.exp(-huge_negative) legitimately overflows
    # to inf in float64 well before it's mathematically infinite.
    z = np.clip(-c * (t - d), -700, 700)
    return a + b / (1.0 + np.exp(z))


def f_power(t, a, b, c):
    return a + b * np.power(t + 1.0, c)


CANDIDATES = {
    "constant": (f_constant, 1),
    "linear": (f_linear, 2),
    "exponential": (f_exponential, 3),
    "logistic": (f_logistic, 4),
    "power_law": (f_power, 3),
}


def initial_guess(name: str, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    y0, y1, ymean = y[0], y[-1], y.mean()
    if name == "constant":
        return np.array([ymean])
    if name == "linear":
        slope = (y1 - y0) / max(t[-1] - t[0], 1)
        return np.array([y0, slope])
    if name == "exponential":
        return np.array([y1, y0 - y1, 1.0 / max(len(t) / 4, 1)])
    if name == "logistic":
        return np.array([y0, y1 - y0, 0.1, len(t) / 2])
    if name == "power_law":
        return np.array([y1, y0 - y1, -0.5])
    raise ValueError(name)


def fit_all_candidates(t: np.ndarray, y: np.ndarray) -> dict:
    """Returns `{name: (aic, params, fitted_curve)}` for every candidate that
    converges; families that fail to fit are simply omitted (their AIC would
    be undefined, not a fair comparison point anyway)."""
    from scipy.optimize import curve_fit

    n = len(t)
    results = {}
    for name, (fn, num_params) in CANDIDATES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(fn, t, y, p0=initial_guess(name, t, y), maxfev=20000)
            fitted = fn(t, *popt)
            rss = float(np.sum((y - fitted) ** 2))
            if rss <= 0:
                rss = 1e-12
            # AIC = n*ln(RSS/n) + 2*k -- standard form for Gaussian-error least squares.
            aic = n * np.log(rss / n) + 2 * num_params
            results[name] = (aic, popt, fitted)
        except (RuntimeError, ValueError):
            continue
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=20, help="How many units to print per winning family.")
    parser.add_argument("--units", type=int, nargs="+", default=None,
                         help="Specific unit indices to always report the full fit table for, regardless of ranking "
                              "(e.g. units spotted by eye in the heatmap report).")
    args = parser.parse_args()

    config = cfg_mod.load(args.run_name)
    setup_device(config.get("device"))
    model = load_checkpoint(args.run_name, CHECKPOINT)
    window = model.context_length

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])

    print(f"{args.run_name} @ {CHECKPOINT} -- regenerating (seed={args.seed}, temperature={args.temperature}) "
          f"to reproduce the same sequence, then curve-fitting all 1024 dense1 units...\n")
    ids = generate_ids(model, meta["bos_token_id"], meta["eos_token_id"], window,
                        args.max_new_tokens, args.temperature, args.seed)
    print(f"generated {len(ids)} tokens\n")

    prefixes = np.stack([left_pad(ids[: k + 1], window) for k in range(len(ids))])
    matrix = dense1_activations(model, prefixes)  # (num_tokens, 1024)

    t = np.arange(matrix.shape[0], dtype=np.float64)
    num_units = matrix.shape[1]

    winners = {}  # unit -> (best_family, aic_gap_to_runner_up)
    family_counts = {name: 0 for name in CANDIDATES}
    per_unit_best = []

    for u in range(num_units):
        y = matrix[:, u].astype(np.float64)
        if y.std() < 1e-6:
            continue  # dead/constant unit, not interesting either way
        fits = fit_all_candidates(t, y)
        if not fits:
            continue
        ranked = sorted(fits.items(), key=lambda kv: kv[1][0])  # by AIC ascending (lower=better)
        best_name, (best_aic, best_params, _) = ranked[0]
        runner_up_aic = ranked[1][1][0] if len(ranked) > 1 else float("inf")
        family_counts[best_name] += 1
        per_unit_best.append((u, best_name, best_aic, runner_up_aic - best_aic, best_params))

    print("Best-fit family counts across all (non-dead) dense1 units:")
    for name, count in sorted(family_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>12}: {count}")
    print()

    for target_family in ("exponential", "linear"):
        subset = [row for row in per_unit_best if row[1] == target_family]
        subset.sort(key=lambda row: -row[3])  # sort by AIC gap over runner-up, descending (most confidently that family)
        print(f"Top {min(args.top, len(subset))} units most confidently fit by '{target_family}' "
              f"(by AIC margin over the next-best family):")
        for u, name, aic, gap, params in subset[: args.top]:
            print(f"  unit {u:4d}  AIC={aic:8.2f}  margin_over_runner_up={gap:7.2f}  params={np.round(params, 4).tolist()}")
        print()

    if args.units:
        print(f"Full fit table for requested units {args.units}:\n")
        for u in args.units:
            y = matrix[:, u].astype(np.float64)
            print(f"unit {u}: std={y.std():.5f}  min={y.min():.4f}  max={y.max():.4f}")
            fits = fit_all_candidates(t, y)
            ranked = sorted(fits.items(), key=lambda kv: kv[1][0])
            for rank, (name, (aic, params, _)) in enumerate(ranked):
                marker = " <-- best" if rank == 0 else ""
                print(f"    {name:>12}  AIC={aic:8.2f}  params={np.round(params, 4).tolist()}{marker}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
