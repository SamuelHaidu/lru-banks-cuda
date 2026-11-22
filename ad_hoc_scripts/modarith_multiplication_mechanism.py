"""Ad hoc: how is `modarith-baseline-v2` learning `(a * b) mod 97`, and is the
same Fourier-circle mechanism found for addition (see
`modarith_addition_mechanism.py`) also present for multiplication -- or for
its natural analogue, a discrete-log embedding?

Multiplication mod a prime `p` has no additive-rotation structure in `a`, `b`
directly: `a*b mod p` is not a fixed shift of `a` as `b` varies. But under a
discrete-log map `a -> log_g(a)` for a primitive root `g` of `p` (here g=5,
p=97), multiplication BECOMES addition: `log_g(a*b) = log_g(a) + log_g(b) mod
(p-1)`. If the network has found an equivalent trick, its internal
representation should collapse onto `(log_g(a) + log_g(b)) mod 96` the same
way addition's collapses onto `(a + b) mod 97`, and hidden units should show
clean single-frequency FFT structure over THAT axis (not over `a*b mod 97`
directly, which has no such periodic structure in a linear indexing).

This probe checks, for every available checkpoint, both:
  1. Collapse of the query `q` onto `(a*b) mod 97` directly (linear index) --
     the "naive" test, expected to look messy since a*b mod 97 isn't a
     rotation.
  2. Collapse of `q` onto `(log_g(a) + log_g(b)) mod 96` (discrete-log index)
     -- the "grokked-the-trick" test.
  3. Per-unit FFT purity under both indexings, for dense1/dense2/query.

Also re-runs the same three probes for addition at every checkpoint, so any
drift in addition's already-established mechanism (weights are shared across
both tasks in this mixed-operator dataset) is visible side by side.

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_mechanism.py

Writes `runs/modarith-baseline-v2/reports/multiplication_mechanism_over_time.html`.
"""

import re
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

RUN_NAME = "modarith-baseline-v2"
MODULUS = 97
PRIMITIVE_ROOT = 5
BOS, PLUS, MINUS, STAR, EQUALS = 10, 12, 13, 14, 15


def digits(n: int) -> tuple:
    return (n // 10, n % 10)


def discrete_log_table(p: int, g: int) -> np.ndarray:
    """`log[a] = k` such that `g**k mod p == a`, for `a` in `1..p-1`. `log[0]`
    is undefined (multiplication by 0 has no log); set to -1 and excluded from
    the discrete-log-indexed probes below."""
    log = np.full(p, -1, dtype=np.int64)
    x = 1
    for k in range(p - 1):
        log[x] = k
        x = (x * g) % p
    return log


LOG = discrete_log_table(MODULUS, PRIMITIVE_ROOT)


def build_prefixes(op_token: int, window: int) -> np.ndarray:
    prefixes = []
    for a in range(MODULUS):
        a0, a1 = digits(a)
        for b in range(MODULUS):
            b0, b1 = digits(b)
            prefixes.append((BOS, a0, a1, op_token, b0, b1, EQUALS))
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
        target = torch.nn.functional.normalize(model.target_embeddings, dim=-1)
        logits = (q @ target.T) * model.tau
    return (t.detach().cpu().numpy() for t in (h1, h2, q, logits))


def dominant_frequency(signal: np.ndarray) -> tuple:
    spec = np.abs(np.fft.rfft(signal - signal.mean()))
    if spec[1:].sum() == 0:
        return 0, 0.0
    k = 1 + int(np.argmax(spec[1:]))
    frac = spec[k] ** 2 / (spec[1:] ** 2).sum()
    return k, float(frac)


def collapse_ratio(activations: np.ndarray, index: np.ndarray, num_bins: int) -> float:
    """Ratio of within-bin std (grouped by `index`) to overall std. Near 0
    means `activations` is a function of `index` alone."""
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
    counts = np.zeros(num_bins)
    for r in range(num_bins):
        mask = index == r
        counts[r] = mask.sum()
        if mask.any():
            profile[r] = activations[mask].mean(axis=0)
    fracs = np.array([dominant_frequency(profile[:, u])[1] for u in range(profile.shape[1])])
    return float((fracs > 0.5).mean())


def available_epochs(run_name: str) -> list:
    ckpt_dir = cfg_mod.run_dir(run_name) / "checkpoints"
    epochs = []
    for p in sorted(ckpt_dir.glob("checkpoint_*.keras")):
        m = re.match(r"checkpoint_(\d+)\.keras", p.name)
        if m:
            epochs.append(int(m.group(1)))
    return epochs


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))

    epochs = available_epochs(RUN_NAME)
    if not epochs:
        print("No per-epoch checkpoints found.")
        return 1

    a_grid, b_grid = np.meshgrid(np.arange(MODULUS), np.arange(MODULUS), indexing="ij")
    a_flat, b_flat = a_grid.reshape(-1), b_grid.reshape(-1)

    sum_r = (a_flat + b_flat) % MODULUS
    prod_r = (a_flat * b_flat) % MODULUS

    # Discrete-log index: undefined when a=0 or b=0 (log(0) doesn't exist).
    nonzero = (a_flat != 0) & (b_flat != 0)
    dlog_r = np.full(a_flat.shape, -1, dtype=np.int64)
    dlog_r[nonzero] = (LOG[a_flat[nonzero]] + LOG[b_flat[nonzero]]) % (MODULUS - 1)

    window = None
    rows = []
    for epoch in epochs:
        model = load_checkpoint(RUN_NAME, epoch)
        if window is None:
            window = model.context_length
            plus_ids = build_prefixes(PLUS, window)
            star_ids = build_prefixes(STAR, window)

        # -- addition, revisited --------------------------------------------
        h1p, h2p, qp, logits_p = forward_with_activations(model, plus_ids)
        add_collapse = collapse_ratio(qp, sum_r, MODULUS)
        add_h1_clean = clean_unit_fraction(h1p, sum_r, MODULUS)
        add_h2_clean = clean_unit_fraction(h2p, sum_r, MODULUS)
        add_q_clean = clean_unit_fraction(qp, sum_r, MODULUS)
        add_pred = np.argmax(logits_p, axis=-1)
        add_r0_acc = float((add_pred == (sum_r // 10)).mean())

        # -- multiplication, naive (a*b mod 97) index ------------------------
        h1s, h2s, qs, logits_s = forward_with_activations(model, star_ids)
        mul_naive_collapse = collapse_ratio(qs, prod_r, MODULUS)
        mul_naive_q_clean = clean_unit_fraction(qs, prod_r, MODULUS)

        # -- multiplication, discrete-log index (a!=0, b!=0 only) ------------
        qs_nz, h1s_nz, h2s_nz = qs[nonzero], h1s[nonzero], h2s[nonzero]
        dlog_nz = dlog_r[nonzero]
        mul_dlog_collapse = collapse_ratio(qs_nz, dlog_nz, MODULUS - 1)
        mul_dlog_h1_clean = clean_unit_fraction(h1s_nz, dlog_nz, MODULUS - 1)
        mul_dlog_h2_clean = clean_unit_fraction(h2s_nz, dlog_nz, MODULUS - 1)
        mul_dlog_q_clean = clean_unit_fraction(qs_nz, dlog_nz, MODULUS - 1)

        mul_pred = np.argmax(logits_s, axis=-1)
        mul_r0_acc = float((mul_pred == (prod_r // 10)).mean())

        rows.append(dict(
            epoch=epoch,
            add_collapse=add_collapse, add_h1_clean=add_h1_clean, add_h2_clean=add_h2_clean,
            add_q_clean=add_q_clean, add_r0_acc=add_r0_acc,
            mul_naive_collapse=mul_naive_collapse, mul_naive_q_clean=mul_naive_q_clean,
            mul_dlog_collapse=mul_dlog_collapse, mul_dlog_h1_clean=mul_dlog_h1_clean,
            mul_dlog_h2_clean=mul_dlog_h2_clean, mul_dlog_q_clean=mul_dlog_q_clean,
            mul_r0_acc=mul_r0_acc,
        ))
        print(
            f"epoch {epoch:2d} | ADD  collapse={add_collapse:.3f} q_clean={add_q_clean:.3f} r0_acc={add_r0_acc:.3f} "
            f"| MUL(naive a*b) collapse={mul_naive_collapse:.3f} q_clean={mul_naive_q_clean:.3f} "
            f"| MUL(dlog) collapse={mul_dlog_collapse:.3f} h1_clean={mul_dlog_h1_clean:.3f} "
            f"h2_clean={mul_dlog_h2_clean:.3f} q_clean={mul_dlog_q_clean:.3f} r0_acc={mul_r0_acc:.3f}"
        )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ep = [r["epoch"] for r in rows]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Addition: collapse ratio & r0 accuracy over time",
            "Addition: fraction of units with single dominant frequency",
            "Multiplication: collapse ratio (naive a*b vs. discrete-log) & r0 accuracy",
            "Multiplication (discrete-log indexed): fraction of units with single dominant frequency",
        ),
    )
    fig.add_trace(go.Scatter(x=ep, y=[r["add_collapse"] for r in rows], mode="lines+markers", name="add collapse_ratio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=[r["add_r0_acc"] for r in rows], mode="lines+markers", name="add r0_acc", line=dict(dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=ep, y=[r["add_h1_clean"] for r in rows], mode="lines+markers", name="add dense1 clean"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["add_h2_clean"] for r in rows], mode="lines+markers", name="add dense2 clean"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["add_q_clean"] for r in rows], mode="lines+markers", name="add query clean"), row=1, col=2)

    fig.add_trace(go.Scatter(x=ep, y=[r["mul_naive_collapse"] for r in rows], mode="lines+markers", name="mul collapse (a*b naive)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ep, y=[r["mul_dlog_collapse"] for r in rows], mode="lines+markers", name="mul collapse (discrete-log)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ep, y=[r["mul_r0_acc"] for r in rows], mode="lines+markers", name="mul r0_acc", line=dict(dash="dot")), row=2, col=1)

    fig.add_trace(go.Scatter(x=ep, y=[r["mul_dlog_h1_clean"] for r in rows], mode="lines+markers", name="mul(dlog) dense1 clean"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["mul_dlog_h2_clean"] for r in rows], mode="lines+markers", name="mul(dlog) dense2 clean"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["mul_dlog_q_clean"] for r in rows], mode="lines+markers", name="mul(dlog) query clean"), row=2, col=2)

    for r_, c_ in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.update_xaxes(title_text="epoch", row=r_, col=c_)

    fig.update_layout(height=900, width=1300, title_text=f"{RUN_NAME}: addition (revisited) vs. multiplication mechanism over training")

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_mechanism_over_time.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
