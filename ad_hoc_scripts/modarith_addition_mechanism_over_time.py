"""Ad hoc: does `modarith-baseline`'s Fourier-circle representation for
`(a + b) mod 97` (see `modarith_addition_mechanism.py`) appear gradually,
epoch by epoch, or all at once late in training (the grokking pattern)?

Replays the same three probes -- anti-diagonal collapse ratio, per-unit FFT
frequency purity, and `+`-exact-match accuracy -- against every saved
checkpoint (`runs/modarith-baseline/checkpoints/checkpoint_NNNNN.keras`), so
the structure's emergence can be tracked across the whole training run.

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_addition_mechanism_over_time.py

Writes `runs/modarith-baseline/reports/addition_mechanism_over_time.html`.
"""

import re
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint, resolve_checkpoint  # noqa: E402

RUN_NAME = "modarith-baseline"
MODULUS = 97
BOS, PLUS = 10, 12
EQUALS = 15


def digits(n: int) -> tuple:
    return (n // 10, n % 10)


def build_prefixes(window: int) -> np.ndarray:
    prefixes = []
    for a in range(MODULUS):
        a0, a1 = digits(a)
        for b in range(MODULUS):
            b0, b1 = digits(b)
            prefixes.append((BOS, a0, a1, PLUS, b0, b1, EQUALS))
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


def clean_unit_fraction(activations: np.ndarray, r_flat: np.ndarray) -> float:
    profile = np.zeros((MODULUS, activations.shape[1]))
    for r in range(MODULUS):
        profile[r] = activations[r_flat == r].mean(axis=0)
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
    r_flat = (a_flat + b_flat) % MODULUS

    rows = []
    window = None
    for epoch in epochs:
        model = load_checkpoint(RUN_NAME, epoch)
        if window is None:
            window = model.context_length
            token_ids = build_prefixes(window)

        h1, h2, q, logits = forward_with_activations(model, token_ids)

        within_r_std = np.array([q[r_flat == r].std(axis=0).mean() for r in range(MODULUS)]).mean()
        overall_std = q.std(axis=0).mean()
        collapse_ratio = within_r_std / overall_std if overall_std else float("nan")

        h1_clean = clean_unit_fraction(h1, r_flat)
        h2_clean = clean_unit_fraction(h2, r_flat)
        q_clean = clean_unit_fraction(q, r_flat)

        pred = np.argmax(logits, axis=-1)
        # `r` predicted here is the r0 digit only (single next-token after `=`).
        true_r0 = (r_flat // 10)
        r0_acc = float((pred == true_r0).mean())

        rows.append(dict(epoch=epoch, collapse_ratio=collapse_ratio, h1_clean=h1_clean,
                          h2_clean=h2_clean, q_clean=q_clean, r0_acc=r0_acc))
        print(f"epoch {epoch:3d}: collapse_ratio={collapse_ratio:.4f}  "
              f"h1_clean={h1_clean:.3f}  h2_clean={h2_clean:.3f}  q_clean={q_clean:.3f}  "
              f"r0_acc={r0_acc:.4f}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ep = [r["epoch"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Anti-diagonal collapse ratio (lower = more (a+b)-only)",
        "Fraction of units with a single dominant FFT frequency, and r0 accuracy",
    ))
    fig.add_trace(go.Scatter(x=ep, y=[r["collapse_ratio"] for r in rows], mode="lines+markers", name="collapse_ratio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=[r["h1_clean"] for r in rows], mode="lines+markers", name="dense1 clean frac"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["h2_clean"] for r in rows], mode="lines+markers", name="dense2 clean frac"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["q_clean"] for r in rows], mode="lines+markers", name="query clean frac"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=[r["r0_acc"] for r in rows], mode="lines+markers", name="r0 accuracy", line=dict(dash="dot")), row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_layout(height=500, width=1200, title_text=f"{RUN_NAME}: emergence of the (a+b) mod {MODULUS} circuit over training")

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "addition_mechanism_over_time.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
