"""Ad hoc: line plot of a single `dense1` unit's activation (the
highest-variance unit, "y0" -- row 0 of the sorted heatmap in
`modarith_multiplication_dense1_heatmap.py`) across the full multiplication
grid, as a 1D curve rather than a heatmap row -- easier to spot periodicity,
sharp jumps, or a clean functional shape by eye than a color-coded strip.

Same layout as the heatmap script: x-axis is the flattened `(a, b)` index
`a*97 + b`, `a` outer loop / `b` inner loop (so `b` sweeps 0..96 fastest
within each fixed `a` block).

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_dense1_unit0_line.py"

Writes `runs/modarith-baseline-v2/reports/multiplication_dense1_unit0_line.html`.
"""

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
    """`a` outer loop, `b` inner loop -- column order 00,01,...,0_96,10,11,...`"""
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


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)  # (97*97, window), a outer / b inner
    h1 = dense1_activations(model, token_ids)  # (9409, 512)

    # Same "sort by variance" convention as the heatmap script, so unit 0
    # here is the same row 0 shown at the top of that heatmap.
    order = np.argsort(-h1.var(axis=0))
    y0 = h1[:, order[0]]  # (9409,)
    unit_index = int(order[0])

    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter(x=list(range(len(y0))), y=y0, mode="lines"))
    fig.update_xaxes(
        title_text="a*97 + b  (a outer loop 0..96, b inner loop 0..96)",
        tickmode="array",
        tickvals=list(range(0, MODULUS * MODULUS, MODULUS * 10)),
        ticktext=[str(a) for a in range(0, MODULUS, 10)],
    )
    fig.update_layout(
        title=f"{RUN_NAME}: dense1 unit #{unit_index} (highest variance, \"y0\") "
              f"activation across the full a*b mod {MODULUS} grid",
        yaxis_title="GELU activation",
        height=500,
        width=1400,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_dense1_unit0_line.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
