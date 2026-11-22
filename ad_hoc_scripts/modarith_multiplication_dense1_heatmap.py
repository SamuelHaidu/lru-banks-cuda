"""Ad hoc: heatmap of `modarith-baseline-v2`'s `dense1` activations (512
GELU units) across the full multiplication grid, to eyeball for visual
periodic/structural patterns the scalar collapse-ratio/FFT probes in
`modarith_multiplication_mechanism.py` and
`modarith_multiplication_digit_terms.py` might miss (those only test one
candidate index at a time; a raw heatmap can reveal structure -- stripes,
blocks, diagonals -- without first guessing what axis to index by).

Layout: input is `<bos> a0 a1 * b0 b1 =` for every `(a, b)` in `[0,97)^2`,
flattened with `a` as the outer loop and `b` as the inner loop (so column
index = `a*97 + b`, matching "00, 01, 02, ..., 96, 10, 11, ..." -- `b` sweeps
fastest within each fixed `a`). Rows are the 512 `dense1` units. Colored by
raw GELU activation.

Usage:
    .venv/bin/python ad_hoc_scripts/modarith_multiplication_dense1_heatmap.py

Writes `runs/modarith-baseline-v2/reports/multiplication_dense1_heatmap.html`.
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

    # Order units by a simple heuristic (variance) so any structured units
    # cluster near the top rather than being scattered at random row indices
    # -- purely a display aid, doesn't change the underlying data.
    order = np.argsort(-h1.var(axis=0))
    h1_sorted = h1[:, order].T  # (512 units, 9409 columns)

    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=h1_sorted,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="GELU activation"),
        )
    )
    # Tick marks every 97 columns instead of 97 individual vline shapes --
    # same "where does each `a` block start" information at a fraction of
    # the file size/render cost.
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(0, MODULUS * MODULUS, MODULUS * 10)),
        ticktext=[str(a) for a in range(0, MODULUS, 10)],
    )
    fig.update_layout(
        title=f"{RUN_NAME}: dense1 activations (512 units, rows sorted by variance) "
              f"across the full a*b mod {MODULUS} grid (a outer loop, b inner loop -- col = a*97+b)",
        xaxis_title="a (tick label), b sweeps 0..96 within each a-block",
        yaxis_title="dense1 unit (sorted by variance, descending)",
        height=900,
        width=1400,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_dense1_heatmap.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
