"""Ad hoc: line plots of several `dense1` units (by variance rank -- "y0",
"y2", "y5", ... rows 0/2/5/... of the sorted heatmap in
`modarith_multiplication_dense1_heatmap.py`) across the full multiplication
grid, one subplot per unit, so multiple units' curves can be compared
side by side.

Same layout as the other multiplication-mechanism scripts: x-axis is the
flattened `(a, b)` index `a*97 + b`, `a` outer loop / `b` inner loop (`b`
sweeps 0..96 fastest within each fixed `a` block).

Usage:
    .venv/bin/python "ad hoc scripts/modarith_multiplication_dense1_top_units_line.py"
    .venv/bin/python "ad hoc scripts/modarith_multiplication_dense1_top_units_line.py" --ranks 0 2 5

Writes `runs/modarith-baseline-v2/reports/multiplication_dense1_top_units_line.html`.
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 2, 5],
                    help="Variance-rank indices to plot (0 = highest variance). Default: 0 2 5.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)  # (97*97, window), a outer / b inner
    h1 = dense1_activations(model, token_ids)  # (9409, 512)

    order = np.argsort(-h1.var(axis=0))

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ranks = args.ranks
    fig = make_subplots(
        rows=len(ranks), cols=1, shared_xaxes=True,
        subplot_titles=[f"y{r} = dense1 unit #{int(order[r])} (variance rank {r})" for r in ranks],
    )
    tickvals = list(range(0, MODULUS * MODULUS, MODULUS * 10))
    ticktext = [str(a) for a in range(0, MODULUS, 10)]

    for i, r in enumerate(ranks, start=1):
        y = h1[:, order[r]]
        fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode="lines", name=f"y{r}"), row=i, col=1)
        fig.update_yaxes(title_text="GELU activation", row=i, col=1)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=i, col=1)

    fig.update_xaxes(title_text="a*97 + b  (a outer loop 0..96, b inner loop 0..96)", row=len(ranks), col=1)
    fig.update_layout(
        title=f"{RUN_NAME}: dense1 top-variance units, activation across the full a*b mod {MODULUS} grid",
        height=350 * len(ranks),
        width=1400,
    )

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiplication_dense1_top_units_line.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
