"""Ad hoc: reverse-engineer how `modarith-baseline` computes `(a + b) mod 97`,
in the spirit of Nanda et al.'s grokking-modular-addition analysis.

That paper's transformer embeds each *residue* `a` in its own row of an
embedding table, so the natural test is "does the embedding table form a
Fourier circle over 0..96". This architecture is different: `a` is spelled
out as two DIGIT tokens (`a0`, `a1`), each looked up in a 10-row table shared
across all input positions, and the window position (not a separate
embedding row) is what tells the FFN whether a digit belongs to `a` or `b`.
There is no dedicated "embedding of a" to Fourier-analyze directly.

So the adapted questions are:
  1. Is the QUERY (`q`, the FFN's final normalized output, which is what gets
     dot-producted against the target table to produce logits) a function of
     `(a + b) mod 97` alone -- i.e. constant along every anti-diagonal of the
     `(a, b)` grid? That is the behavioral signature of "the network computes
     the sum mod 97 and forgets the individual operands", independent of
     representation.
  2. Do individual hidden units (post dense1 / dense2) show periodic
     structure in `(a + b) mod 97`? If so, at what frequencies -- an FFT over
     the residue axis reveals whether the network builds the classic
     cos/sin-of-residue ("clock") representation or something else (e.g. a
     digit-wise adder-with-carry, which would show up as structure aligned
     with `a0+b0` and a carry term rather than a single clean frequency).
  3. PCA of `q` across the `(a, b)` grid, colored by `(a + b) mod 97`: a
     clean ring/ellipse is the qualitative "yes, it grokked a clock" picture
     from the paper; a scattered/grid-like point cloud says it's doing
     something more like digit-local table lookup.

Usage:
    .venv/bin/python "ad hoc scripts/modarith_addition_mechanism.py"

Writes plots to `runs/modarith-baseline/reports/addition_mechanism.png`.
"""

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

RUN_NAME = "modarith-baseline"
CHECKPOINT = "latest"
MODULUS = 97
BOS, PLUS = 10, 12
EQUALS = 15


def digits(n: int) -> tuple:
    return (n // 10, n % 10)


def build_prefixes(window: int) -> np.ndarray:
    """All 97*97 `<bos> a0 a1 + b0 b1 =` prefixes, left-padded to `window`."""
    prefixes = []
    for a in range(MODULUS):
        a0, a1 = digits(a)
        for b in range(MODULUS):
            b0, b1 = digits(b)
            prefixes.append((BOS, a0, a1, PLUS, b0, b1, EQUALS))
    prefixes = np.array(prefixes, dtype=np.int32)  # (97*97, 7)
    pad = np.zeros((prefixes.shape[0], window - 7), dtype=np.int32)
    return np.concatenate([pad, prefixes], axis=1)


def forward_with_activations(model, token_ids: np.ndarray):
    """Replays `DualSphereLM.call`, keeping every intermediate tensor."""
    import torch

    device = next(model.parameters()).device if hasattr(model, "parameters") else None
    with torch.no_grad():
        ids = torch.as_tensor(token_ids, dtype=torch.int32)
        if device is not None:
            ids = ids.to(device)
        norm_in = model.normalized_input_embeddings()
        flat = norm_in[ids.reshape(-1)]
        u = flat.reshape(ids.shape[0], model.window_size * model.dim_in)
        u = torch.nn.functional.normalize(u, dim=-1)
        h1 = model.dense1(u)
        h2 = model.dense2(h1)
        z = model.dense3(h2)
        q = torch.nn.functional.normalize(z, dim=-1)
    return (t.detach().cpu().numpy() for t in (h1, h2, z, q))


def dominant_frequency(signal_grid: np.ndarray) -> tuple:
    """FFT magnitude spectrum of a length-97 signal indexed by residue; returns
    (top frequency index, fraction of spectral energy at that frequency)."""
    spec = np.abs(np.fft.rfft(signal_grid - signal_grid.mean()))
    if spec[1:].sum() == 0:
        return 0, 0.0
    k = 1 + int(np.argmax(spec[1:]))
    frac = spec[k] ** 2 / (spec[1:] ** 2).sum()
    return k, float(frac)


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    token_ids = build_prefixes(window)
    h1, h2, z, q = forward_with_activations(model, token_ids)

    a_grid, b_grid = np.meshgrid(np.arange(MODULUS), np.arange(MODULUS), indexing="ij")
    a_flat, b_flat = a_grid.reshape(-1), b_grid.reshape(-1)
    r_flat = (a_flat + b_flat) % MODULUS

    # ---- 1. Is q a function of r = (a+b) % 97 alone? ------------------------
    # For each residue r, collect every (a, b) pair with a+b=r (mod 97) and
    # measure the spread of q across that anti-diagonal vs. the spread of q
    # overall. If q only depends on r, within-r spread should be near zero.
    within_r_std = np.zeros(MODULUS)
    for r in range(MODULUS):
        mask = r_flat == r
        within_r_std[r] = q[mask].std(axis=0).mean()
    overall_std = q.std(axis=0).mean()
    print(f"q spread within an anti-diagonal (a+b=const): mean std = {within_r_std.mean():.4f}")
    print(f"q spread overall (across all a,b):            std     = {overall_std:.4f}")
    print(f"ratio (near 0 => q depends only on a+b mod 97): {within_r_std.mean() / overall_std:.4f}\n")

    # ---- 2. Per-unit periodicity in r, via a per-residue mean + FFT --------
    def residue_profile(activations: np.ndarray) -> np.ndarray:
        """(97, num_units): mean activation of each unit, grouped by residue."""
        out = np.zeros((MODULUS, activations.shape[1]))
        for r in range(MODULUS):
            out[r] = activations[r_flat == r].mean(axis=0)
        return out

    for name, acts in (("dense1 (h1)", h1), ("dense2 (h2)", h2), ("query (q)", q)):
        profile = residue_profile(acts)  # (97, units)
        freqs, fracs = [], []
        for u in range(profile.shape[1]):
            k, frac = dominant_frequency(profile[:, u])
            freqs.append(k)
            fracs.append(frac)
        freqs, fracs = np.array(freqs), np.array(fracs)
        clean = fracs > 0.5  # unit's variation over r is dominated by one frequency
        print(f"{name}: {clean.sum()}/{len(freqs)} units have >50% of their "
              f"residue-wise variance at a single FFT frequency")
        if clean.sum():
            top_freqs, counts = np.unique(freqs[clean], return_counts=True)
            order = np.argsort(-counts)
            summary = ", ".join(f"k={top_freqs[i]}(n={counts[i]})" for i in order[:8])
            print(f"    dominant frequencies among those: {summary}")
    print()

    # ---- 3. PCA of q, colored by residue -----------------------------------
    q_centered = q - q.mean(axis=0)
    _, _, vt = np.linalg.svd(q_centered, full_matrices=False)
    pcs = q_centered @ vt[:2].T

    from pathlib import Path

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir = Path(cfg_mod.run_dir(RUN_NAME)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    pc1_by_r = np.array([pcs[r_flat == r, 0].mean() for r in range(MODULUS)])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Query q, top-2 PCs, colored by (a+b) mod 97",
            "Query PC1, averaged per residue r=(a+b) mod 97",
        ),
    )
    fig.add_trace(
        go.Scattergl(
            x=pcs[:, 0],
            y=pcs[:, 1],
            mode="markers",
            marker=dict(size=3, color=r_flat, colorscale="HSV", showscale=True, colorbar=dict(title="r")),
            text=[f"a={a} b={b} r={r}" for a, b, r in zip(a_flat, b_flat, r_flat)],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=list(range(MODULUS)), y=pc1_by_r, mode="lines+markers"), row=1, col=2)
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_xaxes(title_text="r", row=1, col=2)
    fig.update_yaxes(title_text="mean PC1", row=1, col=2)
    fig.update_layout(height=500, width=1100, title_text=f"{RUN_NAME}: addition mechanism ((a+b) mod {MODULUS})")

    out_path = out_dir / "addition_mechanism.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
