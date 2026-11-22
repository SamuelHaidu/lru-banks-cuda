"""Ad hoc: three analyses of `dual-sphere-hyper-token-transform`'s per-token
hypernetwork-FFN output (`transformed`, shape `(window_size, dim_in)`) within
a SINGLE fixed context window -- not across generation steps like
`hyper_token_transform_output_activations.py`, which only ever looked at the
current/last position of a growing window. Uses a real 128-token training
excerpt (matching the model's full `window_size`) so every position is real
content, not left-padding.

1. **Full-window matrix**: all 128 positions' hypernetwork-FFN output
   (heatmap, x=8 output channels, y=position) plus a single-channel line
   plot -- does the transform vary smoothly across position within one
   fixed window, independent of the across-generation-steps view used so
   far?

2. **Hyper-FFN output vs. raw token embedding**, position by position: for
   each of the 8 channels, overlay `transformed[:, c]` against
   `token_emb[:, c]` (what `dense-sphere`'s `dense1` would have seen
   directly, with no hypernetwork transform at all) -- how much does the
   hypernetwork actually change the representation vs. leaving it close to
   the identity map?

3. **Per-position sensitivity to the WHOLE window vs. just that position's
   own token**: `W1/W2/W3` are generated from the entire window, so the
   output at position `t` could in principle depend on tokens far from `t`,
   not just on `token[t-1]`/`token[t]`/`pos[t]` directly. Tested by holding
   the window otherwise fixed and swapping ONE token far away from a probe
   position, then measuring how much the probe position's own output shifts
   -- if it shifts non-trivially, position t's computation is genuinely
   context-dependent (hypernetwork doing real work); if it barely moves,
   position t's output is effectively a function of its own local
   (prev, this, pos) input alone, and the hypernetwork's context-sensitivity
   is negligible in practice despite being architecturally possible.

Usage:
    .venv/bin/python "ad hoc scripts/hyper_token_transform_full_window_analysis.py"
    .venv/bin/python "ad hoc scripts/hyper_token_transform_full_window_analysis.py" --run-name hyper-token-transform-residual-baseline --checkpoint 5
"""

import argparse
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

DEFAULT_RUN_NAME = "hyper-token-transform-residual-baseline"
DEFAULT_CHECKPOINT = "latest"


def hyper_ffn_forward(model, token_ids: np.ndarray):
    """Replays `call()` up through `transformed` AND returns `token_emb` too,
    so both can be compared position by position. `token_ids`: (batch, window).
    Returns `(transformed, token_emb)`, each `(batch, window, dim_in)`."""
    import torch

    batch_size = token_ids.shape[0]
    w, d = model.window_size, model.dim_in
    f = model.token_ffn_width

    with torch.no_grad():
        x = ops.convert_to_tensor(token_ids, dtype="int32")

        norm_in = model.normalized_input_embeddings()
        token_emb = ops.reshape(ops.take(norm_in, ops.reshape(x, (-1,)), axis=0), (batch_size, w, d))

        flat = ops.reshape(token_emb, (batch_size, w * d))
        flat = ops.normalize(flat, axis=-1, order=2)
        w1 = ops.reshape(model.hyper_w1_proj(flat, training=False), (batch_size, f, f))
        w2 = ops.reshape(model.hyper_w2_proj(flat, training=False), (batch_size, f, f))
        w3 = ops.reshape(model.hyper_w3_proj(flat, training=False), (batch_size, f, d))

        sentinel = ops.broadcast_to(model.no_prev_token_embedding, (batch_size, 1, d))
        prev_token = ops.concatenate([sentinel, token_emb[:, :-1, :]], axis=1)
        pos_emb = ops.broadcast_to(model.positional_embeddings[None, :, :], (batch_size, w, d))
        token_input = ops.concatenate([prev_token, token_emb, pos_emb], axis=-1)

        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", token_input, w1))
        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", h, w2))
        transformed = ops.einsum("bwi,bij->bwj", h, w3)

    return ops.convert_to_numpy(transformed), ops.convert_to_numpy(token_emb)


def find_doc_start(tokens: np.ndarray, doc_token_id: int, after: int = 0) -> int:
    hits = np.flatnonzero(tokens[after:] == doc_token_id)
    if hits.size == 0:
        raise ValueError(f"No document marker found at or after index {after}.")
    return int(after + hits[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--doc-start", type=int, default=None,
                         help="Token index to start the 128-token excerpt from. Default: the first training doc.")
    parser.add_argument("--probe-position", type=int, default=100,
                         help="Position used for the sensitivity test (default 100, deep into the window "
                              "so it's unaffected by left-padding edge effects).")
    parser.add_argument("--swap-position", type=int, default=20,
                         help="Position whose token gets swapped for the sensitivity test (default 20, far from "
                              "--probe-position so any effect on the probe is clearly a whole-window effect).")
    args = parser.parse_args()

    config = cfg_mod.load(args.run_name)
    setup_device(config.get("device"))
    model = load_checkpoint(args.run_name, args.checkpoint)
    window = model.context_length

    if not hasattr(model, "hyper_w1_proj"):
        raise ValueError(f"Run {args.run_name!r} is not a dual-sphere-hyper-token-transform model.")

    from model.data import load_tokens
    from tokenizers import Tokenizer

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    train_tokens = load_tokens(str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train")))
    start = args.doc_start if args.doc_start is not None else find_doc_start(
        np.asarray(train_tokens[: window + 1]), meta["doc_token_id"], 0
    )
    doc = np.asarray(train_tokens[start : start + window]).astype(np.int32)
    if len(doc) < window:
        raise ValueError(f"Only {len(doc)} tokens available from index {start}; need {window}.")

    tokenizer = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    token_labels = [tokenizer.id_to_token(int(t)) or f"<id {t}>" for t in doc]
    token_labels = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in token_labels]
    print(f"{args.run_name} @ checkpoint {args.checkpoint} -- {window}-token training excerpt from index {start}:")
    print(tokenizer.decode(doc.tolist())[:300].replace("\n", "\\n"), "...\n")

    # ---- 1 & 2: full-window transform + embedding comparison ---------------
    transformed, token_emb = hyper_ffn_forward(model, doc[None, :])
    transformed, token_emb = transformed[0], token_emb[0]  # (window, dim_in) each

    d = model.dim_in
    identity_corr = [float(np.corrcoef(transformed[:, c], token_emb[:, c])[0, 1]) for c in range(d)]
    identity_mse = float(np.mean((transformed - token_emb) ** 2))
    raw_var = float(np.var(token_emb))
    print(f"transformed vs. raw token_emb: per-channel correlation = {np.round(identity_corr, 3).tolist()}")
    print(f"transformed vs. raw token_emb: MSE = {identity_mse:.4f}  (raw token_emb variance = {raw_var:.4f}, "
          f"ratio = {identity_mse / raw_var:.3f} -- ~0 means near-identity, ~1+ means substantially different)\n")

    # ---- 3: sensitivity of one position's output to a distant swap --------
    probe, swap = args.probe_position, args.swap_position
    baseline_output = transformed[probe].copy()

    rng = np.random.default_rng(0)
    num_trials = 20
    shifts = np.zeros(num_trials)
    for i in range(num_trials):
        doc_perturbed = doc.copy()
        doc_perturbed[swap] = int(rng.integers(0, meta["vocabulary_size"]))
        t_perturbed, _ = hyper_ffn_forward(model, doc_perturbed[None, :])
        shifts[i] = float(np.linalg.norm(t_perturbed[0, probe] - baseline_output))

    # Median (not one specific pair) L2 distance between EVERY pair of
    # adjacent positions -- a single pair can land next to an outlier
    # position and give a wildly unrepresentative scale (this happened with
    # position 100 vs. 99 initially, since positions >=120 in this window
    # have anomalously large output norms -- see the printed per-position
    # norms above/report heatmap).
    adjacent_diffs = np.linalg.norm(transformed[1:] - transformed[:-1], axis=1)
    within_position_scale = float(np.median(adjacent_diffs))
    print(f"Sensitivity test: probe position={probe}, swap position={swap} (distance={abs(probe - swap)})")
    print(f"  L2 shift in position {probe}'s output from swapping position {swap}'s token "
          f"({num_trials} random replacements): mean={shifts.mean():.4f}  std={shifts.std():.4f}  max={shifts.max():.4f}")
    print(f"  (for scale: MEDIAN L2 distance between adjacent positions' own outputs across the "
          f"whole window = {within_position_scale:.4f} -- avoids picking one specific, possibly "
          f"outlier-adjacent pair)")
    print(f"  ratio (swap-induced shift / median adjacent-position difference): {shifts.mean() / within_position_scale:.3f}\n")

    # ---- plots ---------------------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.45, 0.25, 0.3],
        subplot_titles=(
            f"Hypernetwork-FFN output, all {window} positions ({d} channels, "
            f"color clipped to 95th percentile)",
            "Per-position output L2 norm (shows outlier positions the heatmap's color scale hides)",
            "Sensitivity: L2 shift in probe output per random swap trial",
        ),
    )
    # Percentile-based color range, not the true min/max -- a handful of
    # outlier positions can otherwise saturate the whole scale and make
    # every other position look uniformly flat by comparison (this happened
    # with positions >=120 in the default excerpt, which have output norms
    # 5-20x the median).
    z_bound = float(np.percentile(np.abs(transformed), 95))
    fig.add_trace(go.Heatmap(z=transformed, y=[f"{i}: {t!r}" for i, t in enumerate(token_labels)],
                              colorscale="RdBu", zmid=0, zmin=-z_bound, zmax=z_bound,
                              colorbar=dict(title="output", x=0.44)), row=1, col=1)
    position_norms = np.linalg.norm(transformed, axis=1)
    fig.add_trace(go.Scatter(x=list(range(window)), y=position_norms, mode="lines+markers",
                              name="output L2 norm"), row=1, col=2)

    fig.add_trace(go.Bar(x=list(range(num_trials)), y=shifts, name="L2 shift"), row=1, col=3)
    fig.add_hline(y=within_position_scale, line_dash="dash",
                  annotation_text="median adjacent-position scale", row=1, col=3)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="output channel", row=1, col=1)
    fig.update_yaxes(title_text="position", row=1, col=1)
    fig.update_xaxes(title_text="position", row=1, col=2)
    fig.update_yaxes(title_text="L2 norm", row=1, col=2)
    fig.update_xaxes(title_text="trial", row=1, col=3)
    fig.update_yaxes(title_text="L2 shift", row=1, col=3)
    fig.update_layout(
        title=f"{args.run_name} @ checkpoint {args.checkpoint}: full-window hypernetwork-FFN analysis "
              f"(doc starting at token index {start})",
        height=700, width=1500,
    )

    out_dir = Path(cfg_mod.run_dir(args.run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = "" if args.checkpoint == "latest" else f"_ckpt{args.checkpoint}"
    out_path = out_dir / f"hyper_ffn_full_window_analysis{checkpoint_suffix}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
