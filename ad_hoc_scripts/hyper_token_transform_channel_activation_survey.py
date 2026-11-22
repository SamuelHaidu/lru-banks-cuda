"""Ad hoc: is `dual-sphere-hyper-token-transform`'s per-position hypernetwork
output (`transformed`, `dim_in=8` channels per position) split into an
ALWAYS-ON subset of channels (constant/high-magnitude at nearly every
position, in every sample) vs. an EVENT-LIKE subset that only activates at
specific token types (e.g. sentence/clause boundaries) -- the pattern
`hyper_token_transform_flat_vector_activations.py`'s generation-steps heatmap
hinted at via vertical stripes, but that report mixes in left-padding
boundary artifacts from autoregressive generation. This script instead uses
real TEACHER-FORCED training data (`FullEvalDataset`, one seeded target per
document, left-padded per-document exactly like training) and restricts
every statistic to `valid` (non-padding) positions only, so there is no
padding-boundary confound.

For every valid (sample, position) pair across many documents, extracts the
8-dim `transformed` vector, then per channel:
    - mean |activation| and its std across all valid positions/samples
      (high mean, low relative std -> "always on")
    - fraction of positions where |activation| exceeds a global per-channel
      threshold (e.g. its own 90th percentile) -- "activation rate"
    - for high-activation-rate (event-like) channels: which TOKENS most often
      co-occur with a firing event, via a token -> mean-activation table for
      that channel, to see whether it lines up with punctuation / doc
      boundaries / specific words.

Two report views:
1. Per-channel summary bar chart (mean |activation|, activation rate) across
   all 8 channels -- identifies which channels are always-on vs. sparse.
2. Per-sample heatmap (x=position within the real, unpadded content; y=one
   row per sampled document) for the two most "event-like" channels
   (highest activation-rate variance across positions) side by side with the
   two most "always-on" channels, so the always-on/event-like split is
   visible directly, plus a token frequency table for the top event-like
   channel's firing positions.

Usage:
    .venv/bin/python ad_hoc_scripts/hyper_token_transform_channel_activation_survey.py
    .venv/bin/python ad_hoc_scripts/hyper_token_transform_channel_activation_survey.py \\
        --run-name hyper-token-transform-residual-baseline --checkpoint 5 --num-docs 200
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

DEFAULT_RUN_NAME = "hyper-token-transform-residual-baseline"
DEFAULT_CHECKPOINT = "latest"


def hyper_ffn_output_and_mask(model, token_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replays `call()` up through `transformed` (per-position hypernetwork-FFN
    output). `token_ids`, `valid`: (batch, window). Returns `(batch, window, dim_in)`.
    """
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
        transformed = ops.einsum("bwi,bij->bwj", h, w3)  # (batch, w, d)

    return ops.convert_to_numpy(transformed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="'latest', 'best', or an epoch number (e.g. 5).")
    parser.add_argument("--num-docs", type=int, default=512,
                         help="Number of documents (one teacher-forced target position each, per "
                              "FullEvalDataset's seeded scheme) to sample.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--display-docs", type=int, default=40,
                         help="Number of sampled documents to actually plot in the per-sample heatmaps "
                              "(the full --num-docs is used for the summary statistics).")
    args = parser.parse_args()
    run_name = args.run_name
    checkpoint = args.checkpoint

    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))
    model = load_checkpoint(run_name, checkpoint)
    window = model.context_length
    d = model.dim_in

    if not hasattr(model, "hyper_w1_proj"):
        raise ValueError(f"Run {run_name!r} (arch={config['arch']!r}) is not a "
                          f"dual-sphere-hyper-token-transform model -- no hypernetwork to inspect.")

    from tokenizers import Tokenizer

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    tokenizer = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))

    print(f"{run_name} @ checkpoint {checkpoint} -- teacher-forced survey (validation set, one seeded "
          f"target position per doc, corrupt_prob=0)")

    from model.data import _document_bounds, _fill_windows, load_tokens

    val_path = str(cfg_mod.dataset_split_path(config["data"]["dataset"], "validation"))
    tokens = load_tokens(val_path)
    doc_starts, doc_ends = _document_bounds(val_path, tokens, meta["doc_token_id"], use_cache=True)
    num_docs = min(args.num_docs, len(doc_starts))
    doc_starts, doc_ends = doc_starts[:num_docs], doc_ends[:num_docs]

    rng = np.random.default_rng(0)
    targets = rng.integers(doc_starts + 1, doc_ends)
    x, valid = _fill_windows(tokens, targets, doc_starts, window)

    transformed_list = []
    batch = args.batch_size
    for s in range(0, num_docs, batch):
        chunk = x[s : s + batch]
        transformed_list.append(hyper_ffn_output_and_mask(model, chunk, None))
    transformed = np.concatenate(transformed_list, axis=0)  # (num_docs, window, d)

    valid_flat = valid.reshape(-1)
    transformed_flat = transformed.reshape(-1, d)[valid_flat]  # (num_valid_positions, d)
    tokens_flat = x.reshape(-1)[valid_flat]

    print(f"valid positions analyzed: {transformed_flat.shape[0]} "
          f"(out of {valid.size} total (doc, position) slots)")

    # ---- per-channel summary (dim_in=8 level -- coarse, mostly uniform) ----
    abs_act = np.abs(transformed_flat)
    mean_abs = abs_act.mean(axis=0)
    std_abs = abs_act.std(axis=0)
    # Global (not per-channel) threshold: the 90th percentile of ALL
    # magnitudes pooled across every channel. A per-channel threshold makes
    # every channel "fire" exactly 10% of the time by construction (self-
    # normalizing), which can't distinguish always-on from event-like at all.
    global_threshold = float(np.percentile(abs_act, 90))
    activation_rate = (abs_act >= global_threshold).mean(axis=0)
    cv = std_abs / (mean_abs + 1e-8)

    print("\nPer-channel summary (dim_in=8 level, over all valid positions across all sampled docs):")
    print(f"{'channel':>8} {'mean|act|':>10} {'std|act|':>10} {'CV':>8} {'act_rate(global 90th pctl)':>28}")
    for c in range(d):
        print(f"{c:>8} {mean_abs[c]:>10.4f} {std_abs[c]:>10.4f} {cv[c]:>8.3f} {activation_rate[c]:>28.3f}")

    # ---- per-(position, channel) summary (the real 1024-wide flat-vector
    # unit -- this is what the flat-vector heatmap's vertical stripes are
    # actually indexed by, not the 8 raw dim_in channels alone). ----
    abs_by_pos = np.full((window, d), np.nan)
    std_by_pos = np.full((window, d), np.nan)
    for p in range(window):
        pos_valid = valid[:, p]
        if not pos_valid.any():
            continue
        vals = np.abs(transformed[pos_valid, p, :])
        abs_by_pos[p] = vals.mean(axis=0)
        std_by_pos[p] = vals.std(axis=0)
    cv_by_pos = std_by_pos / (abs_by_pos + 1e-8)

    # Flatten to the same 1024-wide layout as the flat vector (position-major,
    # channel-minor: index = p*d + c) and rank by mean magnitude to find the
    # "always-on" flat-vector units the earlier stripes pointed at.
    flat_mean = abs_by_pos.reshape(-1)
    flat_cv = cv_by_pos.reshape(-1)
    valid_units = ~np.isnan(flat_mean)
    ranked = np.argsort(-np.where(valid_units, flat_mean, -np.inf))
    print("\nTop 15 highest-mean-magnitude (position, channel) units in the 1024-wide flat layout "
          "(these are the candidate 'always-on' stripe sources):")
    for idx in ranked[:15]:
        p, c = divmod(int(idx), d)
        print(f"  flat_index={idx:>4}  position={p:>3}  channel={c}  mean|act|={flat_mean[idx]:>8.4f}  "
              f"CV={flat_cv[idx]:.3f}")

    # Most channel-consistent "always-on" signal: which position-channel pairs
    # have BOTH high mean magnitude AND low CV (truly constant, not just
    # occasionally huge).
    score = np.where(valid_units, flat_mean / (flat_cv + 1e-8), -np.inf)
    always_on_ranked = np.argsort(-score)
    print("\nTop 10 (position, channel) units with high magnitude AND low CV (strongest 'always-on' candidates):")
    for idx in always_on_ranked[:10]:
        p, c = divmod(int(idx), d)
        print(f"  flat_index={idx:>4}  position={p:>3}  channel={c}  mean|act|={flat_mean[idx]:>8.4f}  "
              f"CV={flat_cv[idx]:.3f}")

    always_on_channels = list(np.argsort(cv)[:2])          # lowest CV = most constant (dim_in-level)
    event_like_channels = list(np.argsort(-cv)[:2])         # highest CV = most spiky/event-driven

    # Token co-occurrence for the top event-like channel's firing positions
    # (dim_in-level, using the corrected global threshold).
    top_event_channel = event_like_channels[0]
    firing = abs_act[:, top_event_channel] >= global_threshold
    firing_token_ids = tokens_flat[firing]
    unique, counts = np.unique(firing_token_ids, return_counts=True)
    order = np.argsort(-counts)[:20]
    print(f"\nTop 20 tokens co-occurring with channel {top_event_channel}'s firing events "
          f"({firing.sum()} firing positions out of {len(firing)}, global 90th-pctl threshold):")
    for tid, cnt in zip(unique[order], counts[order]):
        tok = tokenizer.id_to_token(int(tid)) or f"<id {tid}>"
        tok = tok.replace("Ġ", " ").replace("Ċ", "\\n")
        print(f"  {tok!r:>15}  count={cnt:>5}  ({cnt / firing.sum() * 100:.1f}% of firings)")

    # ---- plots ---------------------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_pos = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Mean |activation| by (position, channel) -- teacher-forced, valid positions only",
            "Coefficient of variation by (position, channel) -- low = always-on, high = event-like",
        ),
    )
    fig_pos.add_trace(
        go.Heatmap(z=abs_by_pos.T, colorscale="Viridis", colorbar=dict(title="mean|act|", x=0.44)),
        row=1, col=1,
    )
    fig_pos.add_trace(
        go.Heatmap(z=cv_by_pos.T, colorscale="Viridis", colorbar=dict(title="CV", x=1.0)),
        row=1, col=2,
    )
    fig_pos.update_xaxes(title_text="position", row=1, col=1)
    fig_pos.update_xaxes(title_text="position", row=1, col=2)
    fig_pos.update_yaxes(title_text="channel (dim_in)", row=1, col=1)
    fig_pos.update_yaxes(title_text="channel (dim_in)", row=1, col=2)
    fig_pos.update_layout(
        title="Per-(position, channel) magnitude/CV -- the direct source of the flat-vector heatmap's stripes",
        height=400, width=1500,
    )

    display_n = min(args.display_docs, transformed.shape[0])
    disp_transformed = transformed[:display_n]  # (display_n, window, d)
    disp_valid = valid[:display_n]
    disp_tokens = x[:display_n]

    def masked_row(sample_idx, channel):
        row = disp_transformed[sample_idx, :, channel].copy()
        row[~disp_valid[sample_idx]] = np.nan
        return row

    display_channels = always_on_channels + event_like_channels
    channel_labels = (
        [f"ch{c} (always-on, CV={cv[c]:.2f})" for c in always_on_channels]
        + [f"ch{c} (event-like, CV={cv[c]:.2f})" for c in event_like_channels]
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=tuple(f"{lbl} across {display_n} docs (padding shown as gaps)" for lbl in channel_labels),
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), c in zip(positions, display_channels):
        z = np.stack([masked_row(i, c) for i in range(display_n)])
        z_bound = float(np.nanpercentile(np.abs(z), 95)) or 1.0
        fig.add_trace(
            go.Heatmap(z=z, colorscale="RdBu", zmid=0, zmin=-z_bound, zmax=z_bound, showscale=(row, col) == (1, 1)),
            row=row, col=col,
        )
    fig.update_yaxes(title_text="document (sample)", row=1, col=1)
    fig.update_yaxes(title_text="document (sample)", row=2, col=1)
    fig.update_xaxes(title_text="position (real content right-aligned)", row=2, col=1)
    fig.update_xaxes(title_text="position (real content right-aligned)", row=2, col=2)

    fig.update_layout(
        title=f"{run_name} @ checkpoint {checkpoint}: always-on vs. event-like hypernetwork-FFN output channels "
              f"({transformed.shape[0]} teacher-forced docs, valid positions only)",
        height=900, width=1500,
    )

    fig2 = go.Figure(data=[
        go.Bar(name="mean |activation|", x=list(range(d)), y=mean_abs),
        go.Bar(name="std |activation|", x=list(range(d)), y=std_abs),
    ])
    fig2.add_trace(go.Scatter(x=list(range(d)), y=activation_rate, name="activation rate (>=90th pctl)",
                               yaxis="y2", mode="lines+markers", line=dict(color="black", dash="dot")))
    fig2.update_layout(
        title="Per-channel activation summary (higher activation-rate variance across channels = "
              "some channels fire much more selectively than others)",
        xaxis_title="channel", yaxis_title="magnitude",
        yaxis2=dict(title="activation rate", overlaying="y", side="right", range=[0, 1]),
        barmode="group", height=450,
    )

    out_dir = Path(cfg_mod.run_dir(run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = "" if checkpoint == "latest" else f"_ckpt{checkpoint}"
    out_path = out_dir / f"hyper_channel_activation_survey{checkpoint_suffix}.html"

    html_pos = fig_pos.to_html(include_plotlyjs="cdn", full_html=False)
    html1 = fig.to_html(include_plotlyjs=False, full_html=False)
    html2 = fig2.to_html(include_plotlyjs=False, full_html=False)

    always_on_rows = "".join(
        f"<tr><td>{idx}</td><td>{divmod(int(idx), d)[0]}</td><td>{divmod(int(idx), d)[1]}</td>"
        f"<td>{flat_mean[idx]:.4f}</td><td>{flat_cv[idx]:.3f}</td></tr>"
        for idx in always_on_ranked[:10]
    )
    top_magnitude_rows = "".join(
        f"<tr><td>{idx}</td><td>{divmod(int(idx), d)[0]}</td><td>{divmod(int(idx), d)[1]}</td>"
        f"<td>{flat_mean[idx]:.4f}</td><td>{flat_cv[idx]:.3f}</td></tr>"
        for idx in ranked[:15]
    )

    token_rows = "".join(
        f"<tr><td>{tokenizer.id_to_token(int(tid)) or f'&lt;id {tid}&gt;'!r}</td><td>{cnt}</td>"
        f"<td>{cnt / firing.sum() * 100:.1f}%</td></tr>"
        for tid, cnt in zip(unique[order], counts[order])
    )
    summary_table = "".join(
        f"<tr><td>{c}</td><td>{mean_abs[c]:.4f}</td><td>{std_abs[c]:.4f}</td>"
        f"<td>{cv[c]:.3f}</td><td>{activation_rate[c]:.3f}</td></tr>"
        for c in range(d)
    )

    full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>hyper channel activation survey</title></head>
<body style="font-family:sans-serif; margin:24px;">
<h2>{run_name} @ checkpoint {checkpoint}: hypernetwork-FFN output channel activation survey</h2>
<p>{transformed.shape[0]} teacher-forced documents (validation set), {valid.sum()} valid (non-padding) positions analyzed.</p>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
<tr><th>channel</th><th>mean |act|</th><th>std |act|</th><th>CV</th><th>activation rate (global 90th pctl)</th></tr>
{summary_table}
</table>
{html2}
{html_pos}
<h3>Top 10 (position, channel) units with high magnitude AND low CV -- strongest "always-on" candidates</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
<tr><th>flat_index</th><th>position</th><th>channel</th><th>mean |act|</th><th>CV</th></tr>
{always_on_rows}
</table>
<h3>Top 15 highest-mean-magnitude (position, channel) units</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
<tr><th>flat_index</th><th>position</th><th>channel</th><th>mean |act|</th><th>CV</th></tr>
{top_magnitude_rows}
</table>
{html1}
<h3>Top 20 tokens co-occurring with channel {top_event_channel}'s firing events ({firing.sum()} firing positions out of {len(firing)}, global 90th-pctl threshold)</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
<tr><th>token</th><th>count</th><th>% of firings</th></tr>
{token_rows}
</table>
</body></html>
"""
    with open(out_path, "w") as f:
        f.write(full_html)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
