"""Ad hoc: `dual-sphere-hyper-token-transform`'s full CONCATENATED window
vector `u` (shape `(batch, window_size * dim_in)` = 1024-dim at the
defaults) -- the post-residual, L2-normalized flattening of ALL
`window_size` transformed token positions back together, which is what
actually feeds `dense1`. Not the per-token `transformed` (`(window, dim_in)`,
8-dim per position) that `hyper_token_transform_output_activations.py` and
`hyper_token_transform_full_window_analysis.py` look at -- this script looks
at the whole window's representation AFTER it's been reassembled into one
vector, which is a fundamentally different (and much wider) object: the
residual add and final `ops.normalize` happen at this concatenated-vector
level, not per position, so the per-token view can't tell you what `dense1`
itself actually receives.

Replays `DualSphereHyperTokenTransformLM.call()` all the way through `u`:
    flat = normalize(reshape(token_emb, (batch, w*d)))       # pre-hyper input
    ... hypernetwork generates W1/W2/W3, transforms every position ...
    transformed_flat = reshape(transformed, (batch, w*d))
    u = normalize(flat + transformed_flat)                    # -> dense1

Two views, same report:
1. **Across generation steps**: one row per step = that step's full 1024-dim
   `u` vector (heatmap, x=flat channel, y=generation step), same layout as
   `hyper_ffn_output_activations.py`'s per-token report but one level up --
   plus a single-channel line plot.
2. **Single fixed window breakdown**: on one real 128-token training excerpt,
   compares `flat` (pre-hypernetwork, residual-branch input) vs.
   `transformed_flat` (hypernetwork's contribution) vs. `u` (what dense1
   sees) as three overlaid line plots per channel, plus the residual's
   relative contribution (`||transformed_flat|| / ||flat||`) and how much
   `u` differs from a no-transform baseline (`normalize(flat)` alone, i.e.
   `transformed_flat == 0`) via cosine similarity.

Usage:
    .venv/bin/python "ad hoc scripts/hyper_token_transform_flat_vector_activations.py"
    .venv/bin/python "ad hoc scripts/hyper_token_transform_flat_vector_activations.py" --run-name hyper-token-transform-residual-baseline --checkpoint 5
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


def left_pad(ids: list, window: int) -> np.ndarray:
    ids = ids[-window:]
    pad = [0] * (window - len(ids))
    return np.array(pad + ids, dtype=np.int32)


def generate_ids(model, bos_id: int, eos_id: int, window: int, max_new_tokens: int,
                  temperature: float, seed: int) -> list:
    import torch

    rng = np.random.default_rng(seed)
    ids = [bos_id]

    for _ in range(max_new_tokens):
        x = left_pad(ids, window)[None, :]
        with torch.no_grad():
            logits_t = model(ops.convert_to_tensor(x, dtype="int32"), training=False)
            logits = np.asarray(ops.convert_to_numpy(logits_t)[0], dtype=np.float64)

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


def flat_vector_forward(model, token_ids: np.ndarray):
    """Replays `call()` up through `u` (the full concatenated, residual-added,
    L2-normalized flat vector that feeds `dense1`), plus the two components
    that sum into it (`flat`, `transformed_flat`).

    `token_ids`: (batch, window). Returns `(flat, transformed_flat, u)`, each
    `(batch, window * dim_in)`.
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

        transformed_flat = ops.reshape(transformed, (batch_size, w * d))
        u = ops.normalize(flat + transformed_flat, axis=-1, order=2)

    return (
        ops.convert_to_numpy(flat),
        ops.convert_to_numpy(transformed_flat),
        ops.convert_to_numpy(u),
    )


def find_doc_start(tokens: np.ndarray, doc_token_id: int, after: int = 0) -> int:
    hits = np.flatnonzero(tokens[after:] == doc_token_id)
    if hits.size == 0:
        raise ValueError(f"No document marker found at or after index {after}.")
    return int(after + hits[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="'latest', 'best', or an epoch number (e.g. 5).")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-ids", type=int, nargs="+", default=None,
                         help="Skip free generation entirely and run over this exact token id sequence instead "
                              "(used for the generation-steps view; the fixed-window breakdown always uses a "
                              "real training excerpt, see --doc-start).")
    parser.add_argument("--doc-start", type=int, default=None,
                         help="Token index to start the 128-token excerpt from for the fixed-window breakdown. "
                              "Default: the first training doc.")
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

    from model.data import load_tokens
    from tokenizers import Tokenizer

    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # ---- view 1: across generation steps -----------------------------------
    if args.fixed_ids is not None:
        ids = list(args.fixed_ids)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"{run_name} @ checkpoint {checkpoint} -- running on a FIXED {len(ids)}-token sequence "
              f"(not generating): {text}\n")
    else:
        print(f"{run_name} @ checkpoint {checkpoint} -- generating from <bos> only, temperature={args.temperature}...")
        ids = generate_ids(model, meta["bos_token_id"], meta["eos_token_id"], window,
                            args.max_new_tokens, args.temperature, args.seed)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"generated {len(ids)} tokens (including <bos>): {text}\n")

    token_labels = [tokenizer.id_to_token(int(t)) or f"<id {t}>" for t in ids]
    token_labels = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in token_labels]

    prefixes = np.stack([left_pad(ids[: t + 1], window) for t in range(len(ids))])  # (num_steps, window)
    _, _, u_steps = flat_vector_forward(model, prefixes)  # (num_steps, window*dim_in)

    print(f"generation-steps matrix shape: {u_steps.shape} "
          f"({len(ids)} steps x {window * d} flat channels)")

    # ---- view 2: single fixed window breakdown -----------------------------
    train_tokens = load_tokens(str(cfg_mod.dataset_split_path(config["data"]["dataset"], "train")))
    start = args.doc_start if args.doc_start is not None else find_doc_start(
        np.asarray(train_tokens[: window + 1]), meta["doc_token_id"], 0
    )
    doc = np.asarray(train_tokens[start : start + window]).astype(np.int32)
    if len(doc) < window:
        raise ValueError(f"Only {len(doc)} tokens available from index {start}; need {window}.")
    doc_text = tokenizer.decode(doc.tolist())
    print(f"\nFixed-window breakdown -- {window}-token training excerpt from index {start}:")
    print(doc_text[:300].replace("\n", "\\n"), "...\n")

    flat_vec, transformed_flat_vec, u_vec = flat_vector_forward(model, doc[None, :])
    flat_vec, transformed_flat_vec, u_vec = flat_vec[0], transformed_flat_vec[0], u_vec[0]

    residual_ratio = float(np.linalg.norm(transformed_flat_vec) / np.linalg.norm(flat_vec))
    no_transform_baseline = flat_vec / np.linalg.norm(flat_vec)
    identity_cos = cosine(u_vec, no_transform_baseline)
    print(f"||transformed_flat|| / ||flat|| = {residual_ratio:.4f} "
          f"(hypernetwork's contribution relative to the residual branch)")
    print(f"cosine(u, normalize(flat)) = {identity_cos:.4f} "
          f"(1.0 = hypernetwork correction had no directional effect on what dense1 sees; "
          f"lower = it meaningfully redirected the vector)\n")

    # ---- plots ---------------------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_steps, num_channels = u_steps.shape
    default_channel = int(u_steps.var(axis=0).argmax())
    y_labels = [f"{i}: {tok!r}" for i, tok in enumerate(token_labels)]
    x_steps = list(range(num_steps))
    x_channels = list(range(num_channels))

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.45, 0.2, 0.35],
        subplot_titles=(
            f"Full concatenated flat vector u ({num_channels} channels = window_size x dim_in) "
            f"across {num_steps} generation steps -- what dense1 actually receives",
            "single-channel line plot: u's value over generation steps (type a channel index below and press Enter)",
            f"Fixed-window breakdown (excerpt @ index {start}): flat (pre-hypernetwork) vs. transformed_flat "
            f"(hypernetwork's contribution) vs. u (post-residual, normalized) -- channel-by-channel over the "
            f"{num_channels}-wide flat vector",
        ),
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Heatmap(z=u_steps, y=y_labels, colorscale="RdBu", zmid=0,
                    colorbar=dict(title="u value", y=0.86, len=0.4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_steps, y=u_steps[:, default_channel], mode="lines",
                    name=f"channel {default_channel}", line=dict(width=1.5)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_channels, y=flat_vec, mode="lines", name="flat (pre-hyper)",
                    line=dict(width=1, color="gray")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_channels, y=transformed_flat_vec, mode="lines", name="transformed_flat (hyper contribution)",
                    line=dict(width=1, color="orange")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_channels, y=u_vec, mode="lines", name="u (post-residual, normalized)",
                    line=dict(width=1.5, color="crimson")),
        row=3, col=1,
    )

    fig.update_xaxes(title_text=f"flat channel (0..{num_channels - 1})", constrain="domain", row=1, col=1)
    fig.update_yaxes(title_text="generation step", constrain="domain", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="generation step", row=2, col=1)
    fig.update_yaxes(title_text="u value", row=2, col=1)
    fig.update_xaxes(title_text=f"flat channel (0..{num_channels - 1})", row=3, col=1)
    fig.update_yaxes(title_text="value", row=3, col=1)

    fig.update_layout(
        title=f"{run_name} @ checkpoint {checkpoint}: hypernetwork token-transform's full concatenated vector "
              f"(||transformed_flat||/||flat||={residual_ratio:.3f}, cos(u, normalize(flat))={identity_cos:.3f})",
        autosize=True,
        height=1400,
        margin=dict(l=140, r=40, t=90, b=60),
    )

    out_dir = Path(cfg_mod.run_dir(run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = "" if checkpoint == "latest" else f"_ckpt{checkpoint}"
    fixed_suffix = "_fixedtext" if args.fixed_ids is not None else ""
    out_path = out_dir / f"hyper_flat_vector_activations{checkpoint_suffix}{fixed_suffix}.html"

    matrix_json = "[" + ",".join("[" + ",".join(f"{v:.6g}" for v in row) + "]" for row in u_steps.T) + "]"
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, default_width="100vw", default_height="100vh",
                        config={"responsive": True})
    div_id = html.split('id="', 1)[1].split('"', 1)[0]

    control_and_script = f"""
<div style="position:fixed; top:8px; right:24px; z-index:1000; background:white; padding:6px 10px; border:1px solid #ccc; border-radius:4px; font-family:sans-serif;">
  <label for="channel-input">flat channel (0-{num_channels - 1}): </label>
  <input id="channel-input" type="number" min="0" max="{num_channels - 1}" value="{default_channel}" style="width:80px;">
</div>
<script>
(function() {{
  var gd = document.getElementById("{div_id}");
  var matrix = {matrix_json};
  var input = document.getElementById("channel-input");
  input.addEventListener("keydown", function(e) {{
    if (e.key !== "Enter") return;
    var u = parseInt(input.value, 10);
    if (isNaN(u) || u < 0 || u > {num_channels - 1}) return;
    Plotly.restyle(gd, {{y: [matrix[u]], name: ["channel " + u]}}, [1]);
  }});
}})();
</script>
"""
    html = html.replace("</body>", control_and_script + "</body>")

    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    label = "Fixed text" if args.fixed_ids is not None else "Generated text"
    escaped_doc = doc_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text_block = (
        f'<div style="font-family:sans-serif; padding:12px 24px; background:#f7f7f7; border-bottom:1px solid #ddd;">'
        f'<strong>{label}</strong> ({len(ids)} tokens, top view):'
        f'<div style="margin-top:6px; white-space:pre-wrap;">{escaped_text}</div>'
        f'<strong style="display:block; margin-top:10px;">Fixed-window excerpt</strong> '
        f'({window} tokens starting at index {start}, bottom view):'
        f'<div style="margin-top:6px; white-space:pre-wrap;">{escaped_doc}</div></div>'
    )
    html = html.replace("<body>", "<body>" + text_block, 1)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
