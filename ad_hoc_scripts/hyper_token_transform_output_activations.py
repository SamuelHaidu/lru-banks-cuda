"""Ad hoc: the hypernetwork-generated FFN's ACTUAL OUTPUT (`transformed`,
shape `(batch, window_size, dim_in)` -- one `dim_in`-wide vector per token
position) for `dual-sphere-hyper-token-transform`, not the generated WEIGHT
MATRICES (`hyper_w1_proj`/`hyper_w2_proj`/`hyper_w3_proj`) that
`dense_layers_token_activations.py` captures via a forward hook.

That distinction matters: `hyper_w1_proj(flat)` (and w2/w3) are called ONCE
per forward pass, on the whole flattened window -- their output IS the
`(24,24)`/`(24,24)`/`(24,8)` weight matrices themselves, computed from
context alone, before any per-token computation happens. `transformed` is
what you get after actually APPLYING those generated weights to every
position's own `concat(prev_token, this_token, pos_emb)` input via
`GELU(x @ W1) -> GELU(· @ W2) -> · @ W3` -- the real per-token output of the
hypernetwork stage, and what actually flows (via the residual) into
`dense1`. Neither `transformed` nor the einsum/gelu steps that produce it
are a named Keras layer, so no forward hook can capture them; this script
replays `call()` manually via `keras.ops` up through `transformed` instead.

Same report layout as the other token-activation scripts: heatmap (x=the
`dim_in` output channels, y=token position) plus a text-box-driven
single-channel line plot, over text the model generates itself from `<bos>`
(temperature 0.5, stops at `<eos>`).

Usage:
    .venv/bin/python ad_hoc_scripts/hyper_token_transform_output_activations.py
    .venv/bin/python ad_hoc_scripts/hyper_token_transform_output_activations.py --run-name hyper-token-transform-residual-baseline --checkpoint 5
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


def hyper_ffn_output(model, token_ids: np.ndarray) -> np.ndarray:
    """Replays `DualSphereHyperTokenTransformLM.call()` up through
    `transformed` (the hypernetwork-generated FFN's real per-token output),
    via `keras.ops` -- backend-agnostic, matches training/inference numerics
    exactly since it's the same ops the model itself calls.

    Returns `(num_rows, window_size, dim_in)`.
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
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-ids", type=int, nargs="+", default=None,
                         help="Skip free generation entirely and run over this exact token id sequence instead.")
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
    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

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

    prefixes = np.stack([left_pad(ids[: t + 1], window) for t in range(len(ids))])  # (num_tokens, window)
    full_output = hyper_ffn_output(model, prefixes)  # (num_tokens, window, dim_in)

    # `transformed`'s row at generation step t is only meaningful for the
    # CURRENT position (index -1, i.e. token t itself) -- earlier rows of
    # that same forward pass belong to earlier tokens' own (shorter) context
    # windows at a PAST generation step, not to token t. Take the diagonal:
    # for step t, the position-of-interest is the last real (non-padded)
    # row, i.e. index `t` counting from the left after left-padding, which
    # is simply `full_output[t, -1, :]` since the window is left-padded to
    # put the current token last.
    matrix = full_output[:, -1, :]  # (num_tokens, dim_in)

    print(f"matrix shape: {matrix.shape} ({len(ids)} tokens x {d} hypernetwork-FFN output channels)")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_tokens, num_units = matrix.shape
    default_unit = int(matrix.var(axis=0).argmax())
    y_labels = [f"{i}: {tok!r}" for i, tok in enumerate(token_labels)]
    x_positions = list(range(num_tokens))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"hypernetwork FFN output ({num_units} channels) across {num_tokens} tokens -- "
            f"the actual per-token transform, not the generated weight matrices",
            "single-channel line plot: output value over token positions (type a channel index below and press Enter)",
        ),
        vertical_spacing=0.14,
    )
    fig.add_trace(
        go.Heatmap(z=matrix, y=y_labels, colorscale="RdBu", zmid=0, colorbar=dict(title="output value", y=0.85, len=0.6)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_positions, y=matrix[:, default_unit], mode="lines", name=f"channel {default_unit}", line=dict(width=1.5)),
        row=2, col=1,
    )
    fig.update_xaxes(title_text=f"output channel (0..{num_units - 1})", constrain="domain", row=1, col=1)
    fig.update_yaxes(title_text="token position", constrain="domain", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="token position", row=2, col=1)
    fig.update_yaxes(title_text="output value", row=2, col=1)

    fig.update_layout(
        title=f"{run_name} @ checkpoint {checkpoint}: hypernetwork-generated FFN's actual per-token output",
        autosize=True,
        margin=dict(l=140, r=40, t=90, b=60),
    )

    out_dir = Path(cfg_mod.run_dir(run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = "" if checkpoint == "latest" else f"_ckpt{checkpoint}"
    fixed_suffix = "_fixedtext" if args.fixed_ids is not None else ""
    out_path = out_dir / f"hyper_ffn_output_activations{checkpoint_suffix}{fixed_suffix}.html"

    matrix_json = "[" + ",".join("[" + ",".join(f"{v:.6g}" for v in row) + "]" for row in matrix.T) + "]"
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, default_width="100vw", default_height="100vh",
                        config={"responsive": True})
    div_id = html.split('id="', 1)[1].split('"', 1)[0]

    control_and_script = f"""
<div style="position:fixed; top:8px; right:24px; z-index:1000; background:white; padding:6px 10px; border:1px solid #ccc; border-radius:4px; font-family:sans-serif;">
  <label for="unit-input">output channel (0-{num_units - 1}): </label>
  <input id="unit-input" type="number" min="0" max="{num_units - 1}" value="{default_unit}" style="width:80px;">
</div>
<script>
(function() {{
  var gd = document.getElementById("{div_id}");
  var matrix = {matrix_json};
  var input = document.getElementById("unit-input");
  input.addEventListener("keydown", function(e) {{
    if (e.key !== "Enter") return;
    var u = parseInt(input.value, 10);
    if (isNaN(u) || u < 0 || u > {num_units - 1}) return;
    Plotly.restyle(gd, {{y: [matrix[u]], name: ["channel " + u]}}, [1]);
  }});
}})();
</script>
"""
    html = html.replace("</body>", control_and_script + "</body>")

    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    label = "Fixed text" if args.fixed_ids is not None else "Generated text"
    text_block = (
        f'<div style="font-family:sans-serif; padding:12px 24px; background:#f7f7f7; border-bottom:1px solid #ddd;">'
        f'<strong>{label}</strong> ({len(ids)} tokens):'
        f'<div style="margin-top:6px; white-space:pre-wrap;">{escaped_text}</div></div>'
    )
    html = html.replace("<body>", "<body>" + text_block, 1)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
