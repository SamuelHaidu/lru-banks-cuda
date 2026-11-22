"""Ad hoc: activation matrices for dense1, dense2, AND dense3 together in one
HTML report, for a single run + checkpoint, either on text the model
generates itself from `<bos>` (temperature 0.5, stops at `<eos>`) or on a
fixed token sequence passed explicitly (`--fixed-ids`) -- letting two
checkpoints of the same run be compared on identical input, since
autoregressive sampling diverges between checkpoints even with the same seed.

Same generic layer-capture mechanism as
`baseline_distilled_dense1_token_activations.py` (a forward hook, works for
any `BaseLM`), just run three times (once per layer) against one shared
generated/fixed token sequence, and combined into a single figure: three
heatmaps stacked vertically, each with its own single-unit line-plot picker
(separate text boxes, since dense1/dense2/dense3 have different unit counts).

Usage:
    .venv/bin/python "ad hoc scripts/dense_layers_token_activations.py" --run-name gpu-retriver-tokenid --checkpoint 5
    .venv/bin/python "ad hoc scripts/dense_layers_token_activations.py" --run-name gpu-retriver-tokenid --checkpoint latest \\
        --fixed-ids 0 436 452 260 399 ...
"""

import argparse
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

DEFAULT_RUN_NAME = "gpu-retriver-tokenid"
DEFAULT_CHECKPOINT = "latest"
LAYER_NAMES = ("dense1", "dense2", "dense3")


def layer_activations(model, layer_name: str, token_ids: np.ndarray) -> np.ndarray:
    """Runs a real forward pass and captures `getattr(model, layer_name)`'s
    output via a forward hook -- generic across architectures, since every
    Keras layer under the torch backend is a real `torch.nn.Module`."""
    layer = getattr(model, layer_name)
    captured = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = layer.register_forward_hook(hook)
    try:
        import torch

        with torch.no_grad():
            model(ops.convert_to_tensor(token_ids, dtype="int32"), training=False)
    finally:
        handle.remove()

    if "output" not in captured:
        raise RuntimeError(f"Layer {layer_name!r} was never called during the forward pass.")
    return ops.convert_to_numpy(captured["output"])


def left_pad(ids: list, window: int) -> np.ndarray:
    ids = ids[-window:]
    pad = [0] * (window - len(ids))
    return np.array(pad + ids, dtype=np.int32)


def generate_ids(model, bos_id: int, eos_id: int, window: int, max_new_tokens: int,
                  temperature: float, seed: int) -> list:
    """Local greedy/temperature sampling loop, mirrors `model/generation.py`."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="'latest', 'best', or an epoch number (e.g. 5).")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-ids", type=int, nargs="+", default=None,
                         help="Skip free generation entirely and run all three layers over this exact token id "
                              "sequence instead -- lets two different checkpoints be compared on IDENTICAL input.")
    parser.add_argument("--layer-names", type=str, nargs="+", default=list(LAYER_NAMES),
                         help="Which layer attribute names to include, in order (default: dense1 dense2 dense3). "
                              "e.g. --layer-names dense1 gate_dense dense2 dense3 for dual-sphere-mid-gate.")
    args = parser.parse_args()
    run_name = args.run_name
    checkpoint = args.checkpoint
    layer_names = args.layer_names

    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))
    model = load_checkpoint(run_name, checkpoint)
    window = model.context_length

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

    matrices = {}
    for layer_name in layer_names:
        m = layer_activations(model, layer_name, prefixes)
        matrices[layer_name] = m
        print(f"{layer_name}: matrix shape {m.shape}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_tokens = len(ids)
    y_labels = [f"{i}: {tok!r}" for i, tok in enumerate(token_labels)]
    x_positions = list(range(num_tokens))

    num_layers = len(layer_names)
    row_heights = []
    for _ in layer_names:
        row_heights += [0.7 / num_layers, 0.3 / num_layers]

    subplot_titles = []
    for layer_name in layer_names:
        num_units = matrices[layer_name].shape[1]
        subplot_titles.append(f"'{layer_name}' activations ({num_units} units) across {num_tokens} tokens")
        subplot_titles.append(f"'{layer_name}' single-unit line plot (type a unit index in the box and press Enter)")

    fig = make_subplots(
        rows=num_layers * 2, cols=1,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
    )

    default_units = {}
    for i, layer_name in enumerate(layer_names):
        matrix = matrices[layer_name]
        num_units = matrix.shape[1]
        default_unit = int(matrix.var(axis=0).argmax())
        default_units[layer_name] = default_unit

        heatmap_row = i * 2 + 1
        line_row = i * 2 + 2

        fig.add_trace(
            go.Heatmap(z=matrix, y=y_labels, colorscale="RdBu", zmid=0,
                       colorbar=dict(title="activation", len=1 / num_layers, y=1 - (i + 0.35) / num_layers)),
            row=heatmap_row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_positions, y=matrix[:, default_unit], mode="lines",
                       name=f"{layer_name} unit {default_unit}", line=dict(width=1.5)),
            row=line_row, col=1,
        )

        fig.update_xaxes(title_text=f"'{layer_name}' unit (0..{num_units - 1})", constrain="domain", row=heatmap_row, col=1)
        fig.update_yaxes(title_text="token position", constrain="domain", autorange="reversed", row=heatmap_row, col=1)
        fig.update_xaxes(title_text="token position", row=line_row, col=1)
        fig.update_yaxes(title_text="activation value", row=line_row, col=1)

    fig.update_layout(
        title=f"{run_name} @ checkpoint {checkpoint}: dense1/dense2/dense3 activations "
              f"({'fixed text' if args.fixed_ids is not None else f'temperature={args.temperature}, seed={args.seed}'})",
        autosize=True,
        margin=dict(l=140, r=40, t=110, b=60),
        showlegend=False,
    )

    out_dir = Path(cfg_mod.run_dir(run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = "" if checkpoint == "latest" else f"_ckpt{checkpoint}"
    fixed_suffix = "_fixedtext" if args.fixed_ids is not None else ""
    out_path = out_dir / f"dense_layers_token_activations{checkpoint_suffix}{fixed_suffix}.html"

    html = fig.to_html(include_plotlyjs="cdn", full_html=True, default_width="100vw",
                        default_height=f"{350 * num_layers}px", config={"responsive": True})
    div_id = html.split('id="', 1)[1].split('"', 1)[0]

    # One text box + one Plotly.restyle call per layer, each targeting its
    # own line-plot trace index. Trace order in the figure is
    # [heatmap0, line0, heatmap1, line1, heatmap2, line2, ...], so layer i's
    # line trace is at index 2*i + 1.
    controls = []
    scripts = []
    for i, layer_name in enumerate(layer_names):
        matrix = matrices[layer_name]
        num_units = matrix.shape[1]
        trace_index = 2 * i + 1
        matrix_json = "[" + ",".join("[" + ",".join(f"{v:.6g}" for v in row) + "]" for row in matrix.T) + "]"
        controls.append(
            f'<div style="margin-bottom:4px;">'
            f'<label for="unit-input-{i}">\'{layer_name}\' unit (0-{num_units - 1}): </label>'
            f'<input id="unit-input-{i}" type="number" min="0" max="{num_units - 1}" '
            f'value="{default_units[layer_name]}" style="width:80px;"></div>'
        )
        scripts.append(f"""
  (function() {{
    var matrix_{i} = {matrix_json};
    var input_{i} = document.getElementById("unit-input-{i}");
    input_{i}.addEventListener("keydown", function(e) {{
      if (e.key !== "Enter") return;
      var u = parseInt(input_{i}.value, 10);
      if (isNaN(u) || u < 0 || u > {num_units - 1}) return;
      Plotly.restyle(gd, {{y: [matrix_{i}[u]], name: ["{layer_name} unit " + u]}}, [{trace_index}]);
    }});
  }})();""")

    control_and_script = f"""
<div style="position:fixed; top:8px; right:24px; z-index:1000; background:white; padding:8px 10px; border:1px solid #ccc; border-radius:4px; font-family:sans-serif;">
  {''.join(controls)}
</div>
<script>
(function() {{
  var gd = document.getElementById("{div_id}");
  {''.join(scripts)}
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
