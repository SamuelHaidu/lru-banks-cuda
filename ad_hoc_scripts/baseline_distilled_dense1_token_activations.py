"""Ad hoc: activation matrix of any named layer, for any run, one row per
token of text the model GENERATES ITSELF from `<bos>` alone (temperature
0.5), stopping naturally at `<eos>` -- not a training excerpt or a
hand-written prompt. Also shows the generated text at the top of the report
and a single-unit line plot (unit chosen via a text box, not a dropdown --
scrolling through 1024 dropdown entries is unusable) of that unit's
activation across every generated token position.

Two parameters make this work for any architecture: `--run-name` (which run
to load) and `--layer-name` (which sub-layer's output to record, e.g.
`dense1`, `dense2`, `trans_mlp1` -- any attribute name the architecture
assigned a layer to in its `__init__`). A forward hook captures that layer's
output during a normal `model(...)` call, so there is no per-architecture
branching to reimplement each forward pass by hand -- this works for
`dual-sphere`, `alibi-sphere`, or any future `BaseLM` unmodified.

The model only ever predicts ONE next token from a fixed causal window (see
`model/interfaces.py`'s `BaseLM` docstring) -- there is no persistent
per-position hidden state the way a transformer with a KV-cache carries
state between generation steps; every step recomputes the full forward pass
over the current window from scratch. So "activation at token t" here means:
feed the model the causal prefix ending at token `t` (left zero-padded to
`window_size`), exactly what happens internally at generation step `t`, and
record the named layer's output for that prefix.

Generation is implemented locally (mirrors `model/generation.py`'s loop)
rather than calling `model.generate()` and re-tokenizing its returned text,
because re-encoding decoded text is not guaranteed to round-trip to the
exact same ids -- this way the ids used for the activation matrix are
provably the same ids that were actually sampled.

Usage:
    .venv/bin/python ad_hoc_scripts/baseline_distilled_dense1_token_activations.py
    .venv/bin/python ad_hoc_scripts/baseline_distilled_dense1_token_activations.py --run-name baseline --layer-name dense1
    .venv/bin/python ad_hoc_scripts/baseline_distilled_dense1_token_activations.py --run-name transformer-alibi --layer-name dense1
"""

import argparse
from pathlib import Path

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

DEFAULT_RUN_NAME = "baseline-from-distilled-table"
DEFAULT_LAYER_NAME = "dense1"
DEFAULT_CHECKPOINT = "latest"


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
    """Local greedy/temperature sampling loop, mirrors `model/generation.py`
    exactly, but returns the raw id sequence (not just decoded text) so the
    activation matrix below is guaranteed to match what was actually sampled.
    Calls `model(...)` directly -- the real, architecture-specific `call()`
    -- so this works unmodified for `alibi-sphere` or any other `BaseLM`."""
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
    parser.add_argument("--layer-name", type=str, default=DEFAULT_LAYER_NAME,
                         help="Attribute name of the layer to record, e.g. dense1, dense2, trans_mlp1.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="'latest', 'best', or an epoch number (e.g. 5) -- see `resolve_checkpoint`.")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-ids", type=int, nargs="+", default=None,
                         help="Skip free generation entirely and run the activation matrix over this exact token id "
                             "sequence instead -- lets two different checkpoints be compared on IDENTICAL input, "
                             "since autoregressive sampling diverges between checkpoints even with the same seed "
                             "(different logits -> different sampled token -> different subsequent context).")
    args = parser.parse_args()
    run_name = args.run_name
    layer_name = args.layer_name
    checkpoint = args.checkpoint

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
        generated_text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"{run_name} @ checkpoint {checkpoint} -- running on a FIXED {len(ids)}-token sequence "
              f"(not generating): {generated_text}\n")
    else:
        print(f"{run_name} @ checkpoint {checkpoint} -- generating from <bos> only, temperature={args.temperature}...")
        ids = generate_ids(model, meta["bos_token_id"], meta["eos_token_id"], window,
                            args.max_new_tokens, args.temperature, args.seed)
        generated_text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"generated {len(ids)} tokens (including <bos>): {generated_text}\n")

    token_labels = [tokenizer.id_to_token(int(t)) or f"<id {t}>" for t in ids]
    token_labels = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in token_labels]

    prefixes = np.stack([left_pad(ids[: t + 1], window) for t in range(len(ids))])  # (num_tokens, window)
    matrix = layer_activations(model, layer_name, prefixes)  # (num_tokens, num_units) -- x=units, y=tokens

    print(f"matrix shape: {matrix.shape} ({len(ids)} tokens x {matrix.shape[1]} '{layer_name}' units)")

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
            f"'{layer_name}' activations ({num_units} units) across {num_tokens} self-generated tokens",
            "single-unit line plot: activation value over token positions (type a unit index below and press Enter)",
        ),
        vertical_spacing=0.14,
    )

    fig.add_trace(
        go.Heatmap(z=matrix, y=y_labels, colorscale="RdBu", zmid=0, colorbar=dict(title="activation", y=0.85, len=0.6)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_positions, y=matrix[:, default_unit], mode="lines", name=f"unit {default_unit}", line=dict(width=1.5)),
        row=2, col=1,
    )

    fig.update_xaxes(title_text=f"'{layer_name}' unit (0..{num_units - 1})", constrain="domain", row=1, col=1)
    fig.update_yaxes(title_text="token position", constrain="domain", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="token position", row=2, col=1)
    fig.update_yaxes(title_text="activation value", row=2, col=1)

    fig.update_layout(
        title=f"{run_name} @ checkpoint {checkpoint}: '{layer_name}' activations on self-generated text "
              f"(temperature={args.temperature}, seed={args.seed})",
        autosize=True,
        margin=dict(l=140, r=40, t=90, b=60),
    )

    out_dir = Path(cfg_mod.run_dir(run_name)) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint suffix in the filename: "latest"/"best" reuse the same
    # filename as before (unchanged default behavior), but a specific epoch
    # number gets its own file so it never overwrites a later checkpoint's
    # report -- e.g. comparing an early-training snapshot against `latest`
    # requires both to exist on disk at once.
    checkpoint_suffix = "" if checkpoint == "latest" else f"_ckpt{checkpoint}"
    fixed_suffix = "_fixedtext" if args.fixed_ids is not None else ""
    out_path = out_dir / f"{layer_name}_token_activations{checkpoint_suffix}{fixed_suffix}.html"

    # Text box (not a dropdown -- 1024 scrollable options is unusable) that
    # drives the line plot via a small inline script operating on Plotly's
    # JS API directly. `full_html=True` (write_html's default) gives us a
    # <body> we can append this control + script into after the figure's
    # own markup, referencing the same div id Plotly assigned it.
    matrix_json = "[" + ",".join("[" + ",".join(f"{v:.6g}" for v in row) + "]" for row in matrix.T) + "]"
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, default_width="100vw", default_height="100vh",
                        config={"responsive": True})
    div_id = html.split('id="', 1)[1].split('"', 1)[0]

    control_and_script = f"""
<div style="position:fixed; top:8px; right:24px; z-index:1000; background:white; padding:6px 10px; border:1px solid #ccc; border-radius:4px; font-family:sans-serif;">
  <label for="unit-input">'{layer_name}' unit (0-{num_units - 1}): </label>
  <input id="unit-input" type="number" min="0" max="{num_units - 1}" value="{default_unit}" style="width:80px;">
</div>
<script>
(function() {{
  var gd = document.getElementById("{div_id}");
  var matrix = {matrix_json};  // [unit][token] activation values, precomputed server-side
  var input = document.getElementById("unit-input");
  input.addEventListener("keydown", function(e) {{
    if (e.key !== "Enter") return;
    var u = parseInt(input.value, 10);
    if (isNaN(u) || u < 0 || u > {num_units - 1}) return;
    Plotly.restyle(gd, {{y: [matrix[u]], name: ["unit " + u]}}, [1]);
  }});
}})();
</script>
"""
    html = html.replace("</body>", control_and_script + "</body>")

    # Prepend the generated text as readable HTML above the chart.
    escaped_text = generated_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text_block = (
        f'<div style="font-family:sans-serif; padding:12px 24px; background:#f7f7f7; border-bottom:1px solid #ddd;">'
        f'<strong>Generated text</strong> (temperature={args.temperature}, seed={args.seed}, {len(ids)} tokens):'
        f'<div style="margin-top:6px; white-space:pre-wrap;">{escaped_text}</div></div>'
    )
    html = html.replace("<body>", "<body>" + text_block, 1)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
