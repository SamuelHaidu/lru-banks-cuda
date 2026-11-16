# rnn-research

Spherical language models on Intel XPU.

A token's context window is looked up in a small spherical embedding table,
compressed by a bias-free FFN into a query on a wider sphere, and read out as
cosine similarity against a second embedding table. One architecture, one
dataset, one backend.

```bash
.venv/bin/model create   --run-name baseline
.venv/bin/model train    --run-name baseline
.venv/bin/model evaluate --run-name baseline
.venv/bin/model generate --run-name baseline --prompt "Once upon a time"
```

No environment variables need to be set: the XPU is the default device.

See [AGENTS.md](AGENTS.md) for the architecture, the layout, and the rules that
keep it small.

## CUDA large-batch note

On an A40, `data.batch_size=73728` with `training.learning_rate=0.02` trains
stably (spherical normalization keeps activations bounded, allowing a much
higher LR than plain transformers tolerate at this batch size). Memory in
use: ~42GiB.
