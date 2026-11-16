"""Run configuration: one file, one home for every value.

`runs/<run-name>/config.json` is the *only* place a run's hyperparameters
live. No argparse flag anywhere in this codebase carries a hyperparameter
default -- that pattern (a flag default silently winning over a persisted
config value) caused three separate, hard-to-find bugs in the previous
codebase, one of which corrupted an evaluation and inverted an architecture
comparison.

Dataset-level facts (`doc_token_id`, `avg_chars_per_token`,
`vocabulary_size`, tokenizer name) deliberately do *not* live here. They live
in `datasets/<name>/metadata.json`, so two runs on the same corpus can never
disagree about them. A run config only names its dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"

# The single source of defaults. `model create` writes this out (merged with
# the architecture's own defaults and the dataset's metadata); everything
# after that is `model config set`.
DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": None,
    "arch": "dual-sphere",
    "arch_params": {},
    "data": {
        "dataset": "tinystories-10k",
        "batch_size": 512,
        # rows per forward pass when it is less than batch_size: the gradients of batch_size / micro_batch_size
        # passes are summed into one update (null: the whole batch in one pass)
        "micro_batch_size": None,
        # with training.stateful: batches between jumps of the streams to new random document starts
        "stream_chunks": 16,
        # with training.dense_validation: validation rows' length and count (null: the model's window, batch_size)
        "validation_window": None,
        "validation_batch_size": None,
        "steps_per_epoch": 1000,
        # packed archs: one epoch = one pass over the corpus in shuffled windows of the model's window, from a random
        # offset each epoch (rows start anywhere in a document, zero state); steps_per_epoch is then derived
        "random_windows": False,
        # with random_windows: the fraction of a pass one epoch covers (0.25: four epochs per pass)
        "epoch_passes": 1.0,
        "corrupt_prob": 0.15,
        "seed": 42,
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        # warmup-stable-decay: from update `lr_decay_start` the learning rate falls linearly to `lr_min_ratio` x
        # over `lr_decay_steps` updates (0: no decay, constant after the warmup)
        "lr_decay_start": 0,
        "lr_decay_steps": 0,
        "lr_min_ratio": 0.1,
        "weight_decay": 0.01,
        "epochs": 100,
        "validation_steps": 50,
        # `torch.compile` fusion of the train/test step. Keras's torch backend
        # never enables this automatically (`jit_compile="auto"` always
        # resolves to False on this backend) -- it must be requested per run.
        # Off by default: the compiler backend doesn't work on this project's
        # Intel Iris Xe XPU, so this is meant to be flipped on only for CUDA.
        "jit_compile": False,
        # Weight-name substrings to keep trainable; every other weight is
        # frozen. `null` (default) trains everything, as before.
        "freeze_except": None,
        # Dense supervision: predict the next token at every window position
        # (`T` targets per sample instead of 1) with the fused readout loss.
        # RoPE archs only. Validation stays single-target, so val_loss is
        # comparable with every other run.
        "dense_supervision": False,
        # packed archs: validate on packed rows at every target (the same rows each epoch) instead of one sampled
        # target per window; val_loss then averages all assistant / document targets of those rows
        "dense_validation": False,
        # truncated BPTT: row b of each batch continues row b of the previous one (the state is carried, detached)
        "stateful": False,
        # Linear learning-rate warmup over the first N optimizer steps (0: off), and
        # clipping of the global gradient norm (null: off). Both guard the first Adam
        # steps of freshly initialized wide layers.
        "warmup_steps": 0,
        "global_clipnorm": None,
        # Numbered checkpoints (checkpoints/checkpoint_NNNNN.keras): every N epochs (1: each epoch, 0: none). The
        # latest (model.keras, the resume point) and best.keras are always written. A checkpoint holds the weights
        # and the optimizer state (gated-deltanet 56M params: 645 MB each).
        "checkpoint_every": 1,
    },
    "device": "xpu",
    # Fine-tuning: `{"run": <name>, "checkpoint": "latest" | "best" | <epoch>}`
    # initializes a fresh run from another run's weights (e.g. at a longer
    # `window_size`, see `model create --init-from`). `null` = train from scratch.
    "init_from": None,
    # Partial initialization of a fresh run: `{"run": <name>, "checkpoint": "latest" | "best" | <epoch>,
    # "weights": ["token_embedding", "readout"]}` copies those layers' weights (same shapes) from another run,
    # whatever its architecture config; everything else starts from the arch's own init. `null` = off.
    "init_weights_from": None,
}

# Keys allowed in each section. `validate()` rejects anything else, so a typo
# in `model config set` fails loudly instead of being silently ignored.
_ALLOWED = {
    "data": set(DEFAULT_CONFIG["data"]),
    "training": set(DEFAULT_CONFIG["training"]),
}
_TOP_LEVEL = set(DEFAULT_CONFIG)


class ConfigError(ValueError):
    """Raised for a malformed config or an unknown key."""


def run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def config_path(run_name: str) -> Path:
    return run_dir(run_name) / "config.json"


def load(run_name: str) -> Dict[str, Any]:
    """Loads and validates `runs/<run_name>/config.json`."""
    path = config_path(run_name)
    if not path.exists():
        raise ConfigError(
            f"No config at {path}. Create the run first: `model create --run-name {run_name}`"
        )
    with open(path) as f:
        cfg = json.load(f)
    validate(cfg)
    return cfg


def save(run_name: str, cfg: Dict[str, Any]) -> Path:
    """Validates and writes `runs/<run_name>/config.json`."""
    validate(cfg)
    path = config_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return path


def validate(cfg: Dict[str, Any]) -> None:
    """Rejects unknown top-level and section keys.

    `arch_params` is intentionally *not* key-checked here: it is validated by
    the architecture's own constructor signature, which is the only thing that
    actually knows what it accepts.
    """
    unknown_top = set(cfg) - _TOP_LEVEL
    if unknown_top:
        raise ConfigError(
            f"Unknown top-level config key(s): {sorted(unknown_top)}. "
            f"Allowed: {sorted(_TOP_LEVEL)}"
        )
    for section, allowed in _ALLOWED.items():
        unknown = set(cfg.get(section, {})) - allowed
        if unknown:
            raise ConfigError(
                f"Unknown key(s) in '{section}': {sorted(unknown)}. Allowed: {sorted(allowed)}"
            )
    for required in ("arch", "arch_params"):
        if required not in cfg:
            raise ConfigError(f"Config is missing required key '{required}'.")


def get(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Reads a dotted key, e.g. `get(cfg, "training.epochs")`."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Config key not found: {dotted_key!r}")
        node = node[part]
    return node


def set_value(cfg: Dict[str, Any], dotted_key: str, raw_value: str) -> Any:
    """Sets a dotted key in place, parsing the value as JSON.

    JSON parsing is what makes `true`, `null`, `0.001` and `[1, 2]` all work
    from the shell; anything that is not valid JSON is stored as a plain
    string (so `set arch dual-sphere` does the obvious thing).

    The key must already exist -- this is a *setter*, not a way to invent new
    configuration surface by typo.
    """
    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        value = raw_value

    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Config key not found: {dotted_key!r}")
        node = node[part]
    leaf = parts[-1]
    # `arch_params` is arch-defined, so new keys are legitimate there; every
    # other section is fixed and a missing key means a typo.
    if leaf not in node and parts[0] != "arch_params":
        raise ConfigError(
            f"Config key not found: {dotted_key!r}. Existing keys here: {sorted(node)}"
        )
    node[leaf] = value
    validate(cfg)
    return value


# --------------------------------------------------------------------------
# Dataset metadata: the single home for dataset-level facts.
# --------------------------------------------------------------------------

def dataset_dir(name: str) -> Path:
    return DATASETS_DIR / name


def load_dataset_metadata(name: str) -> Dict[str, Any]:
    """Loads `datasets/<name>/metadata.json`."""
    path = dataset_dir(name) / "metadata.json"
    if not path.exists():
        available = sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir()) if DATASETS_DIR.exists() else []
        raise ConfigError(f"No dataset metadata at {path}. Available datasets: {available}")
    with open(path) as f:
        return json.load(f)


def dataset_split_path(name: str, split: str) -> Path:
    """Resolves the `.npy` token file for a split via the dataset metadata."""
    meta = load_dataset_metadata(name)
    if split not in meta:
        raise ConfigError(f"Dataset {name!r} has no split {split!r}. Splits: {sorted(k for k in meta if isinstance(meta[k], dict))}")
    return dataset_dir(name) / meta[split]["file"]


def tokenizer_path(dataset_name: str) -> Path:
    """Resolves the tokenizer named by a dataset's metadata."""
    meta = load_dataset_metadata(dataset_name)
    return TOKENIZERS_DIR / meta["tokenizer"] / "tokenizer.json"
