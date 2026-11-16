"""Training and resume.

Deliberately thin: `model.fit` with four built-in Keras callbacks (plus one
dataset-conditional callback for equation-level accuracy on modular-arithmetic
datasets, see `_build_equation_metric_callback`) and nothing else. No EMA, no
tokens/sec, no experiment tracker, no in-training sample generation. Those all
previously lived in a 1305-line `Trainer` whose step accounting was the source
of repeated resume bugs.

Two properties make this work:

  * **One epoch == `steps_per_epoch` batches.** "Step" is therefore always
    `epoch * steps_per_epoch`, so resume is just `initial_epoch` -- there is
    no step-offset machinery, no manifest, and nothing to keep in sync.
  * **`.keras` carries the optimizer.** Keras 3 saves optimizer state
    (Adam's `m`/`v` moments) inside the archive, so reloading `model.keras`
    resumes with momentum intact rather than restarting the moments at zero.

Resume state is read from `metrics.csv` -- the file the CSVLogger has to write
anyway. Nothing else records progress, so nothing else can disagree with it.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend, setup_device, describe_device

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.archs import build_model  # noqa: E402
from model.callbacks import ModularArithmeticEquationAccuracy  # noqa: E402
from model.data import build_datasets, gradient_accumulation  # noqa: E402
from model.losses import SparseCategoricalCrossentropyFromLogits  # noqa: E402
from model.schedules import LinearWarmup  # noqa: E402


class _EveryNEpochs(keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that saves after every `every`-th epoch only (epochs counted from 1)."""

    def __init__(self, every: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.every = max(1, int(every))

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        if (epoch + 1) % self.every == 0:
            super().on_epoch_end(epoch, logs)


def learning_rate_from(config: Dict[str, Any]) -> Any:
    """`training.learning_rate`, with its linear warmup schedule when `training.warmup_steps` > 0."""
    t = config["training"]
    lr: Any = t["learning_rate"]
    decay = int(t.get("lr_decay_steps") or 0)
    if int(t.get("warmup_steps") or 0) > 0 or decay:
        lr = LinearWarmup(lr, int(t.get("warmup_steps") or 0), int(t.get("lr_decay_start") or 0), decay,
                          float(t.get("lr_min_ratio", 0.1)))
    return lr


def resolve_optimizer(config: Dict[str, Any]) -> keras.optimizers.Optimizer:
    """Builds the optimizer from `config["training"]`."""
    t = config["training"]
    name = str(t["optimizer"]).lower()
    kwargs: Dict[str, Any] = {"learning_rate": learning_rate_from(config)}
    if t.get("global_clipnorm"):
        kwargs["global_clipnorm"] = float(t["global_clipnorm"])
    if name in ("adamw", "sgd", "adam"):
        if name == "adamw":
            kwargs["weight_decay"] = t["weight_decay"]
        elif t.get("weight_decay"):
            kwargs["weight_decay"] = t["weight_decay"]
    cls = {
        "adam": keras.optimizers.Adam,
        "adamw": keras.optimizers.AdamW,
        "sgd": keras.optimizers.SGD,
    }.get(name)
    if cls is None:
        raise ValueError(f"Unknown optimizer {t['optimizer']!r}. Use one of: adam, adamw, sgd.")
    return cls(**kwargs)


def last_completed_epoch(metrics_path: Path) -> int:
    """Number of epochs already finished, read from `metrics.csv`.

    Keras writes a zero-based `epoch` column, so a file whose final row reads
    `epoch=36` means 37 epochs are done and training resumes at
    `initial_epoch=37`. Returns 0 when the file is missing or empty.
    """
    if not metrics_path.exists():
        return 0
    try:
        with open(metrics_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("epoch") not in (None, "")]
        return int(rows[-1]["epoch"]) + 1 if rows else 0
    except (OSError, ValueError, KeyError):
        return 0


def build_or_load(run_name: str, config: Dict[str, Any], verbose: int = 1) -> Tuple[keras.Model, int]:
    """Returns `(model, initial_epoch)`, resuming automatically when possible."""
    latest = cfg_mod.run_dir(run_name) / "model.keras"
    initial_epoch = last_completed_epoch(cfg_mod.run_dir(run_name) / "metrics.csv")

    freeze_except = config["training"].get("freeze_except")

    if latest.exists() and initial_epoch > 0:
        model = keras.saving.load_model(latest)
        # the saved optimizer keeps its moments and step count, but the learning rate follows config.json (edited
        # between runs, e.g. with a new batch size)
        saved = model.optimizer._learning_rate
        saved_lr = getattr(saved, "learning_rate", None)
        saved_lr = float(saved_lr if saved_lr is not None else keras.ops.convert_to_numpy(saved))
        if saved_lr != float(config["training"]["learning_rate"]):
            if verbose:
                print(f"Learning rate {saved_lr:g} -> {config['training']['learning_rate']:g} (config.json).")
            model.optimizer.learning_rate = learning_rate_from(config)
        _apply_freeze(model, freeze_except, verbose)
        _exclude_from_weight_decay(model, model.optimizer)
        if verbose:
            print(f"Resuming '{run_name}' from epoch {initial_epoch} (optimizer state restored).")
        return model, initial_epoch

    model = build_model(config)
    init_from = config.get("init_from")
    if init_from:
        _init_from_run(model, init_from, verbose)
    if config.get("init_weights_from"):
        _init_weights_from_run(model, config["init_weights_from"], verbose)
    _apply_freeze(model, freeze_except, verbose)
    model.compile(
        optimizer=resolve_optimizer(config),
        loss=SparseCategoricalCrossentropyFromLogits(),
        metrics=["sparse_categorical_accuracy"],
        # `.get` with a default: run configs written before this key existed
        # don't have it, and should keep training exactly as before (eager).
        jit_compile=config["training"].get("jit_compile", False),
    )
    _exclude_from_weight_decay(model, model.optimizer)
    if verbose and not init_from:
        print(f"Initialized a fresh '{config['arch']}' model for run '{run_name}'.")
    return model, 0


def _exclude_from_weight_decay(model: keras.Model, optimizer: Any) -> None:
    """Archs list variables weight decay must not touch in `no_weight_decay_names`
    (e.g. raw sigmoid/softplus parameters, which decay would pull to a fixed value).
    Not serialized with the optimizer, so applied on fresh builds and on resume."""
    names = getattr(model, "no_weight_decay_names", None)
    if names and optimizer is not None and hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(var_names=list(names))


class _StepLossProgbar(keras.callbacks.ProgbarLogger):
    """Progress bar that shows the current step's loss, also on the end-of-epoch line.

    Dense train steps report their batch loss as `loss`; Keras would print the epoch
    average on the final line. metrics.csv still records the average.
    """

    def on_train_batch_end(self, batch, logs=None):
        self._last_loss = None if logs is None else logs.get("loss")
        super().on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        shown = dict(logs or {})
        if getattr(self, "_last_loss", None) is not None:
            shown["loss"] = self._last_loss
        super().on_epoch_end(epoch, shown)


# Config keys allowed to differ between a fine-tuning run and its source.
_INIT_FROM_MAY_DIFFER = {"name", "trainable", "dtype", "window_size"}


def _init_from_run(model: keras.Model, init_from: Dict[str, Any], verbose: int) -> None:
    """Loads `init_from`'s checkpoint into the freshly built `model`.

    The architecture config must match except for `window_size`; per-distance
    weights are resized (see `model.archs._context`). Optimizer state starts
    fresh -- this is fine-tuning, not resuming.
    """
    from model.archs._context import transfer_weights
    from model.evaluate import load_checkpoint

    source_run = init_from["run"]
    checkpoint = init_from.get("checkpoint", "latest")
    source = load_checkpoint(source_run, checkpoint)
    if type(source) is not type(model):
        raise ValueError(f"init_from run {source_run!r} is {type(source).__name__}, this run builds {type(model).__name__}.")
    src_cfg, dst_cfg = source.get_config(), model.get_config()
    differing = sorted(k for k in set(src_cfg) | set(dst_cfg)
                       if k not in _INIT_FROM_MAY_DIFFER and src_cfg.get(k) != dst_cfg.get(k))
    if differing:
        detail = ", ".join(f"{k}: {src_cfg.get(k)!r} -> {dst_cfg.get(k)!r}" for k in differing)
        raise ValueError(f"init_from run {source_run!r} has a different architecture config ({detail}).")
    transfer_weights(source, model)
    if verbose:
        print(f"Initialized from run '{source_run}' (checkpoint {checkpoint}): "
              f"context {source.context_length} -> {model.context_length}, fresh optimizer state.")


def _init_weights_from_run(model: keras.Model, spec: Dict[str, Any], verbose: int) -> None:
    """Copies the weights of the layers named in `spec["weights"]` (e.g. `token_embedding`, `readout`; shell-style
    wildcards work, e.g. `l0_*`) from `spec["run"]`'s checkpoint into the freshly built `model`. Variables are matched
    by their path inside the model (the model's own name stripped) and must have the same shape; the architecture
    config may differ."""
    import fnmatch

    from model.evaluate import load_checkpoint

    names = list(spec.get("weights") or ["token_embedding", "readout"])
    source = load_checkpoint(spec["run"], spec.get("checkpoint", "latest"))
    inner = lambda v: v.path.split("/", 1)[1] if "/" in v.path else v.path
    src = {inner(v): v for v in source.weights}
    copied = []
    for v in model.weights:
        key = inner(v)
        if not any(fnmatch.fnmatchcase(key.split("/", 1)[0], n) for n in names):
            continue
        if key not in src and not any(fnmatch.fnmatchcase(key.split("/", 1)[0], n) for n in names if n == key.split("/", 1)[0]):
            # matched by a wildcard only: weights the source does not have (new parts, e.g. a memory) keep their init
            continue
        if key not in src:
            raise ValueError(f"init_weights_from: run {spec['run']!r} has no weight {key!r}")
        if tuple(src[key].shape) != tuple(v.shape):
            raise ValueError(f"init_weights_from: {key!r} is {tuple(src[key].shape)} in {spec['run']!r}, {tuple(v.shape)} here")
        v.assign(src[key].value)
        copied.append(key)
    missing = [n for n in names if not any(fnmatch.fnmatchcase(k.split("/", 1)[0], n) for k in copied)]
    if missing:
        raise ValueError(f"init_weights_from: no weights named {missing} in this model")
    if verbose:
        print(f"Initialized {', '.join(copied)} from run '{spec['run']}' (checkpoint {spec.get('checkpoint', 'latest')}).")


def _apply_freeze(model: keras.Model, freeze_except: Optional[list], verbose: int) -> None:
    """Freezes every weight whose name isn't listed in `freeze_except`.

    `freeze_except` is a list of exact weight names (e.g. `["tau"]`).
    `None`/empty leaves the model untouched.
    """
    if not freeze_except:
        return
    for w in model.weights:
        w.trainable = w.name in freeze_except
    if verbose:
        trainable_names = [w.name for w in model.weights if w.trainable]
        print(f"Frozen all weights except: {trainable_names}")


def _build_equation_metric_callback(config: Dict[str, Any]) -> Optional[ModularArithmeticEquationAccuracy]:
    """Attaches `ModularArithmeticEquationAccuracy` for modular-arithmetic
    datasets built by `datasets/build_modular_arithmetic.py`, detected via
    the `modulus` key their `metadata.json` carries (absent on every other
    dataset, e.g. `tinystories-10k`) -- so this stays inert for every
    non-arithmetic run without a config flag to remember to set.
    """
    meta = cfg_mod.load_dataset_metadata(config["data"]["dataset"])
    if "modulus" not in meta:
        return None
    ops = {int(token_id): symbol for token_id, symbol in meta["vocab"].items() if symbol in ("+", "-", "*")}
    return ModularArithmeticEquationAccuracy(
        modulus=meta["modulus"],
        ops=ops,
        bos_token_id=meta["bos_token_id"],
        equals_token_id=next(int(tid) for tid, sym in meta["vocab"].items() if sym == "="),
    )


def train(run_name: str, epochs: Optional[int] = None, verbose: int = 1) -> keras.callbacks.History:
    """Trains or resumes a run. Every hyperparameter comes from `config.json`.

    Args:
        run_name: Name of the run under `runs/`.
        epochs: Overrides `training.epochs` for this invocation only; the
            config value is left untouched.
        verbose: Keras progress verbosity (`0` silences it, used by tests).
    """
    config = cfg_mod.load(run_name)
    setup_device(config.get("device"))
    run_path = cfg_mod.run_dir(run_name)
    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, initial_epoch = build_or_load(run_name, config, verbose=verbose)
    if config["training"].get("dense_supervision"):
        # Set on every start (fresh or resumed): the flag is runtime state, not saved.
        if not hasattr(model, "dense_hidden"):
            raise ValueError(f"Arch {config['arch']!r} has no dense path; dense_supervision needs a RoPE arch.")
        model.dense_supervision_enabled = True
        model.dense_validation_enabled = bool(config["training"].get("dense_validation"))
        model.grad_accumulation = gradient_accumulation(config)
        model.stateful_training = bool(config["training"].get("stateful"))
        if model.stateful_training and (model.grad_accumulation > 1 or not hasattr(model, "dense_hidden_valid")):
            raise ValueError("training.stateful needs an arch with `dense_hidden_valid` and no micro-batches.")
        # Keras builds the compiled loss/metrics by tracing the first *training*
        # batch, whose dense (B, T) targets don't fit the single-target outputs.
        # Build them on a dummy single-target batch instead (no data consumed).
        model.evaluate(np.zeros((1, model.context_length), np.int32), np.zeros((1,), np.int32), verbose=0)
    if gradient_accumulation(config) > 1 and not config["training"].get("dense_supervision"):
        raise ValueError("data.micro_batch_size (gradient accumulation) needs training.dense_supervision true.")
    packed = bool(getattr(model, "packed_documents", False))
    if packed and not config["training"].get("dense_supervision"):
        raise ValueError(f"Arch {config['arch']!r} trains on packed documents; set training.dense_supervision true.")
    train_ds, val_ds = build_datasets(config, model.context_length, packed=packed,
                                      static_memory=getattr(model, "static_memory_spec", None))
    if hasattr(train_ds, "_epoch"):  # epoch-dependent samplers (streams, random windows): resume at the next epoch
        train_ds._epoch = initial_epoch

    total_epochs = int(epochs if epochs is not None else config["training"]["epochs"])
    if initial_epoch >= total_epochs:
        if verbose:
            print(f"Run '{run_name}' already completed {initial_epoch}/{total_epochs} epochs. Nothing to do.")
        return keras.callbacks.History()

    params = model.parameter_breakdown()
    if verbose:
        print(
            f"device={describe_device()}  arch={config['arch']}  "
            f"params={params['total']:,} (embeddings {params['embeddings']:,} / mechanism {params['mechanism']:,})\n"
            f"dataset={config['data']['dataset']}  context={model.context_length}  "
            f"batch={config['data']['batch_size']}  steps/epoch={len(train_ds) // gradient_accumulation(config)}"
            f"{f' ({len(train_ds) / train_ds.steps_per_pass:.2f} pass, random windows)' if getattr(train_ds, 'random_windows', False) else ''}  "
            f"{'dense supervision  ' if config['training'].get('dense_supervision') else ''}"
            f"epochs {initial_epoch} -> {total_epochs}"
        )

    callbacks = [
        # Latest: what `train` resumes from and what `evaluate --checkpoint latest` reads.
        keras.callbacks.ModelCheckpoint(filepath=str(run_path / "model.keras")),
        # Immutable per-epoch history (every `training.checkpoint_every` epochs); step == epoch * steps_per_epoch.
        *([_EveryNEpochs(int(config["training"].get("checkpoint_every", 1)),
                         filepath=str(ckpt_dir / "checkpoint_{epoch:05d}.keras"))]
          if int(config["training"].get("checkpoint_every", 1)) > 0 else []),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"), monitor="val_loss", save_best_only=True
        ),
        # Also the resume record -- see `last_completed_epoch`.
        keras.callbacks.CSVLogger(str(run_path / "metrics.csv"), append=True),
    ]
    equation_metric = _build_equation_metric_callback(config)
    if equation_metric is not None:
        # Runs before CSVLogger in the list above only matters for callback
        # ordering within `fit`; Keras calls every `on_epoch_end` hook before
        # any callback reads `logs`, so append order here is irrelevant to
        # what ends up in `metrics.csv`.
        callbacks.append(equation_metric)

    if verbose and config["training"].get("dense_supervision"):
        # Last in the list, so it runs after CSVLogger / History have recorded the epoch average.
        callbacks.append(_StepLossProgbar())

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=verbose,
    )
