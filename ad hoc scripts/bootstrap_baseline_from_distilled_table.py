"""Ad hoc: bootstraps `baseline-from-distilled-table`'s epoch 1 by
transplanting `distill-gpt-to-baseline`'s trained `input_embeddings`
(dim_in=8 small table) into a freshly-built `dual-sphere` model before its
very first `model.fit()` call, then trains that one epoch through the
project's normal callback set (ModelCheckpoint x3 + CSVLogger).

Why this can't just be `model train`: `model/train.py::build_or_load` only
resumes from `model.keras` when `metrics.csv` already has completed-epoch
rows; a fresh run always calls `build_model(config)` from scratch, so there
is no existing hook to inject a pretrained weight before the first epoch
without editing `train.py` itself. This script instead does exactly what
`train()`'s fresh-model branch does (build, compile, same callbacks), with
one extra step in between: copy the pretrained embedding table in. After
this script writes `metrics.csv`'s first row, `model train --run-name
baseline-from-distilled-table --epochs 10` (unmodified, ordinary CLI) can
resume epochs 2-10 exactly like any other run.

Usage:
    .venv/bin/python "ad hoc scripts/bootstrap_baseline_from_distilled_table.py"
"""

from model.device import setup_backend, setup_device, describe_device

setup_backend()

import keras  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.archs import build_model  # noqa: E402
from model.data import build_datasets  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.losses import SparseCategoricalCrossentropyFromLogits  # noqa: E402
from model.train import resolve_optimizer  # noqa: E402

RUN_NAME = "baseline-from-distilled-table"
SOURCE_RUN = "distill-gpt-to-baseline"
SOURCE_CHECKPOINT = "latest"


def main() -> int:
    import torch

    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))
    run_path = cfg_mod.run_dir(RUN_NAME)
    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_path / "metrics.csv"
    if metrics_path.exists():
        raise RuntimeError(
            f"{metrics_path} already has rows -- this bootstrap step only applies to a "
            "brand-new run with no completed epochs. Use `model train` to continue an "
            "already-bootstrapped run."
        )

    model = build_model(config)

    source_model = load_checkpoint(SOURCE_RUN, SOURCE_CHECKPOINT)
    if not hasattr(source_model, "input_embeddings"):
        raise ValueError(f"{SOURCE_RUN!r} has no `input_embeddings` to transplant.")
    if source_model.input_embeddings.shape != model.input_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: source {source_model.input_embeddings.shape} vs "
            f"target {model.input_embeddings.shape}."
        )

    with torch.no_grad():
        model.input_embeddings.assign(source_model.input_embeddings.value)
    model.input_embeddings.trainable = False
    print(
        f"Transplanted `input_embeddings` ({tuple(model.input_embeddings.shape)}) "
        f"from '{SOURCE_RUN}'@{SOURCE_CHECKPOINT} into a fresh '{config['arch']}' model, "
        f"and FROZE it (trainable={model.input_embeddings.trainable}) -- the test is "
        f"whether the distilled table's structure alone helps, not whether its values "
        f"are merely a good starting point for further gradient descent."
    )

    model.compile(
        optimizer=resolve_optimizer(config),
        loss=SparseCategoricalCrossentropyFromLogits(),
        metrics=["sparse_categorical_accuracy"],
        jit_compile=config["training"].get("jit_compile", False),
    )

    train_ds, val_ds = build_datasets(config, model.context_length)

    params = model.parameter_breakdown()
    print(
        f"device={describe_device()}  arch={config['arch']}  "
        f"params={params['total']:,} (embeddings {params['embeddings']:,} / mechanism {params['mechanism']:,})\n"
        f"dataset={config['data']['dataset']}  context={model.context_length}  "
        f"batch={config['data']['batch_size']}  steps/epoch={config['data']['steps_per_epoch']}  "
        f"bootstrap epoch 0 -> 1 (of {config['training']['epochs']} total; "
        f"resume the rest with `model train --run-name {RUN_NAME}`)"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(run_path / "model.keras")),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_dir / "checkpoint_{epoch:05d}.keras")),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_dir / "best.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(metrics_path), append=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        initial_epoch=0,
        callbacks=callbacks,
        verbose=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
