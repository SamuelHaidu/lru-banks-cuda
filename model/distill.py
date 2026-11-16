"""Knowledge distillation: a frozen teacher, a trainable student, KL loss.

A new, self-contained training procedure -- not a change to `model/train.py`
or `model/interfaces.py`. `BaseLM.call()` already returns `(batch,
vocabulary_size)` logits for every architecture, which is all distillation
needs: any two architectures that share `vocabulary_size` and
`context_length` can be scored on the same input batch, teacher and student
alike.

Distillation runs live under `runs/<name>/` like every other run (so
`runs/<name>/model.keras`, `checkpoints/`, `metrics.csv` all mean the same
thing they do for `model train`), but they are configured by their own
`distill_config.json`, not `config.json` -- `model/config.py`'s schema
(`_ALLOWED`, `validate()`) is a fixed, hand-maintained contract for ordinary
runs and is deliberately not touched here.

Training goes through `model.fit()`, identically to `model/train.py`: same
callbacks (`ModelCheckpoint` x3, `CSVLogger`), same resume convention (one
epoch == `steps_per_epoch` batches, `initial_epoch` read from
`metrics.csv`), same progress bar and `sparse_categorical_accuracy` metric.
The only thing that differs from an ordinary run is what `compute_loss`
computes: a `DistillationModel` wraps a frozen teacher and a trainable
student, `call(x)` returns the student's logits (so metrics/accuracy read
exactly like any other architecture's), and `compute_loss` additionally runs
the teacher on the same `x` to blend in the soft-target KL term.

Loss (Hinton et al., 2015): for each example,

    loss = alpha * CE(student_logits, hard_target)
         + (1 - alpha) * T^2 * KL(softmax(teacher_logits / T) || softmax(student_logits / T))

`T^2` rescales the soft-target gradient back to the same order of magnitude
as the hard-target gradient (the standard correction, since dividing logits
by `T` shrinks the softmax gradient by `1/T^2`). `alpha=0.0` is pure
distillation (no hard-label supervision at all); this project's dataset only
ever has one hard label per window anyway, matching `BaseLM`'s single-target
contract, so the CE term (when `alpha > 0`) is computed over the exact same
`(batch,)` target tensor the dataset already provides -- nothing about the
dataset changes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend, setup_device, describe_device

setup_backend()

import keras  # noqa: E402
from keras import ops  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401  (registers every architecture's serializable class)
from model.archs import build_model  # noqa: E402
from model.data import build_datasets  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.train import resolve_optimizer, last_completed_epoch  # noqa: E402


def distill_config_path(run_name: str) -> Path:
    return cfg_mod.run_dir(run_name) / "distill_config.json"


DEFAULT_DISTILL_CONFIG: Dict[str, Any] = {
    "run_name": None,
    "teacher_run": None,
    "teacher_checkpoint": "best",
    "student_arch": "dual-sphere",
    "student_arch_params": {},
    "data": {
        "dataset": "tinystories-10k",
        "batch_size": 512,
        "steps_per_epoch": 1000,
        "corrupt_prob": 0.15,
        "seed": 42,
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": 10,
        "validation_steps": 50,
        "temperature": 2.0,
        # Weight on the hard-label CE term; (1 - alpha) weights the KL term.
        # 0.0 is pure distillation.
        "alpha": 0.0,
    },
    "device": "xpu",
}


def load_distill_config(run_name: str) -> Dict[str, Any]:
    path = distill_config_path(run_name)
    if not path.exists():
        raise cfg_mod.ConfigError(
            f"No distill config at {path}. Create it with `create_distill_run` first."
        )
    with open(path) as f:
        return json.load(f)


def save_distill_config(run_name: str, cfg: Dict[str, Any]) -> Path:
    path = distill_config_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return path


def create_distill_run(
    run_name: str,
    teacher_run: str,
    student_arch: str,
    dataset: Optional[str] = None,
    teacher_checkpoint: str = "best",
    force: bool = False,
) -> Dict[str, Any]:
    """Writes `runs/<run_name>/distill_config.json`, mirroring `model create`.

    `dataset` defaults to the teacher's own dataset, so teacher and student
    train on the same corpus unless told otherwise. The student's
    `arch_params` defaults come from the architecture's own signature, same
    as `model create` -- `vocabulary_size` is pinned from the dataset
    metadata so it cannot drift from the corpus.
    """
    from model.archs import arch_defaults, ARCHS

    if student_arch not in ARCHS:
        raise cfg_mod.ConfigError(f"Unknown student arch {student_arch!r}. Available: {sorted(ARCHS)}")

    path = distill_config_path(run_name)
    if path.exists() and not force:
        raise cfg_mod.ConfigError(f"Distill run '{run_name}' already exists at {path}. Pass force=True to overwrite.")

    teacher_config = cfg_mod.load(teacher_run)
    dataset_name = dataset or teacher_config["data"]["dataset"]
    meta = cfg_mod.load_dataset_metadata(dataset_name)

    cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULT_DISTILL_CONFIG))
    cfg["run_name"] = run_name
    cfg["teacher_run"] = teacher_run
    cfg["teacher_checkpoint"] = teacher_checkpoint
    cfg["student_arch"] = student_arch
    cfg["data"]["dataset"] = dataset_name

    params = arch_defaults(student_arch)
    params.pop("embeddings_initializer", None)
    params["vocabulary_size"] = meta["vocabulary_size"]
    cfg["student_arch_params"] = params

    save_distill_config(run_name, cfg)

    # Also write an ordinary `config.json` describing the student alone
    # (same arch/arch_params/data/training/device shape `model create`
    # writes), purely so `model evaluate`/`model generate` -- which read
    # `config.json` and load a plain architecture checkpoint -- work
    # unmodified against the distilled student. The student's own
    # `runs/<name>/model.keras` is a plain architecture checkpoint (see
    # `build_or_load_distillation_model`), so nothing about those commands
    # needs to know distillation was involved.
    student_config: Dict[str, Any] = json.loads(json.dumps(cfg_mod.DEFAULT_CONFIG))
    student_config["run_name"] = run_name
    student_config["arch"] = student_arch
    student_config["arch_params"] = cfg["student_arch_params"]
    student_config["data"] = cfg["data"]
    student_config["device"] = cfg["device"]
    cfg_mod.save(run_name, student_config)

    return cfg


class _StudentCheckpoint(keras.callbacks.Callback):
    """Saves `self.model.student` -- `keras.callbacks.ModelCheckpoint` always
    saves the model it is attached to (the `DistillationModel` wrapper),
    which cannot be reloaded generically (see
    `build_or_load_distillation_model`'s docstring). Same
    filepath/monitor/save_best_only semantics as `ModelCheckpoint`, applied
    to the student submodel instead.
    """

    def __init__(self, filepath: str, monitor: Optional[str] = None, save_best_only: bool = False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float("inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        path = self.filepath.format(epoch=epoch + 1)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None or current >= self.best:
                return
            self.best = current
        self.model.student.save(path)


@keras.saving.register_keras_serializable(package="model")
class DistillationModel(keras.Model):
    """Wraps a frozen teacher and a trainable student for `model.fit()`.

    `call(x)` returns the *student's* logits -- so every metric compiled in
    (`sparse_categorical_accuracy`, the progress bar, `ModelCheckpoint`,
    `CSVLogger`) reads and behaves exactly as it does for an ordinary
    `model/train.py` run. `compute_loss` is the only override: it additionally
    runs the (non-trainable) teacher on the same batch to build the
    soft-target KL loss, following Keras's own documented pattern for custom
    loss computation (`keras.Model.compute_loss`'s docstring example).
    """

    def __init__(self, teacher: keras.Model, student: keras.Model, temperature: float = 2.0, alpha: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.teacher.trainable = False
        self.student = student
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    @property
    def context_length(self) -> int:
        return self.student.context_length

    @property
    def vocabulary_size(self) -> int:
        return self.student.vocabulary_size

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.student(inputs, training=training)

    def compute_loss(self, x: Any = None, y: Any = None, y_pred: Any = None, sample_weight: Any = None, training: bool = True) -> Any:
        del sample_weight
        teacher_logits = self.teacher(x, training=False)
        student_logits = y_pred if y_pred is not None else self.student(x, training=training)

        t = self.temperature
        teacher_log_probs = ops.log_softmax(teacher_logits / t, axis=-1)
        student_log_probs = ops.log_softmax(student_logits / t, axis=-1)
        teacher_probs = ops.exp(teacher_log_probs)

        # KL(teacher || student) = sum(p_teacher * (log p_teacher - log p_student))
        kl_per_example = ops.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
        kl_loss = ops.mean(kl_per_example) * (t ** 2)

        if self.alpha > 0.0:
            labels = ops.cast(ops.reshape(y, (-1, 1)), "int32")
            student_log_probs_t1 = ops.log_softmax(student_logits, axis=-1)
            picked = ops.take_along_axis(student_log_probs_t1, labels, axis=-1)
            ce_loss = ops.mean(-ops.squeeze(picked, axis=-1))
            loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss
        else:
            loss = kl_loss

        self._loss_tracker.update_state(loss, sample_weight=ops.shape(x)[0])
        return loss

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "temperature": self.temperature, "alpha": self.alpha}


def build_or_load_distillation_model(run_name: str, cfg: Dict[str, Any], verbose: int = 1) -> Tuple[DistillationModel, int]:
    """Mirrors `model/train.py::build_or_load` for a distillation run.

    `runs/<name>/model.keras` persists the *student* only (via
    `DistillationModel.call` returning the student's logits, `save()` still
    saves the whole wrapper's variables, but reloading through the wrapper
    hits Keras's generic model deserialization, which cannot revive nested
    sub-models from `get_config()` without a custom `from_config`). Simpler
    and just as correct: save/reload the student submodel's own `.keras`
    file directly, and rebuild a fresh `DistillationModel` wrapper (with the
    teacher reloaded from its own run) every time -- the teacher is frozen
    and deterministic, so nothing is lost by not persisting it inside this
    run's checkpoints.
    """
    run_path = cfg_mod.run_dir(run_name)
    latest = run_path / "model.keras"
    initial_epoch = last_completed_epoch(run_path / "metrics.csv")

    teacher = load_checkpoint(cfg["teacher_run"], cfg["teacher_checkpoint"])

    if latest.exists() and initial_epoch > 0:
        student = keras.saving.load_model(latest)
        if verbose:
            print(
                f"Resuming distill run '{run_name}' from epoch {initial_epoch} "
                "(student weights restored; optimizer moments restart fresh -- only the "
                "student submodel is persisted, see build_or_load_distillation_model)."
            )
    else:
        student = build_model({"arch": cfg["student_arch"], "arch_params": cfg["student_arch_params"]})
        initial_epoch = 0
        if verbose:
            print(f"Initialized a fresh distillation run '{run_name}' (teacher={cfg['teacher_run']}, student={cfg['student_arch']}).")

    if student.context_length != teacher.context_length or student.vocabulary_size != teacher.vocabulary_size:
        raise ValueError(
            f"Teacher/student shape mismatch: teacher context={teacher.context_length} "
            f"vocab={teacher.vocabulary_size}, student context={student.context_length} "
            f"vocab={student.vocabulary_size}. Distillation requires both to match."
        )

    model = DistillationModel(
        teacher=teacher, student=student,
        temperature=cfg["training"]["temperature"], alpha=cfg["training"]["alpha"],
    )
    model.compile(
        optimizer=resolve_optimizer({"training": cfg["training"]}),
        metrics=["sparse_categorical_accuracy"],
        jit_compile=cfg["training"].get("jit_compile", False),
    )
    return model, initial_epoch


def distill(run_name: str, epochs: Optional[int] = None, verbose: int = 1) -> keras.callbacks.History:
    """Trains or resumes a distillation run, identically to `model/train.py::train`."""
    cfg = load_distill_config(run_name)
    setup_device(cfg.get("device"))
    run_path = cfg_mod.run_dir(run_name)
    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, initial_epoch = build_or_load_distillation_model(run_name, cfg, verbose=verbose)
    train_ds, val_ds = build_datasets({"data": cfg["data"], "training": cfg["training"]}, model.context_length)

    total_epochs = int(epochs if epochs is not None else cfg["training"]["epochs"])
    if initial_epoch >= total_epochs:
        if verbose:
            print(f"Distill run '{run_name}' already completed {initial_epoch}/{total_epochs} epochs. Nothing to do.")
        return keras.callbacks.History()

    if verbose:
        student_params = model.student.parameter_breakdown()
        print(
            f"device={describe_device()}  teacher={cfg['teacher_run']}@{cfg['teacher_checkpoint']}  "
            f"student_arch={cfg['student_arch']}  student_params={student_params['total']:,}  "
            f"temperature={cfg['training']['temperature']}  alpha={cfg['training']['alpha']}\n"
            f"dataset={cfg['data']['dataset']}  context={model.context_length}  "
            f"batch={cfg['data']['batch_size']}  steps/epoch={cfg['data']['steps_per_epoch']}  "
            f"epochs {initial_epoch} -> {total_epochs}"
        )

    callbacks = [
        _StudentCheckpoint(filepath=str(run_path / "model.keras")),
        _StudentCheckpoint(filepath=str(ckpt_dir / "checkpoint_{epoch:05d}.keras")),
        _StudentCheckpoint(filepath=str(ckpt_dir / "best.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_path / "metrics.csv"), append=True),
    ]

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=verbose,
    )
