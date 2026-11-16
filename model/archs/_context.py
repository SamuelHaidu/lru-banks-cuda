"""Changing a trained model's context window: extrapolation and fine-tuning.

Only architectures whose weights do not depend on the window size (apart from
per-distance vectors) can do this. They opt in by overriding
`BaseLM.with_context_length` with `with_context_length` below.

Per-distance weights (e.g. `distance_weights`, aligned with the context axis
so the last entry is distance 1) are end-aligned: distances both windows
share keep their trained values, new distances start at 0, and distances
beyond a shorter window are dropped.
"""

from typing import Any, Iterable

import numpy as np
from keras import ops

PER_DISTANCE_WEIGHTS = ("distance_weights",)


def _relative(path: str) -> str:
    """Weight path without the model-name prefix, so two models built under
    different names still match."""
    return path.split("/", 1)[1] if "/" in path else path


def transfer_weights(src: Any, dst: Any, per_distance: Iterable[str] = PER_DISTANCE_WEIGHTS) -> None:
    """Copies every weight of `src` into `dst`, resizing per-distance vectors.

    Raises if a weight is missing on either side or has an incompatible shape,
    so an architecture mismatch never transfers silently.
    """
    per_distance = tuple(per_distance)
    source = {_relative(w.path): w for w in src.weights}
    target = {_relative(w.path): w for w in dst.weights}
    if set(source) != set(target):
        raise ValueError(
            "Weight sets differ: only in source "
            f"{sorted(set(source) - set(target))}, only in target {sorted(set(target) - set(source))}"
        )
    for key, w in target.items():
        value = ops.convert_to_numpy(source[key])
        if tuple(value.shape) == tuple(w.shape):
            w.assign(value)
        elif (key.split("/")[-1] in per_distance and value.ndim == len(w.shape) >= 1
              and tuple(value.shape[1:]) == tuple(w.shape[1:])):
            # Distance is axis 0 (e.g. `(T,)` or per-dimension `(T, dim)`).
            n = int(w.shape[0])
            resized = np.zeros((n,) + tuple(value.shape[1:]), dtype=value.dtype)
            keep = min(n, value.shape[0])
            resized[-keep:] = value[-keep:]
            w.assign(resized)
        else:
            raise ValueError(f"Cannot transfer weight {key!r}: shape {value.shape} -> {tuple(w.shape)}")


def with_context_length(model: Any, context_length: int) -> Any:
    """A new model of the same class and config at `context_length`, same weights."""
    config = model.get_config()
    config["window_size"] = int(context_length)
    new = type(model).from_config(config)
    transfer_weights(model, new)
    return new
