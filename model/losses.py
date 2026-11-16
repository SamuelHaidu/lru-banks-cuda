"""The training loss.

`keras.losses.SparseCategoricalCrossentropy` cannot be used here. Keras
computes a *per-sample* loss and reduces it itself, so on the torch backend it
calls `torch.nn.functional.cross_entropy(..., reduction="none")`. That kernel
does not run on Gen12LP (Intel Iris Xe):

    RuntimeError: The number of work-items in each dimension of a work-group
                  cannot exceed {512, 512, 512} for this device

Verified directly: at 10,000 classes, `reduction="mean"` and `reduction="sum"`
both succeed on XPU while `reduction="none"` fails, because the "none" variant
launches one work-item per class and the vocabulary (10,000) exceeds the
device's 512 work-group limit.

`log_softmax` + gather expresses the identical quantity using only reductions
that do fit, is written in `keras.ops` (so it stays backend-agnostic), and is
numerically stable for the same reason `cross_entropy` is -- `log_softmax`
subtracts the row max internally rather than exponentiating raw logits.
"""

from typing import Any

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import ops  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class SparseCategoricalCrossentropyFromLogits(keras.losses.Loss):
    """Sparse categorical cross-entropy over logits, per sample.

    Equivalent to `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`
    but built from `log_softmax` + `take_along_axis` so it runs on Intel XPU.

    Args:
        y_true: `(batch,)` or `(batch, 1)` integer class indices.
        y_pred: `(batch, num_classes)` unnormalized logits.
    """

    def __init__(self, name: str = "sparse_categorical_crossentropy", **kwargs: Any):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        labels = ops.cast(ops.reshape(y_true, (-1, 1)), "int32")
        log_probs = ops.log_softmax(y_pred, axis=-1)
        picked = ops.take_along_axis(log_probs, labels, axis=-1)
        return -ops.squeeze(picked, axis=-1)
