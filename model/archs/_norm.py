"""RMSNorm written out in plain ops.

`keras.layers.RMSNormalization` dispatches to torch's fused `rms_norm`, whose
backward fails on the Iris Xe XPU ("tensor does not have a device") above
roughly 10k rows. This layer computes the same thing with elementwise ops.
"""

from typing import Any, Dict

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class RMSNorm(keras.layers.Layer):
    """`x / sqrt(mean(x^2) + eps) * scale` over the last axis, `scale` learned (init 1)."""

    def __init__(self, epsilon: float = 1e-6, **kwargs: Any):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape: Any) -> None:
        self.scale = self.add_weight(name="scale", shape=(int(input_shape[-1]),), initializer="ones")
        self.built = True

    def call(self, x: Any) -> Any:
        ops = keras.ops
        inv = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x * inv * ops.cast(self.scale, x.dtype)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "epsilon": self.epsilon}
