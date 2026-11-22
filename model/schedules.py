"""Learning-rate schedules saved inside checkpoints.

Kept out of `model.train` so that loading a checkpoint (evaluate, generate,
scripts) can find them: `model.archs` imports this module, and every loader
imports `model.archs`.
"""

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate rising linearly from ~0 to `learning_rate` over `warmup_steps`, then constant; with
    `decay_steps` > 0 (warmup-stable-decay), from update `decay_start` it falls linearly to `min_ratio` x
    `learning_rate` over `decay_steps` updates and stays there."""

    def __init__(self, learning_rate: float, warmup_steps: int, decay_start: int = 0, decay_steps: int = 0,
                 min_ratio: float = 0.1):
        self.learning_rate = float(learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.decay_start, self.decay_steps, self.min_ratio = int(decay_start), int(decay_steps), float(min_ratio)

    def __call__(self, step):
        ops = keras.ops
        s = ops.cast(step, "float32")
        frac = ops.minimum((s + 1.0) / float(max(self.warmup_steps, 1)), 1.0)
        if self.decay_steps > 0:
            done = ops.clip((s - float(self.decay_start)) / float(self.decay_steps), 0.0, 1.0)
            frac = frac * (1.0 - (1.0 - self.min_ratio) * done)
        return self.learning_rate * frac

    def get_config(self):
        return {"learning_rate": self.learning_rate, "warmup_steps": self.warmup_steps,
                "decay_start": self.decay_start, "decay_steps": self.decay_steps, "min_ratio": self.min_ratio}
