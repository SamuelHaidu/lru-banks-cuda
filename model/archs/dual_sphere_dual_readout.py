"""Dual-Sphere-Dual-Readout LM: same context->FFN pathway as `dual-sphere`,
but trained with two simultaneous softmax objectives instead of one.

Motivation: `distill-gpt-to-baseline` (KL-distilled `dual-sphere`) showed
`input_embeddings` (dim_in=8) developing a genuinely more separated
structure than plain CE-trained `dual-sphere` ever produced -- most visibly,
character names splitting into distinct gender clusters instead of
collapsing into one "any name" blob. This architecture tests whether that
separation can be reproduced by pure cross-entropy supervision alone (no
teacher, no distillation) by forcing `input_embeddings` to *also* be a
prediction target, not just a context-embedding lookup table.

Forward pass (context path identical to `dual-sphere`):
    1. `input_embeddings` (V, dim_in=8), L2-normalized -> S^(dim_in - 1)
    2. gather `window_size` causal ids, flatten, L2-normalize
    3. FFN: `hidden_dim` GELU -> `hidden_dim // 2` GELU -> `dim_out=128` linear -> `z`
    4. `q_big = L2-normalize(z)` -> cosine vs `target_embeddings` (dim_out=128)
       -> `logits_big = cos_sim * tau_big` (this is what `call()` returns,
       satisfying `BaseLM`'s single-logits contract for generate/evaluate)
    5. `q_small = project(z)`: Dense(64, gelu) -> Dense(16, gelu) ->
       Dense(dim_in=8, linear) -> L2-normalize -> cosine vs
       `input_embeddings` itself (TIED: the same table used in step 1 for
       context lookup is also the small readout's prediction target)
       -> `logits_small = cos_sim * tau_small`
    6. loss = 0.5 * CE(logits_big, target) + 0.5 * CE(logits_small, target)

`input_embeddings` therefore receives gradient from two sources every step:
its role embedding context tokens (via the FFN -> big softmax path, same as
`dual-sphere`) and its role as the small softmax's own prediction target
(directly, via the small-readout CE loss). Whether that second gradient
source is enough to reproduce the gender/category separation seen after
distillation -- without ever involving a teacher model -- is the open
question this architecture exists to test.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


def _ce_from_logits(logits: Any, labels: Any) -> Any:
    """Per-example sparse categorical cross-entropy from logits.

    Identical construction to `model.losses.SparseCategoricalCrossentropyFromLogits`
    (`log_softmax` + `take_along_axis`, not `keras.losses.SparseCategoricalCrossentropy`)
    for the same reason: the "none"-reduction cross-entropy kernel launches
    one work-item per class, which exceeds this project's Intel XPU
    work-group limit (512) at a 10,000-word vocabulary.
    """
    labels = ops.cast(ops.reshape(labels, (-1, 1)), "int32")
    log_probs = ops.log_softmax(logits, axis=-1)
    picked = ops.take_along_axis(log_probs, labels, axis=-1)
    return -ops.squeeze(picked, axis=-1)


@keras.saving.register_keras_serializable(package="model")
class DualSphereDualReadoutLM(BaseLM):
    """`dual-sphere`'s context->FFN pathway, with a second small-table readout
    trained jointly (50/50 CE blend) instead of a single big-table softmax.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context/small-readout manifold S^(dim_in - 1).
        dim_out: Dimensionality of the big-readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens.
        hidden_dim: Width of the first FFN layer; the second is half of it.
        small_proj_dims: Hidden widths of the big->small projection
            (`dim_out` -> `small_proj_dims[0]` GELU -> `small_proj_dims[1]`
            GELU -> `dim_in` linear + L2-normalize).
        initial_tau: Initial cosine-readout temperature, shared as the
            starting value for both `tau_big` and `tau_small`.
        learnable_tau: Whether both taus train.
        embeddings_initializer: Initializer for both tables.
        activation: FFN hidden activation ("gelu" or "tanh").
        loss_weight_big: Weight on the big-table CE term; `1 - loss_weight_big`
            weights the small-table CE term. Default 0.5 (50/50, per spec).
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim_in: int = 8,
        dim_out: int = 128,
        window_size: int = 128,
        hidden_dim: int = 512,
        small_proj_dims: Any = (64, 16),
        initial_tau: float = 9.6158,
        learnable_tau: bool = True,
        embeddings_initializer: Any = "uniform",
        activation: str = "gelu",
        loss_weight_big: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim_in <= 0 or dim_out <= 0:
            raise ValueError(f"`dim_in`/`dim_out` must be positive, got {dim_in}/{dim_out}")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")
        if hidden_dim < 2:
            raise ValueError(f"`hidden_dim` must be >= 2, got {hidden_dim}")
        if activation not in ("gelu", "tanh"):
            raise ValueError(f"`activation` must be 'gelu' or 'tanh', got {activation!r}")
        if len(small_proj_dims) != 2 or any(d < 1 for d in small_proj_dims):
            raise ValueError(f"`small_proj_dims` must be 2 positive ints, got {small_proj_dims}")
        if not 0.0 <= loss_weight_big <= 1.0:
            raise ValueError(f"`loss_weight_big` must be in [0, 1], got {loss_weight_big}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.small_proj_dims = (int(small_proj_dims[0]), int(small_proj_dims[1]))
        self.initial_tau = float(initial_tau)
        self.learnable_tau = bool(learnable_tau)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.activation = str(activation)
        self.loss_weight_big = float(loss_weight_big)

        self.dense1 = layers.Dense(self.hidden_dim, activation=self.activation, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=self.activation, use_bias=False, name="ffn_dense2")
        self.dense3 = layers.Dense(self.dim_out, activation=None, use_bias=False, name="ffn_dense3")

        # Big -> small projection: dim_out -> small_proj_dims[0] (GELU) ->
        # small_proj_dims[1] (GELU) -> dim_in (linear, then L2-normalized in call()).
        self.proj1 = layers.Dense(self.small_proj_dims[0], activation="gelu", use_bias=False, name="proj_dense1")
        self.proj2 = layers.Dense(self.small_proj_dims[1], activation="gelu", use_bias=False, name="proj_dense2")
        self.proj3 = layers.Dense(self.dim_in, activation=None, use_bias=False, name="proj_dense3")

        self.build()

    # -- BaseLM contract ------------------------------------------------------

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    # -- weights --------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.input_embeddings = self.add_weight(
            shape=(self._vocabulary_size, self.dim_in),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="input_emb",
        )
        self.target_embeddings = self.add_weight(
            shape=(self._vocabulary_size, self.dim_out),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="target_emb",
        )
        self.tau_big = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_tau),
            trainable=self.learnable_tau,
            name="tau_big",
        )
        self.tau_small = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_tau),
            trainable=self.learnable_tau,
            name="tau_small",
        )
        self.dense1.build((None, self.window_size * self.dim_in))
        self.dense2.build((None, self.hidden_dim))
        self.dense3.build((None, self.hidden_dim // 2))
        self.proj1.build((None, self.dim_out))
        self.proj2.build((None, self.small_proj_dims[0]))
        self.proj3.build((None, self.small_proj_dims[1]))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        """Input table projected onto S^(dim_in - 1). Doubles as the small
        readout's prediction target (tied weights) -- see module docstring."""
        return ops.normalize(self.input_embeddings, axis=-1, order=2)

    def normalized_target_embeddings(self) -> Any:
        """Target table projected onto S^(dim_out - 1)."""
        return ops.normalize(self.target_embeddings, axis=-1, order=2)

    # -- forward --------------------------------------------------------------

    def _query(self, inputs: Any, training: bool = False) -> Any:
        """Returns `z`, the pre-normalize dim_out=128 FFN output -- shared by
        both readout heads."""
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]

        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        u = ops.reshape(flat, (batch_size, self.window_size * self.dim_in))
        u = ops.normalize(u, axis=-1, order=2)

        h = self.dense1(u, training=training)
        h = self.dense2(h, training=training)
        z = self.dense3(h, training=training)
        return z

    def _big_logits(self, z: Any) -> Any:
        q_big = ops.normalize(z, axis=-1, order=2)
        cos_sim = ops.matmul(q_big, ops.transpose(self.normalized_target_embeddings()))
        return cos_sim * ops.cast(self.tau_big, cos_sim.dtype)

    def _small_logits(self, z: Any, training: bool = False) -> Any:
        p = self.proj1(z, training=training)
        p = self.proj2(p, training=training)
        p = self.proj3(p, training=training)
        q_small = ops.normalize(p, axis=-1, order=2)
        cos_sim = ops.matmul(q_small, ops.transpose(self.normalized_input_embeddings()))
        return cos_sim * ops.cast(self.tau_small, cos_sim.dtype)

    def call(self, inputs: Any, training: bool = False) -> Any:
        """Returns the big-table logits only, satisfying `BaseLM`'s
        `(batch, vocabulary_size)` contract -- this is what `generate()`,
        `evaluate()`, and every other shared consumer sees. The small-table
        readout only participates via `compute_loss` during training."""
        z = self._query(inputs, training=training)
        return self._big_logits(z)

    def compute_loss(self, x: Any = None, y: Any = None, y_pred: Any = None, sample_weight: Any = None, training: bool = True) -> Any:
        """`loss_weight_big * CE(logits_big, y) + (1 - loss_weight_big) * CE(logits_small, y)`.

        Recomputes `z` once and derives both logits from it (rather than
        reusing `y_pred`, which is only the big logits) so both readout
        heads see the same forward pass's `training` flag.
        """
        del sample_weight
        z = self._query(x, training=training)
        logits_big = y_pred if y_pred is not None else self._big_logits(z)
        logits_small = self._small_logits(z, training=training)

        ce_big = ops.mean(_ce_from_logits(logits_big, y))
        ce_small = ops.mean(_ce_from_logits(logits_small, y))
        loss = self.loss_weight_big * ce_big + (1.0 - self.loss_weight_big) * ce_small

        self._loss_tracker.update_state(loss, sample_weight=ops.shape(x)[0])
        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "small_proj_dims": list(self.small_proj_dims),
            "initial_tau": self.initial_tau,
            "learnable_tau": self.learnable_tau,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
            "activation": self.activation,
            "loss_weight_big": self.loss_weight_big,
        }
        return {**super().get_config(), **config}
