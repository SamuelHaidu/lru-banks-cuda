"""Dual-Sphere-Mid-Gate LM: `dual-sphere` plus one RNN-style gate on the FFN.

Identical to `dual-sphere` except for one addition between the first and
second FFN layers: a sigmoid gate, computed from `dense1`'s output in
parallel with `dense2`, that elementwise-multiplies `dense2`'s output before
`dense3`. This is the same shape of gate an LSTM/GRU cell uses on its hidden
state, just applied once in the middle of an otherwise feedforward network
rather than across recurrent time steps -- there is no actual recurrence
here, `dense2` still reads `dense1`'s raw (ungated) output directly.

Forward pass:
    1. input table ``(V, dim_in)``, L2-normalized -> S^(dim_in - 1)
    2. gather ``window_size`` causal ids, flatten -> R^(window_size*dim_in)
    3. ``h1 = dense1(u)``                              -- ``hidden_dim`` GELU
    4. ``h2 = dense2(h1)``                              -- ``hidden_dim // 2`` GELU
    5. ``gate = sigmoid(gate_dense(h1))``               -- ``hidden_dim // 2`` sigmoid, reads ``h1`` not ``h2``
    6. ``h2_gated = h2 * gate``                         -- elementwise
    7. ``z = dense3(h2_gated)``                          -- ``dim_out`` linear
    8. L2-normalize -> query on S^(dim_out - 1)
    9. target table ``(V, dim_out)``, L2-normalized
    10. ``logits = (q @ target.T) * tau``
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class DualSphereMidGateLM(BaseLM):
    """`dual-sphere` with a sigmoid gate between `dense1` and `dense2`'s output.

    Same conventions as `dual-sphere`: uniform embedding initializer,
    bias-free layers throughout (including the gate -- see class docstring in
    `dual_sphere.py` for the rationale), spherical projection via
    `ops.normalize(x, axis=-1, order=2)`.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens, left zero-padded near the
            start of a document.
        hidden_dim: Width of the first FFN layer; the second (and the gate)
            are half of it.
        initial_tau: Initial cosine-readout temperature. Because logits are
            bounded by ``tau``, this also sets the loss floor at
            ``ln(1 + (V-1)*exp(-2*tau))``; the default 9.6158 puts that floor
            far below any achievable loss for V=10k.
        learnable_tau: Whether ``tau`` trains.
        embeddings_initializer: Initializer for both tables.
        activation: FFN hidden activation ("gelu" or "tanh"). The gate is
            always sigmoid, independent of this setting.
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim_in: int = 8,
        dim_out: int = 128,
        window_size: int = 128,
        hidden_dim: int = 512,
        initial_tau: float = 9.6158,
        learnable_tau: bool = True,
        embeddings_initializer: Any = "uniform",
        activation: str = "gelu",
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
        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.initial_tau = float(initial_tau)
        self.learnable_tau = bool(learnable_tau)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.activation = str(activation)

        self.dense1 = layers.Dense(self.hidden_dim, activation=self.activation, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=self.activation, use_bias=False, name="ffn_dense2")
        self.gate_dense = layers.Dense(self.hidden_dim // 2, activation="sigmoid", use_bias=False, name="ffn_gate")
        self.dense3 = layers.Dense(self.dim_out, activation=None, use_bias=False, name="ffn_dense3")

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
        self.tau = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_tau),
            trainable=self.learnable_tau,
            name="tau",
        )
        self.dense1.build((None, self.window_size * self.dim_in))
        self.dense2.build((None, self.hidden_dim))
        self.gate_dense.build((None, self.hidden_dim))
        self.dense3.build((None, self.hidden_dim // 2))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        """Input table projected onto S^(dim_in - 1)."""
        return ops.normalize(self.input_embeddings, axis=-1, order=2)

    def normalized_target_embeddings(self) -> Any:
        """Target table projected onto S^(dim_out - 1)."""
        return ops.normalize(self.target_embeddings, axis=-1, order=2)

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]

        # See `dual_sphere.py`'s `call()` for why left-padding filler ids are
        # never masked to zero -- identical reasoning applies unchanged here.
        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        u = ops.reshape(flat, (batch_size, self.window_size * self.dim_in))
        u = ops.normalize(u, axis=-1, order=2)

        h1 = self.dense1(u, training=training)
        h2 = self.dense2(h1, training=training)
        gate = self.gate_dense(h1, training=training)
        h2_gated = h2 * gate
        z = self.dense3(h2_gated, training=training)

        q = ops.normalize(z, axis=-1, order=2)
        cos_sim = ops.matmul(q, ops.transpose(self.normalized_target_embeddings()))
        return cos_sim * ops.cast(self.tau, cos_sim.dtype)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "initial_tau": self.initial_tau,
            "learnable_tau": self.learnable_tau,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
            "activation": self.activation,
        }
        return {**super().get_config(), **config}
