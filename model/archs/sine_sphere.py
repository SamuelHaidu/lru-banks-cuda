"""Sine-Sphere LM: S^(dim_in-1) context -> S^(dim_out-1) readout with sine activations.

Harmonic variant of the dual-sphere architecture. Non-recurrent: a fixed causal
window of `window_size` token ids is looked up in a small spherical input
table, flattened, and compressed by a bias-free 3-layer sine FFN into a query on
a second, wider sphere. Logits are the cosine similarity between that query and
a separate target table, scaled by a learnable temperature.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
def sine(x: Any) -> Any:
    """Sine activation function: sin(x)."""
    return ops.sin(x)


@keras.saving.register_keras_serializable(package="model")
class SineSphereLM(BaseLM):
    """Bias-free FFN with sine activations over a spherical context window, cosine readout.

    Conventions shared with every architecture here:
      - Embedding tables use the ``"uniform"`` initializer
        (``RandomUniform(-0.05, 0.05)``), never ``glorot_uniform``: Glorot
        treats a ``(vocab_size, dim)`` table's fan-in as ``vocab_size``, which
        crushes per-row norm for wide/thin tables.
      - No bias anywhere. Biases give the FFN a content-independent shortcut
        and tend to create attractors in the readout.
      - Spherical projection is ``ops.normalize(x, axis=-1, order=2)`` with
        the Keras default epsilon -- no ad-hoc epsilon overrides.

    Forward pass:
        1. input table ``(V, dim_in)``, L2-normalized -> S^(dim_in - 1)
        2. gather ``window_size`` causal ids, flatten -> R^(window_size*dim_in)
        3. FFN: ``hidden_dim`` sine -> ``hidden_dim // 2`` sine -> ``dim_out`` linear
        4. L2-normalize -> query on S^(dim_out - 1)
        5. target table ``(V, dim_out)``, L2-normalized
        6. ``logits = (q @ target.T) * tau``

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens, left zero-padded near the
            start of a document.
        hidden_dim: Width of the first FFN layer; the second is half of it.
        initial_tau: Initial cosine-readout temperature. Because logits are
            bounded by ``tau``, this also sets the loss floor at
            ``ln(1 + (V-1)*exp(-2*tau))``; the default 9.6158 puts that floor
            far below any achievable loss for V=10k.
        learnable_tau: Whether ``tau`` trains.
        embeddings_initializer: Initializer for both tables.
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

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.initial_tau = float(initial_tau)
        self.learnable_tau = bool(learnable_tau)
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        self.dense1 = layers.Dense(self.hidden_dim, activation=sine, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=sine, use_bias=False, name="ffn_dense2")
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

        # Left-padding filler ids are embedded like any other token and are
        # never masked to zero afterwards. Multiplying by a `token_ids == 0`
        # mask is unsafe: 0 is also a real vocabulary id (`<bos>`), so a
        # minimal-context example whose only valid context token is `<bos>`
        # would have its whole window zeroed. The FFN is bias-free, so
        # sin(0)=0 propagates that exact zero into `ops.normalize`, whose
        # backward pass differentiates `sqrt(sum(x**2))` -- 0/0 produces real
        # NaN gradients regardless of the forward epsilon guard. The padding
        # row simply learns to be a "start of window" sentinel instead.
        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        u = ops.reshape(flat, (batch_size, self.window_size * self.dim_in))
        u = ops.normalize(u, axis=-1, order=2)

        h = self.dense1(u, training=training)
        h = self.dense2(h, training=training)
        z = self.dense3(h, training=training)

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
        }
        return {**super().get_config(), **config}
