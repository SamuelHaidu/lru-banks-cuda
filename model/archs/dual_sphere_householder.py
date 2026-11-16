"""Dual-Sphere-Householder LM: `dual-sphere`'s context->FFN pathway, with the
per-token context embeddings passed through a sequence of 8 learned
Householder reflections before flattening into the FFN readout.

Motivation: `dual-sphere`'s small `dim_in=8` context table is used as-is,
straight from the embedding lookup, flattened and fed to the FFN. This
architecture instead applies a composed sequence of 8 Householder
reflections -- H_8 . H_7 . ... . H_1 -- to every one of the window's
`dim_in=8` token embeddings, before flattening. Each reflection's vector is
*generated*, not a fixed learned constant: 8 independent bias-free
`Dense(1024 -> 8)` layers (`window_size * dim_in = 128 * 8 = 1024` for the
default sizes) all read the *same* flattened, L2-normalized context window
and each produce one reflection vector `v_i`. The composed reflection is
then applied identically to every token position in the window (same 8
reflections for every one of the 128 tokens, derived once per example from
the whole window).

A Householder reflection is `H(x) = x - 2*(v.x)/(v.v) * v` -- an orthogonal
(norm-preserving) transform of the vector it's applied to, with respect to
the hyperplane orthogonal to `v`. This is applied here via the implicit
formula (no `(dim_in, dim_in)` matrix is ever materialized), which is both
more numerically direct and considerably cheaper than building and batching
explicit reflection matrices -- `O(batch * window * dim_in)` per reflection
instead of `O(batch * window * dim_in^2)`.

Everything downstream of the reflected, flattened window (the FFN readout,
target table, tau-scaled cosine logits) is identical to `dual-sphere`.
"""

from typing import Any, Dict, List, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


def _householder_reflect(x: Any, v: Any) -> Any:
    """`H(x) = x - 2*(v.x)/(v.v) * v`, broadcast over `x`'s token axis.

    Args:
        x: `(batch, window_size, dim_in)` -- the vectors being reflected.
        v: `(batch, dim_in)` -- one reflection vector per example, shared
            across every token position in `x`.

    Returns:
        `(batch, window_size, dim_in)`, norm-preserving per token.
    """
    v_expanded = ops.expand_dims(v, axis=1)  # (batch, 1, dim_in), broadcasts over window_size
    vv = ops.sum(v * v, axis=-1, keepdims=True)  # (batch, 1)
    vv = ops.expand_dims(vv, axis=1)  # (batch, 1, 1)
    vx = ops.sum(x * v_expanded, axis=-1, keepdims=True)  # (batch, window_size, 1)
    # `vv` can be exactly zero only if a reflection-vector Dense layer's
    # output collapses to the zero vector for some example -- vanishingly
    # unlikely with real weights, but guarded the same way `ops.normalize`
    # guards division by a near-zero norm elsewhere in this project.
    return x - 2.0 * vx * v_expanded / ops.maximum(vv, keras.config.epsilon())


@keras.saving.register_keras_serializable(package="model")
class DualSphereHouseholderLM(BaseLM):
    """`dual-sphere`'s FFN readout, preceded by 8 learned Householder
    reflections applied to every context token.

    Forward pass:
        1. `input_embeddings` (V, dim_in), L2-normalized -> S^(dim_in - 1)
        2. gather `window_size` causal ids -> `(B, window_size, dim_in)`
        3. flatten, L2-normalize -> `(B, window_size * dim_in)` (the shared
           input to all 8 reflection-vector generators)
        4. 8x bias-free `Dense(window_size * dim_in -> dim_in)` -> 8
           reflection vectors `v_1..v_8`, each `(B, dim_in)`
        5. apply `H_8 . H_7 . ... . H_1` to every token in step 2's
           `(B, window_size, dim_in)` tensor (same 8 reflections reused for
           every position)
        6. flatten the reflected tokens -> `(B, window_size * dim_in)`,
           L2-normalize
        7. FFN: `hidden_dim` GELU -> `hidden_dim // 2` GELU -> `dim_out` linear
        8. L2-normalize -> query on S^(dim_out - 1)
        9. target table `(V, dim_out)`, L2-normalized
        10. `logits = (q @ target.T) * tau`

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1), and
            of each Householder reflection.
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens.
        num_reflections: Number of composed Householder reflections (8, per spec).
        hidden_dim: Width of the first FFN layer; the second is half of it.
        initial_tau: Initial cosine-readout temperature.
        learnable_tau: Whether `tau` trains.
        embeddings_initializer: Initializer for both tables.
        activation: FFN hidden activation ("gelu" or "tanh").
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim_in: int = 8,
        dim_out: int = 128,
        window_size: int = 128,
        num_reflections: int = 8,
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
        if num_reflections <= 0:
            raise ValueError(f"`num_reflections` must be positive, got {num_reflections}")
        if hidden_dim < 2:
            raise ValueError(f"`hidden_dim` must be >= 2, got {hidden_dim}")
        if activation not in ("gelu", "tanh"):
            raise ValueError(f"`activation` must be 'gelu' or 'tanh', got {activation!r}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.num_reflections = int(num_reflections)
        self.hidden_dim = int(hidden_dim)
        self.initial_tau = float(initial_tau)
        self.learnable_tau = bool(learnable_tau)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.activation = str(activation)

        # 8 independent bias-free Dense(window_size*dim_in -> dim_in) layers,
        # one reflection vector generator per Householder reflection -- all
        # reading the same flattened, L2-normalized context window.
        self.reflection_generators: List[layers.Dense] = [
            layers.Dense(self.dim_in, activation=None, use_bias=False, name=f"reflection_gen_{i}")
            for i in range(self.num_reflections)
        ]

        self.dense1 = layers.Dense(self.hidden_dim, activation=self.activation, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=self.activation, use_bias=False, name="ffn_dense2")
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
        flat_dim = self.window_size * self.dim_in
        for gen in self.reflection_generators:
            gen.build((None, flat_dim))
        self.dense1.build((None, flat_dim))
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

        # Same left-padding convention as `dual-sphere`: padding ids are
        # embedded like any other token, never masked to zero (see that
        # module's docstring for why zeroing is unsafe here).
        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        tokens = ops.reshape(flat, (batch_size, self.window_size, self.dim_in))

        flat_window = ops.reshape(tokens, (batch_size, self.window_size * self.dim_in))
        flat_window = ops.normalize(flat_window, axis=-1, order=2)

        reflected = tokens
        for gen in self.reflection_generators:
            v = gen(flat_window, training=training)
            reflected = _householder_reflect(reflected, v)

        u = ops.reshape(reflected, (batch_size, self.window_size * self.dim_in))
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
            "num_reflections": self.num_reflections,
            "hidden_dim": self.hidden_dim,
            "initial_tau": self.initial_tau,
            "learnable_tau": self.learnable_tau,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
            "activation": self.activation,
        }
        return {**super().get_config(), **config}
