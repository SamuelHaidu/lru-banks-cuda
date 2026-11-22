"""Dual-Sphere-Linear-Recurrence LM: `dual-sphere`'s context->FFN pathway,
with the 128 mini tokens passed through a context-conditioned linear
recurrence before flattening into the FFN readout, computed via
`keras.ops.associative_scan` -- Keras's own parallel-prefix-scan primitive
(Blelloch 1990), not a hand-rolled implementation. Not a redundant
shared-rotation trick either (see `dual_sphere_householder.py`'s docstring
for why applying the *same* transform to every token changes nothing about
their pairwise geometry).

Recurrence: `h_t = A @ h_{t-1} + B @ x_t`, where `x_t` is mini token `t`'s
embedding (`dim_in=8`) and `A`, `B` (each `dim_in x dim_in`) are *shared
across every position* but *generated per example* from the flattened
context window -- one `Dense(window_size*dim_in -> 2*dim_in^2)` layer
("1024x128" for the default sizes: `128*8=1024` in, `2*8*8=128` out),
reshaped to `(2, dim_in, dim_in)` and split into `A`/`B`.

Composing affine maps is associative by construction, independent of what
`A`/`B` actually are:

    combine((A_i, b_i), (A_j, b_j)) = (A_j @ A_i, A_j @ b_i + b_j)

-- so `keras.ops.associative_scan(combine, (a_seq, b_seq), axis=1)` computes
all `window_size` prefix states `h_1..h_window_size` (each a causal summary
of every mini token up to and including that position) in
`O(log window_size)` parallel combine stages, not `window_size` sequential
steps -- exactly the primitive this needs, supplied by Keras itself.

All `window_size` hidden states are flattened, L2-normalized, and fed to the
same FFN readout as `dual-sphere` -- every downstream table, tau, and logits
computation is identical.
"""

from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


def _combine(left: Tuple[Any, Any], right: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """`combine((A_i, b_i), (A_j, b_j)) = (A_j @ A_i, A_j @ b_i + b_j)` --
    composition of the affine maps `h -> A@h + b`, associative by
    construction. Matches the elementwise signature `keras.ops.associative_scan`
    expects: `left`/`right` are `(A, b)` pairs sharing a scan-axis dimension.
    """
    a_i, b_i = left
    a_j, b_j = right
    a_combined = ops.einsum("...ij,...jk->...ik", a_j, a_i)
    b_combined = ops.einsum("...ij,...j->...i", a_j, b_i) + b_j
    return a_combined, b_combined


@keras.saving.register_keras_serializable(package="model")
class DualSphereLinearRecurrenceLM(BaseLM):
    """`dual-sphere`'s FFN readout, preceded by a context-conditioned linear
    recurrence over the window's mini tokens, computed via a parallel
    associative scan.

    Forward pass:
        1. `input_embeddings` (V, dim_in), L2-normalized -> S^(dim_in - 1)
        2. gather `window_size` causal ids -> `(B, window_size, dim_in)`
        3. flatten, L2-normalize -> `(B, window_size * dim_in)`
        4. one bias-free `Dense(window_size * dim_in -> 2 * dim_in^2)` ->
           reshape to `(B, 2, dim_in, dim_in)` -> split into shared `A`, `B_mat`
           (`(B, dim_in, dim_in)` each)
        5. `b_t = B_mat @ x_t` for every token (one batched matmul)
        6. parallel associative scan of `(A, b_t)` (`A` broadcast to every
           position) -> all `window_size` hidden states `h_1..h_window_size`
        7. flatten `(B, window_size, dim_in)` -> `(B, window_size * dim_in)`,
           L2-normalize
        8. FFN: `hidden_dim` GELU -> `hidden_dim // 2` GELU -> `dim_out` linear
        9. L2-normalize -> query on S^(dim_out - 1)
        10. target table `(V, dim_out)`, L2-normalized
        11. `logits = (q @ target.T) * tau`

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1), and
            of the recurrence's hidden state / `A`,`B` matrices.
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens.
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

        # Generates the shared (per-example, not per-position) recurrence
        # kernel: window_size*dim_in -> 2*dim_in^2, reshaped/split into A, B.
        self.kernel_gen = layers.Dense(
            2 * self.dim_in * self.dim_in, activation=None, use_bias=False, name="kernel_gen"
        )

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
        self.kernel_gen.build((None, flat_dim))
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

        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        tokens = ops.reshape(flat, (batch_size, self.window_size, self.dim_in))

        flat_window = ops.reshape(tokens, (batch_size, self.window_size * self.dim_in))
        flat_window = ops.normalize(flat_window, axis=-1, order=2)

        kernel_flat = self.kernel_gen(flat_window, training=training)  # (B, 2*dim_in^2)
        kernel = ops.reshape(kernel_flat, (batch_size, 2, self.dim_in, self.dim_in))
        a_shared = kernel[:, 0, :, :]  # (B, dim_in, dim_in)
        b_mat = kernel[:, 1, :, :]  # (B, dim_in, dim_in)

        b_terms = ops.einsum("bij,bwj->bwi", b_mat, tokens)  # (B, window_size, dim_in)
        a_broadcast = ops.tile(a_shared[:, None, :, :], (1, self.window_size, 1, 1))  # (B, window_size, dim_in, dim_in)

        _, hidden_states = ops.associative_scan(_combine, (a_broadcast, b_terms), axis=1)  # (B, window_size, dim_in)

        u = ops.reshape(hidden_states, (batch_size, self.window_size * self.dim_in))
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
            "activation": self.activation,
        }
        return {**super().get_config(), **config}
