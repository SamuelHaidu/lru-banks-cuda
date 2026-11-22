"""Dual-Sphere-Householder-Rank8 LM: `dual-sphere-householder`'s 8 shared,
context-derived Householder reflections, each further transformed per token
by a full-rank 8x8 matrix generated from a single shared projector -- a
K/V-style projection (one set of weights, applied identically to every
token's own embedding), not a per-token lookup table and not the earlier
per-token-FFN variant's 3-layer transform.

`dual-sphere-householder-per-token` (an earlier, working variant) fixed the
plain `dual-sphere-householder` architecture's geometric redundancy -- a
reflection shared identically across every token in the window is a rigid
rotation of the whole window and provably changes nothing about pairwise
token-to-token relationships -- but its fix (a 3-layer shared FFN, called 8
sequential times, once per reflection stage) turned out to be dominated by
XPU kernel-launch overhead: measured at ~235ms/step, meaningfully slower
than a 2-block ALiBi Transformer's ~170ms/step despite far less real
computation.

This variant achieves the same goal (per-token variation in what would
otherwise be a shared, redundant reflection) with a cheaper mechanism:

    1. `M_token = reshape(shared_projector(token), (dim_in, dim_in))`
       -- one shared, bias-free `Dense(dim_in -> dim_in^2)` layer (512
       weights for the default `dim_in=8`, `8*64=512`), applied to every
       token's own embedding via one batched matmul (no per-token Python
       loop, no per-stage FFN) -- exactly like `W_k`/`W_v` in attention are
       one set of shared weights applied identically to every token.
    2. `H_shared` -- the ordinary Householder matrix for stage `i`, built
       *explicitly* (unlike the other two variants, which use the implicit
       reflection formula) from the shared, context-derived reflection
       vector `v_i` (from the same `Dense(window_size*dim_in -> dim_in)`
       generators `dual-sphere-householder` uses).
    3. `H_token = M_token @ H_shared` -- the shared reflection matrix,
       adjusted per token by that token's own full-rank `M_token`.
    4. `token' = H_token @ token`.

All four steps are batched `einsum`s over `(batch, window_size, ...)`
tensors -- no Python-level loop over token positions, and far fewer, larger
kernel dispatches per stage than the per-token-FFN variant's 3 sequential
small Dense calls. The 8 reflection stages remain sequential (stage `i+1`'s
`M_token`/`H_token` are computed on stage `i`'s already-reflected tokens),
since that dependency is inherent to composing reflections in sequence.
"""

from typing import Any, Dict, List, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class DualSphereHouseholderRank8LM(BaseLM):
    """`dual-sphere-householder`'s 8 shared reflections, each adjusted per
    token by a full-rank `dim_in x dim_in` matrix from one shared projector.

    Forward pass:
        1. `input_embeddings` (V, dim_in), L2-normalized -> S^(dim_in - 1)
        2. gather `window_size` causal ids -> `(B, window_size, dim_in)`
        3. flatten, L2-normalize -> `(B, window_size * dim_in)` (shared input
           to the 8 reflection-vector generators)
        4. 8x bias-free `Dense(window_size * dim_in -> dim_in)` -> 8 shared
           reflection vectors `v_1..v_8`, each `(B, dim_in)`
        5. one shared bias-free `Dense(dim_in -> dim_in^2)` ("token_matrix_proj")
           applied to every token's own embedding, reshaped to
           `(B, window_size, dim_in, dim_in)` -> `M_token`
        6. for each of the 8 stages, in order:
           a. build the explicit Householder matrix `H_shared` `(B, dim_in, dim_in)`
              from `v_i` (`I - 2*outer(v_i,v_i)/dot(v_i,v_i)`)
           b. `H_token = M_token @ H_shared` (batched over `window_size`)
           c. `token' = H_token @ token` (batched matrix-vector product)
        7. flatten the reflected tokens -> `(B, window_size * dim_in)`, L2-normalize
        8. FFN: `hidden_dim` GELU -> `hidden_dim // 2` GELU -> `dim_out` linear
        9. L2-normalize -> query on S^(dim_out - 1)
        10. target table `(V, dim_out)`, L2-normalized
        11. `logits = (q @ target.T) * tau`

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the context manifold S^(dim_in - 1), and
            of each Householder reflection / per-token matrix.
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
        # one shared (context-level) reflection vector generator per stage.
        self.reflection_generators: List[layers.Dense] = [
            layers.Dense(self.dim_in, activation=None, use_bias=False, name=f"reflection_gen_{i}")
            for i in range(self.num_reflections)
        ]

        # Single shared projector, applied to every token's own embedding
        # (K/V-style: one set of weights, not a per-token table) -> dim_in^2
        # values per token, reshaped into that token's own dim_in x dim_in matrix.
        self.token_matrix_proj = layers.Dense(
            self.dim_in * self.dim_in, activation=None, use_bias=False, name="token_matrix_proj"
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
        for gen in self.reflection_generators:
            gen.build((None, flat_dim))
        self.token_matrix_proj.build((None, self.window_size, self.dim_in))
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

        # Per-token matrix: shared projector applied to each token's own
        # (fixed, original) embedding -- computed once, reused at every
        # reflection stage (the projector reads the token identity, not the
        # evolving reflected state).
        m_flat = self.token_matrix_proj(tokens, training=training)  # (B, window, dim_in^2)
        m_token = ops.reshape(m_flat, (batch_size, self.window_size, self.dim_in, self.dim_in))

        identity = ops.eye(self.dim_in, dtype=tokens.dtype)

        reflected = tokens
        for gen in self.reflection_generators:
            v = gen(flat_window, training=training)  # (B, dim_in)
            vv = ops.sum(v * v, axis=-1)  # (B,)
            vv = ops.reshape(vv, (batch_size, 1, 1))
            outer = ops.einsum("bi,bj->bij", v, v)  # (B, dim_in, dim_in)
            h_shared = identity[None, :, :] - 2.0 * outer / ops.maximum(vv, keras.config.epsilon())  # (B, dim_in, dim_in)

            h_token = ops.einsum("bwij,bjk->bwik", m_token, h_shared)  # (B, window, dim_in, dim_in)
            reflected = ops.einsum("bwij,bwj->bwi", h_token, reflected)  # (B, window, dim_in)

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
