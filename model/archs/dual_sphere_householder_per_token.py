"""Dual-Sphere-Householder-Per-Token LM: `dual-sphere-householder` with each
of its 8 shared, context-derived Householder reflection vectors further
transformed per token before being applied.

`dual-sphere-householder` generates 8 reflection vectors from the flattened
context window (one Dense(1024 -> 8) per reflection, all reading the same
window-level input), then applies the *same* 8 reflections to every token in
the window. That turned out to be a rigid, isometric rotation of the whole
window -- the pairwise cosine similarity between every pair of tokens is left
completely unchanged (a shared orthogonal transform preserves all angles),
so it adds no expressive power beyond what the downstream FFN's own Dense
layers could already represent.

This architecture fixes that: each reflection vector `v_i` (context-derived,
shared) is *retransformed per token* through a small shared FFN that also
sees that token's own embedding, before being used to reflect that specific
token. Concretely, for reflection stage `i` and token `t`:

    concat_input   = [v_i ; token_t]                       (16,)
    v_i,t          = Dense(128,gelu)(Dense(64,gelu)(concat_input))  -- wait, see below
                    = per_token_transform(concat_input)     (8,)
    token_t'       = H(token_t, v_i,t)                      (8,)

`per_token_transform` is one shared `Dense(16->64, gelu) -> Dense(64->128,
gelu) -> Dense(128->8, linear)` network, reused identically at all 8
reflection stages (the input differs each stage: `v_i` changes, the token
embedding being transformed at that stage stays the current per-token
state). Because `v_i,t` now genuinely differs per token (not just per
example), the composed transform can reshape token-to-token relative
geometry -- unlike the shared-only version, this is not reducible to a
single window-wide rotation.
"""

from typing import Any, Dict, List, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


def _householder_reflect(x: Any, v: Any) -> Any:
    """`H(x) = x - 2*(v.x)/(v.v) * v`, elementwise over `x`'s and `v`'s
    matching leading axes (both `(batch, window_size, dim_in)` here -- unlike
    `dual_sphere_householder.py`'s version, `v` is no longer broadcast from a
    single per-example vector, it already varies per token).
    """
    vv = ops.sum(v * v, axis=-1, keepdims=True)
    vx = ops.sum(x * v, axis=-1, keepdims=True)
    return x - 2.0 * vx * v / ops.maximum(vv, keras.config.epsilon())


@keras.saving.register_keras_serializable(package="model")
class DualSphereHouseholderPerTokenLM(BaseLM):
    """`dual-sphere-householder`, with each shared reflection vector further
    transformed per token by a small shared FFN before being applied.

    Forward pass:
        1. `input_embeddings` (V, dim_in), L2-normalized -> S^(dim_in - 1)
        2. gather `window_size` causal ids -> `(B, window_size, dim_in)`
        3. flatten, L2-normalize -> `(B, window_size * dim_in)` (shared input
           to the 8 reflection-vector generators, as in `dual-sphere-householder`)
        4. 8x bias-free `Dense(window_size * dim_in -> dim_in)` -> 8 shared
           reflection vectors `v_1..v_8`, each `(B, dim_in)`
        5. for each of the 8 stages, in order:
           a. broadcast `v_i` to every token position -> `(B, window_size, dim_in)`
           b. concatenate with the current per-token state -> `(B, window_size, 2*dim_in)`
           c. `per_token_transform` (shared across all 8 stages):
              `Dense(64, gelu) -> Dense(128, gelu) -> Dense(dim_in, linear)`
              -> per-token reflection vector `v_i,t`, `(B, window_size, dim_in)`
           c. apply `H(token, v_i,t)` to every token
        6. flatten the reflected tokens -> `(B, window_size * dim_in)`, L2-normalize
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
        per_token_hidden_dims: Hidden widths of the shared per-token
            transform (`2*dim_in` -> `per_token_hidden_dims[0]` GELU ->
            `per_token_hidden_dims[1]` GELU -> `dim_in` linear).
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
        per_token_hidden_dims: Any = (64, 128),
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
        if len(per_token_hidden_dims) != 2 or any(d < 1 for d in per_token_hidden_dims):
            raise ValueError(f"`per_token_hidden_dims` must be 2 positive ints, got {per_token_hidden_dims}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.window_size = int(window_size)
        self.num_reflections = int(num_reflections)
        self.per_token_hidden_dims = (int(per_token_hidden_dims[0]), int(per_token_hidden_dims[1]))
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

        # Shared per-token transform, reused at every reflection stage:
        # [v_i ; token] (2*dim_in) -> per_token_hidden_dims[0] (GELU) ->
        # per_token_hidden_dims[1] (GELU) -> dim_in (linear).
        self.per_token_dense1 = layers.Dense(
            self.per_token_hidden_dims[0], activation="gelu", use_bias=False, name="per_token_dense1"
        )
        self.per_token_dense2 = layers.Dense(
            self.per_token_hidden_dims[1], activation="gelu", use_bias=False, name="per_token_dense2"
        )
        self.per_token_dense3 = layers.Dense(
            self.dim_in, activation=None, use_bias=False, name="per_token_dense3"
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
        self.per_token_dense1.build((None, self.window_size, 2 * self.dim_in))
        self.per_token_dense2.build((None, self.window_size, self.per_token_hidden_dims[0]))
        self.per_token_dense3.build((None, self.window_size, self.per_token_hidden_dims[1]))
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

    def _per_token_transform(self, v_shared: Any, tokens: Any, training: bool = False) -> Any:
        """Broadcasts `v_shared` (B, dim_in) to every token position, concatenates
        with the current per-token state, and runs the shared per-token FFN.

        Called 8 times per forward pass (once per reflection stage, each
        depending on the previous stage's output -- a real data dependency
        that cannot be batched away). Measured on XPU: `keras.layers.Dense`'s
        per-call wrapper overhead (shape validation, activation dispatch)
        costs roughly 2x the underlying matmul for tensors this small
        (window-shaped, `(batch, window_size, ~16-128)`). Calling
        `ops.matmul` directly against each layer's own trained `kernel` --
        same weights, same math, `layer(x)` and `ops.matmul(x, layer.kernel)`
        are identical since every Dense here is bias-free -- bypasses that
        overhead and roughly halves this function's cost, which matters
        because it dominates this architecture's total step time (8 sequential
        calls, not parallelizable across stages).

        Returns `(B, window_size, dim_in)` -- one reflection vector per token.
        """
        v_broadcast = ops.repeat(ops.expand_dims(v_shared, axis=1), self.window_size, axis=1)
        concat_input = ops.concatenate([v_broadcast, tokens], axis=-1)
        # `approximate=False` matches `keras.layers.Dense(activation="gelu")`'s
        # default exactly (`keras.activations.gelu`'s default) -- `ops.gelu`'s
        # own default (`approximate=True`) is a different function and would
        # silently diverge from what `per_token_dense1`/`per_token_dense2`'s
        # stored weights were actually trained against.
        p = ops.gelu(ops.matmul(concat_input, self.per_token_dense1.kernel), approximate=False)
        p = ops.gelu(ops.matmul(p, self.per_token_dense2.kernel), approximate=False)
        return ops.matmul(p, self.per_token_dense3.kernel)

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]

        norm_in = self.normalized_input_embeddings()
        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        tokens = ops.reshape(flat, (batch_size, self.window_size, self.dim_in))

        flat_window = ops.reshape(tokens, (batch_size, self.window_size * self.dim_in))
        flat_window = ops.normalize(flat_window, axis=-1, order=2)

        reflected = tokens
        for gen in self.reflection_generators:
            v_shared = gen(flat_window, training=training)
            v_per_token = self._per_token_transform(v_shared, reflected, training=training)
            reflected = _householder_reflect(reflected, v_per_token)

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
            "per_token_hidden_dims": list(self.per_token_hidden_dims),
            "hidden_dim": self.hidden_dim,
            "initial_tau": self.initial_tau,
            "learnable_tau": self.learnable_tau,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
            "activation": self.activation,
        }
        return {**super().get_config(), **config}
