"""Dual-Sphere-Hyper-Token-Transform LM: a per-window hypernetwork generates a
tiny shared FFN that transforms every token position before the usual
`dual-sphere` FFN runs.

Forward pass:
    1. token table ``(V, dim_in)``, L2-normalized -> S^(dim_in - 1); gather
       ``window_size`` causal ids -> ``(batch, window_size, dim_in)``.
    2. positional table ``(window_size, dim_in)``, one learned row per
       position (not L2-normalized -- see class docstring).
    3. Flatten the token embeddings to ``(batch, window_size * dim_in)`` and
       project it three ways (no bias) into the WEIGHTS of a tiny per-token
       FFN, shared across all ``window_size`` positions of that example but
       different from one example to the next (a hypernetwork):
           - ``1024 -> 576`` reshaped to ``(24, 24)``  -> ``W1``
           - ``1024 -> 576`` reshaped to ``(24, 24)``  -> ``W2``
           - ``1024 -> 192`` reshaped to ``(24, 8)``   -> ``W3``
       (1024/576/192 are the sizes at the default ``window_size=128,
       dim_in=8``; see ``__init__`` for how they scale with those params.)
    4. Per position ``t``, build a 24-dim input:
       ``concat(token_emb[t-1] (or a learned "no previous token" embedding
       when t=0), token_emb[t], pos_emb[t])``, then run it through the
       hypernetwork-generated FFN: ``GELU(x @ W1) -> GELU(· @ W2) -> · @ W3``
       (8-dim output), batched across positions via ``ops.einsum`` since
       ``W1``/``W2``/``W3`` differ per training example.
    5. Concatenate the 128 transformed (8-dim) positions back into
       ``(batch, window_size * dim_in)``, ADD A RESIDUAL CONNECTION (the
       original flattened, L2-normalized token embeddings from step 3, before
       the hypernetwork touched them) around the whole token-transform block
       -- ``transformer``-style skip connection, so the hypernetwork only has
       to learn a *correction* to the raw embeddings rather than reconstruct
       them from scratch -- then L2-normalize the sum.
    6. Feed that into `dual-sphere`'s unchanged FFN: ``dense1`` (``hidden_dim``
       GELU) -> ``dense2`` (``hidden_dim // 2`` GELU) -> ``dense3``
       (``dim_out`` linear) -> L2-normalize -> query on S^(dim_out - 1).
    7. Cosine similarity against the L2-normalized target table, scaled by
       ``tau``.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class DualSphereHyperTokenTransformLM(BaseLM):
    """`dual-sphere` preceded by a per-window hypernetwork token transform.

    Same conventions as `dual-sphere`: uniform embedding initializer,
    bias-free layers throughout, spherical projection via
    `ops.normalize(x, axis=-1, order=2)`.

    The positional embedding table and the "no previous token" sentinel are
    NOT L2-normalized -- they are added into a concatenation alongside
    already-normalized token embeddings, not compared via cosine similarity
    to anything, so there is no spherical-manifold reason to constrain them;
    forcing them onto a sphere would only reduce their expressiveness for no
    benefit here.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the token/positional embeddings and of the
            hypernetwork FFN's input/output per position (concatenated
            triple is ``3 * dim_in`` wide, so the FFN's first layer is
            ``(3*dim_in, 3*dim_in)`` and its last is ``(3*dim_in, dim_in)``
            -- the default ``dim_in=8`` gives the ``24x24, 24x24, 24x8``
            shapes from the design this architecture was specified against).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens, left zero-padded near the
            start of a document.
        hidden_dim: Width of `dense1`; `dense2` is half of it (same meaning
            as in `dual-sphere`).
        initial_tau: Initial cosine-readout temperature. See `dual-sphere`.
        learnable_tau: Whether ``tau`` trains.
        embeddings_initializer: Initializer for the token/target tables (the
            positional table and "no previous token" sentinel always use
            `"uniform"`, matching every embedding-like weight in this
            project -- see the class docstring on why they aren't
            normalized).
        activation: `dense1`/`dense2` hidden activation ("gelu" or "tanh").
            The hypernetwork-generated FFN's hidden activation is always
            GELU, independent of this setting (it was specified as GELU
            explicitly and is not a `dual-sphere`-inherited knob).
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

        # Hypernetwork FFN dims: input/hidden = 3*dim_in (prev token + this
        # token + positional embedding, concatenated), output = dim_in.
        self.token_ffn_width = 3 * self.dim_in
        flat_dim = self.window_size * self.dim_in

        self.hyper_w1_proj = layers.Dense(
            self.token_ffn_width * self.token_ffn_width, activation=None, use_bias=False, name="hyper_w1_proj"
        )
        self.hyper_w2_proj = layers.Dense(
            self.token_ffn_width * self.token_ffn_width, activation=None, use_bias=False, name="hyper_w2_proj"
        )
        self.hyper_w3_proj = layers.Dense(
            self.token_ffn_width * self.dim_in, activation=None, use_bias=False, name="hyper_w3_proj"
        )

        self.dense1 = layers.Dense(self.hidden_dim, activation=self.activation, use_bias=False, name="ffn_dense1")
        self.dense2 = layers.Dense(self.hidden_dim // 2, activation=self.activation, use_bias=False, name="ffn_dense2")
        self.dense3 = layers.Dense(self.dim_out, activation=None, use_bias=False, name="ffn_dense3")

        self._flat_dim = flat_dim
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
        # Not L2-normalized -- see class docstring.
        self.positional_embeddings = self.add_weight(
            shape=(self.window_size, self.dim_in),
            initializer=initializers.get("uniform"),
            trainable=True,
            name="positional_emb",
        )
        self.no_prev_token_embedding = self.add_weight(
            shape=(self.dim_in,),
            initializer=initializers.get("uniform"),
            trainable=True,
            name="no_prev_token_emb",
        )
        self.tau = self.add_weight(
            shape=(),
            initializer=initializers.Constant(self.initial_tau),
            trainable=self.learnable_tau,
            name="tau",
        )

        self.hyper_w1_proj.build((None, self._flat_dim))
        self.hyper_w2_proj.build((None, self._flat_dim))
        self.hyper_w3_proj.build((None, self._flat_dim))
        self.dense1.build((None, self._flat_dim))
        self.dense2.build((None, self.hidden_dim))
        self.dense3.build((None, self.hidden_dim // 2))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        """Token table projected onto S^(dim_in - 1)."""
        return ops.normalize(self.input_embeddings, axis=-1, order=2)

    def normalized_target_embeddings(self) -> Any:
        """Target table projected onto S^(dim_out - 1)."""
        return ops.normalize(self.target_embeddings, axis=-1, order=2)

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]
        w, d = self.window_size, self.dim_in
        f = self.token_ffn_width  # 3*dim_in

        # 1. Token embeddings, L2-normalized, gathered per position.
        norm_in = self.normalized_input_embeddings()
        token_emb = ops.reshape(
            ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0), (batch_size, w, d)
        )

        # 2. Hypernetwork: the whole window's flattened token embeddings
        # generate one shared tiny FFN's weights for this example.
        flat = ops.reshape(token_emb, (batch_size, w * d))
        flat = ops.normalize(flat, axis=-1, order=2)
        w1 = ops.reshape(self.hyper_w1_proj(flat, training=training), (batch_size, f, f))
        w2 = ops.reshape(self.hyper_w2_proj(flat, training=training), (batch_size, f, f))
        w3 = ops.reshape(self.hyper_w3_proj(flat, training=training), (batch_size, f, d))

        # 3. Per-position FFN input: concat(prev token, this token, position).
        # `prev_token[t] = token_emb[t-1]`, with a learned sentinel at t=0
        # (there is no position -1).
        sentinel = ops.broadcast_to(self.no_prev_token_embedding, (batch_size, 1, d))
        prev_token = ops.concatenate([sentinel, token_emb[:, :-1, :]], axis=1)  # (batch, w, d)
        pos_emb = ops.broadcast_to(self.positional_embeddings[None, :, :], (batch_size, w, d))
        token_input = ops.concatenate([prev_token, token_emb, pos_emb], axis=-1)  # (batch, w, f)

        # 4. Apply the hypernetwork-generated FFN to every position, batched
        # via einsum since W1/W2/W3 differ per example (not shared globally).
        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", token_input, w1))
        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", h, w2))
        transformed = ops.einsum("bwi,bij->bwj", h, w3)  # (batch, w, d)

        # 5. Reassemble, add a residual connection around the whole
        # hypernetwork token-transform block (input `flat` + transformed
        # output, both 1024-dim at the defaults), then normalize like
        # `dual-sphere`'s flatten step. Residual is added BEFORE
        # normalization -- normalizing each branch separately first and then
        # adding would let the transform's contribution swing arbitrarily
        # relative to the input depending on `transformed`'s raw norm, which
        # is exactly the scale information a residual connection is meant to
        # preserve (same reasoning as pre-norm residual blocks elsewhere).
        transformed_flat = ops.reshape(transformed, (batch_size, w * d))
        u = flat + transformed_flat
        u = ops.normalize(u, axis=-1, order=2)

        # 6-7. `dual-sphere`'s unchanged FFN + cosine readout.
        h_ffn = self.dense1(u, training=training)
        h_ffn = self.dense2(h_ffn, training=training)
        z = self.dense3(h_ffn, training=training)

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
