"""Dual-Sphere-Hyper-Gate LM: `dual-sphere-hyper-token-transform` fused with
`dual-sphere-mid-gate` -- the per-window hypernetwork token transform feeds
`dense1`, and a sigmoid gate (computed from `dense1`'s output, in parallel
with `dense2`) elementwise-multiplies `dense2`'s output before `dense3`.

Both mechanisms are kept exactly as they are in their own standalone
architectures, just composed sequentially:

    tokens -> hypernetwork token transform (see
              `dual_sphere_hyper_token_transform.py`) -> u
    u -> dense1 -> h1
    h1 -> dense2 -> h2                    (unchanged, reads h1 directly)
    h1 -> gate_dense -> sigmoid -> gate    (parallel branch, see
                                            `dual_sphere_mid_gate.py`)
    h2 * gate -> dense3 -> z
    normalize(z) -> query -> cosine readout
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class DualSphereHyperGateLM(BaseLM):
    """Hypernetwork token transform (from `dual-sphere-hyper-token-transform`)
    feeding a gated FFN (from `dual-sphere-mid-gate`).

    See `dual_sphere_hyper_token_transform.py` and `dual_sphere_mid_gate.py`
    for the individual mechanisms' full rationale -- this class only
    documents how they compose. Same conventions throughout: uniform
    embedding initializer, bias-free layers, spherical projection via
    `ops.normalize(x, axis=-1, order=2)`. The positional embedding table and
    the "no previous token" sentinel are not L2-normalized, for the same
    reason given in `dual_sphere_hyper_token_transform.py`'s class docstring.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim_in: Dimensionality of the token/positional embeddings and of the
            hypernetwork FFN's input/output per position (concatenated
            triple is ``3 * dim_in`` wide; default ``dim_in=8`` gives the
            ``24x24, 24x24, 24x8`` hypernetwork-FFN shapes).
        dim_out: Dimensionality of the readout manifold S^(dim_out - 1).
        window_size: Causal lookback in tokens, left zero-padded near the
            start of a document.
        hidden_dim: Width of `dense1`; `dense2` and the gate are half of it.
        initial_tau: Initial cosine-readout temperature.
        learnable_tau: Whether ``tau`` trains.
        embeddings_initializer: Initializer for the token/target tables.
        activation: `dense1`/`dense2` hidden activation ("gelu" or "tanh").
            The hypernetwork-generated FFN's hidden activation is always
            GELU; the gate's activation is always sigmoid -- neither is
            affected by this setting.
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
        self.gate_dense = layers.Dense(self.hidden_dim // 2, activation="sigmoid", use_bias=False, name="ffn_gate")
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
        # Not L2-normalized -- see `dual_sphere_hyper_token_transform.py`.
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
        self.gate_dense.build((None, self.hidden_dim))
        self.dense3.build((None, self.hidden_dim // 2))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        """Token table projected onto S^(dim_in - 1)."""
        return ops.normalize(self.input_embeddings, axis=-1, order=2)

    def normalized_target_embeddings(self) -> Any:
        """Target table projected onto S^(dim_out - 1)."""
        return ops.normalize(self.target_embeddings, axis=-1, order=2)

    # -- forward --------------------------------------------------------------

    def call_query(self, inputs: Any, training: bool = False) -> Any:
        """Forward pass up to (but not including) the final readout matmul.

        Returns the L2-normalized query `q`, `(batch, dim_out)` -- everything
        `call()` does except the last `matmul` against `target_embeddings`
        and the `* tau` scaling. Exists so a training loop can feed `q`
        straight into `model.fused_cosine_loss.fused_cosine_cross_entropy`
        without `call()` ever materializing the full `(batch,
        vocabulary_size)` logits tensor -- see that module's docstring for
        why this matters (dense-supervision memory cost).
        """
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]
        w, d = self.window_size, self.dim_in
        f = self.token_ffn_width  # 3*dim_in

        # -- Hypernetwork token transform (identical to
        # `dual_sphere_hyper_token_transform.py`) --------------------------
        norm_in = self.normalized_input_embeddings()
        token_emb = ops.reshape(
            ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0), (batch_size, w, d)
        )

        flat = ops.reshape(token_emb, (batch_size, w * d))
        flat = ops.normalize(flat, axis=-1, order=2)
        w1 = ops.reshape(self.hyper_w1_proj(flat, training=training), (batch_size, f, f))
        w2 = ops.reshape(self.hyper_w2_proj(flat, training=training), (batch_size, f, f))
        w3 = ops.reshape(self.hyper_w3_proj(flat, training=training), (batch_size, f, d))

        sentinel = ops.broadcast_to(self.no_prev_token_embedding, (batch_size, 1, d))
        prev_token = ops.concatenate([sentinel, token_emb[:, :-1, :]], axis=1)  # (batch, w, d)
        pos_emb = ops.broadcast_to(self.positional_embeddings[None, :, :], (batch_size, w, d))
        token_input = ops.concatenate([prev_token, token_emb, pos_emb], axis=-1)  # (batch, w, f)

        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", token_input, w1))
        h = ops.nn.gelu(ops.einsum("bwi,bij->bwj", h, w2))
        transformed = ops.einsum("bwi,bij->bwj", h, w3)  # (batch, w, d)

        transformed_flat = ops.reshape(transformed, (batch_size, w * d))
        u = flat + transformed_flat  # residual, pre-normalize -- see hyper-token-transform's rationale
        u = ops.normalize(u, axis=-1, order=2)

        # -- Gated FFN (identical to `dual_sphere_mid_gate.py`) -------------
        h1 = self.dense1(u, training=training)
        h2 = self.dense2(h1, training=training)
        gate = self.gate_dense(h1, training=training)
        h2_gated = h2 * gate
        z = self.dense3(h2_gated, training=training)

        return ops.normalize(z, axis=-1, order=2)

    def call(self, inputs: Any, training: bool = False) -> Any:
        q = self.call_query(inputs, training=training)
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
