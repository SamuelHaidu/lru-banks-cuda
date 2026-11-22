"""nGPT: Normalized Transformer with Representation Learning on the Hypersphere.

Faithful implementation of Loshchilov, Hsieh, Sun & Ginsburg (NVIDIA),
"nGPT: Normalized Transformer with Representation Learning on the
Hypersphere" (arXiv:2410.01131, ICLR 2025) -- Table 1, Table 3, Section 2,
and Appendix A.1/A.2/A.6, followed as written:

  * RoPE (Rotary Position Embeddings) inside attention, not ALiBi.
  * Query/key normalization + per-head-dim learnable scaling `s_qk`
    (eq. 15-16).
  * SwiGLU MLP with learnable scaling `s_u`, `s_nu` on its two intermediate
    projections (eq. 17-21).
  * The residual update is NOT a plain additive residual. Each block's
    output is itself normalized, then blended into the residual stream by
    a per-dimension learnable "eigen learning rate" (`alpha_A`, `alpha_M`),
    and the *entire updated residual stream is renormalized* -- the
    Riemannian retraction step (eq. 10-11):
        h <- Norm(h + alpha_A * (Norm(ATTN(h)) - h))
        h <- Norm(h + alpha_M * (Norm(MLP(h)) - h))
  * All weight matrices (`W_q, W_k, W_v, W_o, W_u, W_nu, W_oMLP`) and both
    embedding tables are kept unit-norm along their embedding dimension --
    not just in the forward pass, but explicitly renormalized after every
    optimizer step (Appendix A.6's own warning: "one should make sure that
    not only instantiated model parameters are normalized but also the
    ones which are used by the optimizer -- missing the latter is a common
    bug"). This is implemented here as a `train_step` override, so it is
    self-contained to this architecture and requires no change to
    `model/train.py` or any other file.
  * Optimizer recipe per Table 3: plain Adam, weight_decay=0.0 (weight
    decay is redundant once every matrix is explicitly kept unit-norm).
    `model/train.py::resolve_optimizer` reads `training.optimizer` /
    `training.weight_decay` from the run config as usual -- set those to
    `"adam"` / `0.0` in the run's config to match the paper; nothing here
    forces it, consistent with every other architecture in this project.

The one deliberate departure from the paper: logits are computed for the
*single last context position* only, not densely for every position in
parallel. This project's entire training pipeline
(`model/data.py::VariablePrefixDataset`) is single-target -- one document,
one random cut point, one next token -- and `BaseLM.call()`'s contract is
`(batch, context_length) -> (batch, vocabulary_size)` logits for one
position, the same contract every other architecture in this codebase
already follows. Nothing about nGPT's per-layer mechanics (RoPE, SwiGLU,
the retraction update, weight normalization) depends on dense supervision;
only the loss/logits step changes, from "every position" to "the position
that matters for this dataset's single target."
"""

from typing import Any, Dict, List, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import keras_hub  # noqa: E402
from keras import initializers, layers, ops  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


def _l2_normalize_rows(x: Any) -> Any:
    """L2-normalizes each row (last axis) -- used for both hidden states and weight matrices."""
    return ops.normalize(x, axis=-1, order=2)


@keras.saving.register_keras_serializable(package="model")
class NGPTBlock(layers.Layer):
    """One nGPT transformer block: RoPE attention + SwiGLU MLP, both applied
    via the eigen-learning-rate LERP-then-retract update (eq. 10-11).

    Kept as a single layer (rather than splitting attention/MLP into
    separate reusable layers, as `AlibiCausalSelfAttention` does) because
    nGPT's per-block scaling factors (`s_qk`, `s_u`, `s_nu`) and eigen
    learning rates (`alpha_A`, `alpha_M`) are block-local parameters with no
    counterpart in this codebase's other architectures worth generalizing.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        rope_base: float = 10000.0,
        alpha_init: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.mlp_dim = int(mlp_dim)
        self.rope_base = float(rope_base)
        self.alpha_init = float(alpha_init)

        # -- attention projections (normalized after each optimizer step) --
        self.w_q = layers.Dense(self.dim, use_bias=False, name="w_q")
        self.w_k = layers.Dense(self.dim, use_bias=False, name="w_k")
        self.w_v = layers.Dense(self.dim, use_bias=False, name="w_v")
        self.w_o = layers.Dense(self.dim, use_bias=False, name="w_o")

        # RoPE (Su et al. 2024), as used by nGPT -- KerasHub's own layer,
        # not reimplemented here. Operates on `(batch, seq, num_heads,
        # head_dim)`, matching the shape q/k are already reshaped to below.
        self.rope = keras_hub.layers.RotaryEmbedding(max_wavelength=self.rope_base, name="rope")

        # -- SwiGLU MLP projections (normalized after each optimizer step) --
        self.w_u = layers.Dense(self.mlp_dim, use_bias=False, name="w_u")
        self.w_nu = layers.Dense(self.mlp_dim, use_bias=False, name="w_nu")
        self.w_o_mlp = layers.Dense(self.dim, use_bias=False, name="w_o_mlp")

    def build(self, input_shape: Optional[Any] = None) -> None:
        self.w_q.build((None, None, self.dim))
        self.w_k.build((None, None, self.dim))
        self.w_v.build((None, None, self.dim))
        self.w_o.build((None, None, self.dim))
        self.w_u.build((None, None, self.dim))
        self.w_nu.build((None, None, self.dim))
        self.w_o_mlp.build((None, None, self.mlp_dim))

        # Eigen learning rates (eq. 10-11): per-dimension learnable step size.
        self.alpha_a = self.add_weight(
            shape=(self.dim,), initializer=initializers.Constant(self.alpha_init),
            trainable=True, name="alpha_a",
        )
        self.alpha_m = self.add_weight(
            shape=(self.dim,), initializer=initializers.Constant(self.alpha_init),
            trainable=True, name="alpha_m",
        )
        # s_qk (eq. 15-16): per-head-dim scaling restoring q/k magnitude after normalization.
        self.s_qk = self.add_weight(
            shape=(self.head_dim,), initializer=initializers.Constant(1.0),
            trainable=True, name="s_qk",
        )
        # s_u, s_nu (eq. 20-21): scaling on the two SwiGLU intermediate projections.
        self.s_u = self.add_weight(
            shape=(self.mlp_dim,), initializer=initializers.Constant(1.0),
            trainable=True, name="s_u",
        )
        self.s_nu = self.add_weight(
            shape=(self.mlp_dim,), initializer=initializers.Constant(1.0),
            trainable=True, name="s_nu",
        )
        self.built = True

    # -- the constrained weight matrices, for the post-optimizer-step renorm --
    @property
    def normalized_matrices(self) -> List[Any]:
        return [
            self.w_q.kernel, self.w_k.kernel, self.w_v.kernel, self.w_o.kernel,
            self.w_u.kernel, self.w_nu.kernel, self.w_o_mlp.kernel,
        ]

    def _attention(self, h: Any) -> Any:
        batch_size = ops.shape(h)[0]
        seq_len = ops.shape(h)[1]

        q = ops.reshape(self.w_q(h), (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(self.w_k(h), (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(self.w_v(h), (batch_size, seq_len, self.num_heads, self.head_dim))

        # QK normalization + scaling (eq. 15-16): restores q/k onto their own
        # hypersphere after the (normalized) projection, then rescales.
        q = _l2_normalize_rows(q) * self.s_qk
        k = _l2_normalize_rows(k) * self.s_qk

        q = self.rope(q)
        k = self.rope(k)

        # Paper's softmax scale is sqrt(d_k), not the usual 1/sqrt(d_k) --
        # see Section 2.3.2: normalized q/k have dot-product variance
        # 1/d_k instead of the unnormalized 1, so restoring variance 1
        # requires scaling *up* by sqrt(d_k) rather than down.
        scale = float(self.head_dim) ** 0.5
        # No causal mask, no ALiBi bias: this project's training objective
        # supervises a single target token per window, never placed inside
        # the context window itself (see module docstring) -- bidirectional
        # attention over the full window is safe here for the same reason
        # `alibi_bidirectional_attention.py` established for the ALiBi
        # architectures.
        out = ops.dot_product_attention(q, k, v, bias=None, is_causal=False, scale=scale)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))
        return self.w_o(out)

    def _mlp(self, h: Any) -> Any:
        u = self.w_u(h) * self.s_u
        nu = self.w_nu(h) * self.s_nu * (float(self.dim) ** 0.5)  # eq. 21: rescale nu by sqrt(d_model)
        swiglu = u * ops.silu(nu)
        return self.w_o_mlp(swiglu)

    def call(self, h: Any, training: bool = False) -> Any:
        # Attention sub-block: eq. 10.
        h_a = _l2_normalize_rows(self._attention(h))
        h = _l2_normalize_rows(h + self.alpha_a * (h_a - h))

        # MLP sub-block: eq. 11.
        h_m = _l2_normalize_rows(self._mlp(h))
        h = _l2_normalize_rows(h + self.alpha_m * (h_m - h))
        return h

    def get_config(self) -> Dict[str, Any]:
        config = {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "rope_base": self.rope_base,
            "alpha_init": self.alpha_init,
        }
        return {**super().get_config(), **config}


@keras.saving.register_keras_serializable(package="model")
class NGPTLM(BaseLM):
    """nGPT language model: embed -> `num_layers` `NGPTBlock`s -> last-position readout.

    Args:
        vocabulary_size: Token vocabulary size.
        dim: Model/embedding dimension (`d_model` in the paper).
        window_size: Context length.
        num_layers: Number of `NGPTBlock`s.
        num_heads: Attention heads per block.
        mlp_dim: SwiGLU MLP intermediate width (`d_MLP`, paper uses `4*d_model`).
        rope_base: RoPE frequency base (10000 in the paper).
        alpha_init: Initial value for both eigen learning rates (paper:
            ~`1/sqrt(d_model)`, order `1/n_layers`; exposed here as a single
            constructor arg rather than the paper's separate init/scale
            trick, since this project's optimizer wiring has no equivalent
            of nGPT's internal library's per-parameter LR scaling).
        s_z_init: Initial value for the logit scaling vector `s_z` (eq. 3).
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 768,
        rope_base: float = 10000.0,
        alpha_init: float = 0.05,
        s_z_init: float = 1.0,
        embeddings_initializer: str = "normal",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim <= 0:
            raise ValueError(f"`dim` must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `num_heads` ({num_heads})")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"`dim // num_heads` ({dim // num_heads}) must be even for RoPE")
        if mlp_dim <= 0:
            raise ValueError(f"`mlp_dim` must be positive, got {mlp_dim}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.rope_base = float(rope_base)
        self.alpha_init = float(alpha_init)
        self.s_z_init = float(s_z_init)
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        self.blocks = [
            NGPTBlock(
                dim=self.dim, num_heads=self.num_heads, mlp_dim=self.mlp_dim,
                rope_base=self.rope_base, alpha_init=self.alpha_init,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]

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
            shape=(self._vocabulary_size, self.dim),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="input_emb",
        )
        self.output_embeddings = self.add_weight(
            shape=(self._vocabulary_size, self.dim),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="output_emb",
        )
        # s_z (eq. 3): per-vocabulary-dimension logit scaling.
        self.s_z = self.add_weight(
            shape=(self._vocabulary_size,),
            initializer=initializers.Constant(self.s_z_init),
            trainable=True,
            name="s_z",
        )
        for block in self.blocks:
            block.build((None, self.window_size, self.dim))
        self.built = True

    def normalized_input_embeddings(self) -> Any:
        return _l2_normalize_rows(self.input_embeddings)

    def normalized_output_embeddings(self) -> Any:
        return _l2_normalize_rows(self.output_embeddings)

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = ops.cast(inputs, "int32")
        batch_size = ops.shape(token_ids)[0]

        norm_in = self.normalized_input_embeddings()
        norm_out = self.normalized_output_embeddings()

        flat = ops.take(norm_in, ops.reshape(token_ids, (-1,)), axis=0)
        h = ops.reshape(flat, (batch_size, self.window_size, self.dim))

        for block in self.blocks:
            h = block(h, training=training)

        # No dense supervision (see module docstring): only the last
        # context position's representation is read out. `h` is already
        # unit-norm (every block ends with a retraction), so no further
        # normalization is needed before the cosine readout -- matching
        # the paper's own note that no extra final normalization is
        # required (Section 2.2.2).
        last = h[:, -1, :]
        cos_sim = ops.matmul(last, ops.transpose(norm_out))  # (batch, vocab), bounded [-1, 1]
        return cos_sim * self.s_z

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim": self.dim,
            "window_size": self.window_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "rope_base": self.rope_base,
            "alpha_init": self.alpha_init,
            "s_z_init": self.s_z_init,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
        }
        return {**super().get_config(), **config}

    # -- weight normalization after every optimizer step (Appendix A.6) -------

    def train_step(self, data: Any) -> Any:
        """Runs the normal Keras/torch train step, then retracts every
        constrained weight matrix back onto its unit-norm hypersphere.

        Appendix A.6: "one should make sure that not only instantiated
        model parameters are normalized but also the ones which are used
        by the optimizer. Missing the latter is a common bug." Overriding
        `train_step` (rather than a callback external to this class) keeps
        that guarantee intrinsic to the architecture itself, so it holds
        under `model.fit()`, `model.train_on_batch()`, or any other caller,
        with no dependency on `model/train.py`'s callback list.
        """
        result = super().train_step(data)

        import torch

        with torch.no_grad():
            self.input_embeddings.assign(_l2_normalize_rows(self.input_embeddings.value))
            self.output_embeddings.assign(_l2_normalize_rows(self.output_embeddings.value))
            for block in self.blocks:
                for w in block.normalized_matrices:
                    # Dense kernels are (in_dim, out_dim); each output
                    # feature's incoming weight vector (a column) is what
                    # should be unit-norm (matches "normalize along the
                    # embedding dimension" -- the dimension being
                    # projected *from* -- per Section 2.6, step 2).
                    normalized = ops.normalize(w.value, axis=0, order=2)
                    w.assign(normalized)

        return result
