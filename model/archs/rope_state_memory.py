"""`rope-state-memory`: a RoPE decayed token sum, plus a memory addressed by that sum.

Pre-norm residual stream, as in pre-norm transformers: x^0 = embedding, every layer adds to the stream and never
rescales it, and a final RMSNorm feeds the readout. One layer, at position t:

    input    u_t = RMSNorm(x_t)                             what the state and the value read
    state    s_t = lambda R_1 s_{t-1} + gamma u_t           gamma = sqrt(1 - lambda^2)  (LRU normalization)
             = gamma sum_{j <= t} lambda^(t - j) R_{t - j} u_j
             R_k = RoPE rotation by k (half-split pairs, frequencies rope_base^(-p / (dim/2))), lambda = e^{-alpha}
             one scalar decay per layer, learned (init half-life 2 tokens).
    address  a_t = W_k s_{t-1} for the write (the state before the update; 0 right after a document marker),
             a_t' = W_q s_t for the read (the state after it): linear projections, no bias (dim -> num_banks for
             the banks, dim -> key_dim for the linear memory).
    value    v_t = W_v u_t             the current input, no RoPE
    memory   `memory="banks"` (default): S banks, each a dim-wide vector with a learned decay mu_b.
                 w_t = softmax(a_t)                       write weights        r_t = softmax(a_t')              read
                 bank_b <- mu_b bank_b + (1 - mu_b) w_tb v_t,   z_b <- mu_b z_b + (1 - mu_b) w_tb   (written mass)
                 o_t = sum_b r_tb bank_b / sum_b r_tb z_b
             The read is a weighted average of past values, at the scale of v whatever S is:
             o_t = sum_j c_tj v_j / sum_j c_tj,  c_tj = sum_b r_tb (1 - mu_b) mu_b^(t - j) w_jb,
             where r_t . w_j is the probability that the read and the write land in the same bank (sharp when the
             softmaxes are). Without the normalization a read would be ~1/S of a value, and with the gate closed at
             init neither the gate nor the addresses got a usable gradient. W_k = W_q = 0 at init (both softmaxes
             exactly uniform; their gradient s (x) dlogits is not zero), bank half-lives log-spaced from
             `memory_half_life_min` to `_max`: with
             uniform addresses c_tj ~ sum_b (1 - mu_b) mu_b^(t - j), a mixture of log-spaced exponentials ~ 1/(t - j),
             so the read starts as a power-law-weighted average of the past. The kernels carry z as an extra
             column of ones in the values (padded to dim + 16).
             `memory="banks_delta"`: the same banks written by the delta rule (normalized LMS, as DeltaNet):
                 k_t = softmax(a_t) / |.|,  q_t = softmax(a_t') / |.|,  beta_t = sigmoid(w_b . s_{t-1} + b_b)
                 Z_t = D Z_{t-1} + beta_t k_t (v_t - k_t^T D Z_{t-1})^T,   D = diag(mu_b),   o_t = Z_t^T q_t
             Each write stores only what the memory does not already return at that address (the error), so
             banks shared by overlapping addresses coordinate instead of averaging; |k| = 1 and 0 < beta < 1 make
             every step a contraction. Run chunkwise (`delta_chunk` tokens, WY form as in DeltaNet): inside a chunk
             the errors e come from one unit-lower-triangular solve (I + diag(beta) A) e = diag(beta) (V - carry),
             A_ij = k_i^T D^(i - j) k_j, then outputs and the chunk's end state are matmuls; only the (num_banks x
             dim) state goes chunk to chunk. Torch ops (autograd for the backward), no custom kernel.
             `memory="linear"`: M_t = mu M_{t-1} + k_t v_t^T, o_t = sqrt(1 - mu^2) M_t^T q_t with k, q the
             L2-normalized addresses (key_dim x dim, one decay, init half-life `memory_half_life`).
    read     h_t = a_t s_t + g_t o_t                         two independent gates from the state s_t:
             a_t = sigmoid(w_a . s_t + b_a)                  keep the state, starts at 0.9975 (b_a = +6)
             g_t = sigmoid(w_g . s_t + b_g)                  take the memory read, starts at 0.0025 (b_g = -6)
             (a -> 0, g -> 1: the read replaces the state.) `read_mix="add"`: a = 1; `"tanh"`: one coupled
             weight, h = (1 - c)/2 s + (1 + c)/2 o.
    output   m_t = MLP(RMSNorm(h_t))                         dim -> mlp_hidden (4 dim) -> dim, tanh GELU
             y_t = a2_t x_t + g2_t m_t                       x_t the layer input; two gates from RMSNorm(m_t):
             a2_t = sigmoid(w_k . RMSNorm(m_t) + b_k)        keep the input, starts at 0.9975
             g2_t = sigmoid(w_m . RMSNorm(m_t) + b_m)        take the MLP output, starts at 0.0025
             (`mlp_residual="input"`: y_t = x_t + m_t; `"state"`: y_t = h_t + m_t; `"mix"`: one coupled weight.)
             The MLP output layer starts at zero.
    readout  logits = W_out^T RMSNorm(y_t) (last layer); layer n + 1 reads y of layer n as its x.

A fresh model is about logits = W_out^T RMSNorm(embedding): every layer starts as (nearly) the identity on its input,
and learns how much memory and MLP to add or to put in place of the state / input. Norm scales, decays, the address
projections and the gates are not weight-decayed.

Both the state and the memory restart at every document marker (`reset_token`).

Training (dense supervision on packed rows) on the XPU runs both recurrences as sequential scans in custom kernels
(`model/kernels/rsm_scan.sycl`): every input read once, the state in registers. Elsewhere the state runs chunkwise
in torch (one (t x t) decay matrix per chunk, the decay being one scalar) and the memory token by token (the
reference the kernels are tested against). The MLP recomputes its hidden layer in the backward pass.

Generation carries (s, memory) per layer token by token: O(1) per token, unbounded context. `with_context_length(n)`
gives a copy that reads only the last n tokens (finite context).
"""

from typing import Any, Dict, List, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

import torch  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.archs._norm import RMSNorm  # noqa: E402
from model.archs.rope_token_banks import _const  # noqa: E402
from model.archs.wave_arrow import _RecomputeMLP  # noqa: E402
from model.kernels import rsm_scan  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


def _rot(x: Any, cos: Any, sin: Any) -> Any:
    """Half-split RoPE rotation of `x` `(..., dim)` (pairs p, p + dim/2) by angles with these cos / sin `(..., dim/2)`."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class _CarryState(torch.autograd.Function):
    """`s + live_t lambda^(t + 1) R_{t + 1} s0`: the carried state's contribution to the scanned states. The term is
    recomputed in the backward pass (only alpha gets a gradient: the carry is detached), so no row-sized
    intermediate is kept."""

    @staticmethod
    def _term(s0, live, alpha, cos, sin):
        n = live.shape[1]
        lam_t = torch.exp(-alpha * torch.arange(1, n + 1, device=s0.device, dtype=s0.dtype))
        return (live * lam_t)[..., None] * _rot(s0[:, None], cos[1:n + 1], sin[1:n + 1]), lam_t

    @staticmethod
    def forward(ctx, s, s0, live, alpha, cos, sin):  # type: ignore[override]
        ctx.save_for_backward(s0, live, alpha, cos, sin)
        return s + _CarryState._term(s0, live, alpha, cos, sin)[0]

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        s0, live, alpha, cos, sin = ctx.saved_tensors
        term, _ = _CarryState._term(s0, live, alpha, cos, sin)
        pos = torch.arange(1, live.shape[1] + 1, device=g.device, dtype=g.dtype)
        d_alpha = -(g * term).sum(-1).mul(pos).sum().reshape(alpha.shape)
        return g, None, None, d_alpha, None, None


class _TiedReadout(layers.Layer):
    """The readout as the token embedding table, transposed: logits = h E^T. No weights of its own; `kernel`
    (dim, vocab) is a view of the table, so the fused readout loss and every caller of `readout.kernel` work as with
    a Dense readout, and the gradients of both uses reach the one table."""

    def __init__(self, embedding: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self._table = lambda: embedding.embeddings  # a function: not tracked as a second owner of the table

    @property
    def kernel(self) -> Any:
        return self._table().value.T

    def call(self, h: Any) -> Any:
        return torch.as_tensor(h) @ self.kernel


def _rms(x: Any, scale: Any) -> Any:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * scale


@keras.saving.register_keras_serializable(package="model")
class RopeStateMemoryLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        dim: Width of the embeddings, the state, the values and the MLP output.
        window_size: Training row length (packed documents) and validation window. The model has no window.
        key_dim: Width of the linear memory's addresses (W_q, W_k: dim -> key_dim; it is key_dim x dim). The banks'
            addresses are num_banks wide.
        mlp_hidden: Hidden width of the residual MLP (default 4 x dim).
        chunk_size: Chunk length of the parallel scan (speed and memory only; the result is the same for any value).
        reset_token: Document marker id; the state and the memory restart at it (the dataset's `doc_token_id`).
        num_layers: Stacked layers; layer n + 1 reads layer n's output y.
        rope_base: RoPE base of the state's rotation frequencies.
        state_half_life: Initial half-life of the state decay lambda, in tokens (learned from there).
        memory: "banks" (default: softmax-addressed banks, a decay each), "banks_delta" (the same banks written by
            the delta rule, with a learned write strength per token) or "linear" (decayed linear attention).
        delta_chunk: Chunk length of the delta rule's parallel form (speed only; 64 was fastest: 16 / 32 / 64 took
            554 / 396 / 337 ms fwd + bwd at 256 x 256). The decay rate is capped at 80 / delta_chunk.
        num_banks: Banks S of the bank memory (16, 32, 64 or 128 for the XPU kernels).
        memory_half_life_min, memory_half_life_max: The banks' initial half-lives, log-spaced between these
            (a power-law-like mixture); learned per bank.
        memory_half_life: Initial half-life of the linear memory's decay (`memory="linear"`).
        read_gate_bias: Initial bias of the read gate sigmoid(w_g . q + b); weights start at 0, so every gate starts
            at sigmoid(read_gate_bias) (-6: 0.0025, memory reads off).
        write_gate_bias: Initial bias of the write gate of `memory="banks"`: the token's write is
            softmax(W_k s_{t-1}) * beta_t, beta_t = sigmoid(w_w . s_{t-1} + b_w), its own projection of the state
            before the update (the key's input); it attenuates the write scores (the banks and their mass z alike).
            w_w starts at 0, so every gate starts at sigmoid(write_gate_bias) (-6: 0.0025). The read is divided by
            the written mass, so only the gates' ratios between tokens matter: a token with beta near 0 is skipped,
            the banks keep their averages. None: no write gate (runs saved before it existed).
        write_gate_input: What the write gate projects: "state" (default, s_{t-1}, w_w: dim -> 1) or "logits" (the
            write logits W_k s_{t-1}, w_w: num_banks -> 1; the first version, which saturated open: runs saved with
            it load with it).
        pre_norm: RMSNorm on each layer's input (what the state and value read) and before the readout, as in
            pre-norm transformers. False: both read the raw stream (no fixed scale; the stream grows with depth).
        embedding_std: Std of the token embedding init (the forward pass is scale-free through the norms; the scale
            sets how fast Adam's ~lr-sized steps move the embedding relative to its size).
        read_mix: How the memory read o joins the state s. "gates" (default): h = a s + g o with two independent
            gates from the read logits q, a = sigmoid(w_a . q + b_a) (keep the state, b_a starts at keep_gate_bias)
            and g = sigmoid(w_g . q + b_g) (take the read, b_g starts at read_gate_bias): each has its own gradient,
            so the read can be added without first beating the state, and a -> 0, g -> 1 replaces the state.
            "add": h = s + g o. "tanh": one coupled weight, c = tanh(w_g . q + b_g), h = (1 - c)/2 s + (1 + c)/2 o
            (b_g starts at read_gate_bias / 2); its gate stayed shut in training: opening it costs state.
        mlp_residual: What the layer outputs. "gates" (default): m = MLP(RMSNorm(h)), y = a2 x + g2 m, x the layer
            input, with two independent gates from the normalized MLP output, a2 = sigmoid(w_k . RMSNorm(m) + b_k)
            (keep the input, starts at keep_gate_bias) and g2 = sigmoid(w_m . RMSNorm(m) + b_m) (take the MLP,
            starts at mlp_mix_bias). "mix": m = MLP(RMSNorm(h)), a mixing head
            c = tanh(w_m . RMSNorm(m) + b_m) (dim -> 1 on the normalized MLP output, so it cannot saturate by m's
            scale) and y = (1 - c)/2 x + (1 + c)/2 m, x the layer input (the token's embedding in layer 1): the
            layer chooses per position between passing its input on and the MLP's output. w_m starts at 0 and b_m at
            mlp_mix_bias / 2, so the MLP's weight (1 + c)/2 starts at sigmoid(mlp_mix_bias) (-6: 0.0025): like the
            read and write gates, the layer starts as the identity and learns how much to take from the MLP.
            "input": y = x + MLP(RMSNorm(h)), x a residual stream through every layer; "state": y = h + MLP(RMSNorm(h)).
        keep_gate_bias: Initial bias of the keep gates (state, input) of the "gates" mixes (+6: 0.9975).
        read_gate_input: What the read-side gates (the read gate g and the keep-the-state gate a) project: "state"
            (default, s_t, the state after the update: its own dim -> 1 projection, bounded input, as the write gate
            reads s_{t-1}) or "logits" (the read logits W_q s_t: unnormalized, they grow as the addressing sharpens,
            and their softmax-invariant mean is free; runs saved before this option load with it).
        mlp_tf32: Run the MLP matmuls with TF32 tensor cores on CUDA only (no effect on other devices).
        tie_embeddings: True: the readout is the token embedding table transposed (logits = RMSNorm(y) E^T), one
            table for both ends; False (default): a separate readout matrix.
        mlp_recompute: True (default): the MLP's hidden layer is recomputed in the backward pass (memory); False:
            its pre-activations are kept from the forward pass (rows x mlp_hidden floats per layer), one matmul less
            per layer (fused XPU MLP only).
        mlp_fp16: the fused XPU MLP's matmuls in the assembly fp16 kernel (`model/kernels/hgemm.py`: fp16
            products, f32 sums, gradients scaled into fp16's range), ~1.7x the fp32 matmuls; False (default): fp32.
    """

    # raw decays: weight decay would pull them toward a half-life of 1 / ln 2 tokens; the read gate: its bias would
    # be pulled to 0, opening the gate (-6 -> -2.2 in 10k steps at lr 0.01) with no gradient asking for it
    # RMSNorm scales (variable name "scale"): decay would pull them to 0, shrinking what the next op reads
    # address projections: decay would pull the bank addressing back toward uniform (blurry reads)
    # the write gate: the same as the read gate
    # the mixing head and the keep / MLP gates: decay would pull them to 0, a fixed half-half mix
    no_weight_decay_names = ("decay_", "read_gate", "write_gate", "mix_head", "keep_", "mlp_gate", "scale", "address_")
    # 64 rows x 1024 tokens: the same ~65k targets per step as wave-arrow
    config_defaults = {"training": {"learning_rate": 0.01, "warmup_steps": 200, "dense_supervision": True},
                       "data": {"batch_size": 64}}
    packed_documents = True
    stateful_generation = True
    # run the scans with the custom XPU kernels (`model/kernels/rsm_scan.sycl`) when they can; False forces the
    # chunked torch path (the reference the kernels are tested against)
    use_kernels = True

    def __init__(self, vocabulary_size: int = 10000, dim: int = 128, window_size: int = 1024, key_dim: int = 128,
                 mlp_hidden: int = 512, chunk_size: int = 128, reset_token: int = 0,
                 num_layers: int = 1, rope_base: float = 10000.0, state_half_life: float = 2.0,
                 memory: str = "banks", num_banks: int = 128, memory_half_life_min: float = 1.0,
                 memory_half_life_max: float = 1024.0, memory_half_life: float = 256.0, read_gate_bias: float = -6.0,
                 write_gate_bias: Optional[float] = -6.0, write_gate_input: str = "state",
                 mlp_residual: str = "gates", read_mix: str = "gates", mlp_mix_bias: float = -6.0,
                 keep_gate_bias: float = 6.0, read_gate_input: str = "state", delta_chunk: int = 64,
                 embedding_std: float = 0.02, pre_norm: bool = True,
                 mlp_tf32: bool = True, tie_embeddings: bool = False, mlp_recompute: bool = True,
                 mlp_fp16: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        if dim <= 0 or dim % 2:
            raise ValueError(f"`dim` must be positive and even, got {dim}")
        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.key_dim = int(key_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.chunk_size = int(chunk_size)
        self.reset_token = int(reset_token)
        self.num_layers = int(num_layers)
        self.rope_base = float(rope_base)
        self.state_half_life = float(state_half_life)
        self.memory_half_life = float(memory_half_life)
        if memory not in ("banks", "banks_delta", "linear"):
            raise ValueError(f"memory must be 'banks', 'banks_delta' or 'linear', got {memory!r}")
        self.memory = memory
        self.delta_chunk = int(delta_chunk)
        self.num_banks = int(num_banks)
        self.memory_half_life_min = float(memory_half_life_min)
        self.memory_half_life_max = float(memory_half_life_max)
        self.read_gate_bias = float(read_gate_bias)
        self.write_gate_bias = None if write_gate_bias is None else float(write_gate_bias)
        if write_gate_input not in ("state", "logits"):
            raise ValueError(f"write_gate_input must be 'state' or 'logits', got {write_gate_input!r}")
        self.write_gate_input = write_gate_input
        self.embedding_std = float(embedding_std)
        self.pre_norm = bool(pre_norm)
        if mlp_residual not in ("input", "state", "mix", "gates"):
            raise ValueError(f"mlp_residual must be 'input', 'state', 'mix' or 'gates', got {mlp_residual!r}")
        self.mlp_residual = mlp_residual
        if read_mix not in ("add", "tanh", "gates"):
            raise ValueError(f"read_mix must be 'add', 'tanh' or 'gates', got {read_mix!r}")
        self.read_mix = read_mix
        self.mlp_mix_bias = float(mlp_mix_bias)
        self.keep_gate_bias = float(keep_gate_bias)
        if read_gate_input not in ("state", "logits"):
            raise ValueError(f"read_gate_input must be 'state' or 'logits', got {read_gate_input!r}")
        self.read_gate_input = read_gate_input
        self.mlp_tf32 = bool(mlp_tf32)
        self.mlp_recompute = bool(mlp_recompute)
        self.mlp_fp16 = bool(mlp_fp16)
        # small, as in GPT-2 (std 0.02): with the norms the forward pass does not depend on the embedding's scale,
        # and Adam's ~lr-sized steps move a small embedding fast relative to its size (std 1 learned slower)
        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_embedding",
                                                embeddings_initializer=keras.initializers.RandomNormal(stddev=self.embedding_std))
        if self.pre_norm:
            self.final_norm = RMSNorm(name="final_norm")
        for n in range(self.num_layers):
            for name in ("in_norm", "mlp_norm")[0 if self.pre_norm else 1:]:
                setattr(self, self._name(name, n), RMSNorm(name=self._name(name, n)))
            for name, width, bias in (("value", self.dim, False), ("mlp_in", self.mlp_hidden, True),
                                      ("mlp_out", self.dim, True)):
                # the MLP output layer starts at zero (ReZero-like): at init every layer passes its input through,
                # so the model starts as the linear bigram logits = W_out^T embedding, and each layer's addition to
                # the residual stream grows only as far as the gradients ask
                init = "zeros" if name == "mlp_out" else "glorot_uniform"
                setattr(self, self._name(name, n), layers.Dense(width, use_bias=bias, kernel_initializer=init,
                                                                name=self._name(name, n)))
        self.tie_embeddings = bool(tie_embeddings)
        self.readout = (_TiedReadout(self.token_embedding, name="readout") if self.tie_embeddings else
                        layers.Dense(self._vocabulary_size, use_bias=False, name="readout"))
        self.build()

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        D = self.dim
        self.token_embedding.build((None,))
        self.readout.build((None, D))
        if self.pre_norm:
            self.final_norm.build((None, D))
        raw = lambda half_life: _softplus_inv(np.log(2.0) / half_life)
        for n in range(self.num_layers):
            for name, width in ((("in_norm", D),) if self.pre_norm else ()) + (
                    ("mlp_norm", D), ("value", D), ("mlp_in", D), ("mlp_out", self.mlp_hidden)):
                self._w(name, n).build((None, width))
            weight = lambda name, shape, init: setattr(self, self._name(name, n), self.add_weight(
                name=self._name(name, n), shape=shape, initializer=init))
            weight("decay_state", (), keras.initializers.Constant(raw(self.state_half_life)))
            # read / write address projections W_q, W_k (dim -> address width); zero for the banks (uniform softmaxes)
            for name in ("address_q", "address_k"):
                weight(name, (D, self.address_dim), "glorot_uniform" if self.memory == "linear" else "zeros")
            if self.memory == "linear":
                weight("decay_memory", (), keras.initializers.Constant(raw(self.memory_half_life)))
            else:
                half_lives = np.geomspace(self.memory_half_life_min, self.memory_half_life_max, self.num_banks)
                weight("decay_banks", (self.num_banks,), _const([raw(h) for h in half_lives]))
                if self.memory == "banks_delta":  # write strength beta_t = sigmoid(w_b . s_{t-1} + b_b), 0.5 at init
                    weight("write_strength_w", (D, 1), "zeros")
                    weight("write_strength_b", (1,), "zeros")
            # read gate: plain named weights (Keras matches weight-decay exclusions against the variable name)
            for name, shape, init in (("read_gate_w", (self.read_gate_dim, 1), "zeros"),
                                      ("read_gate_b", (1,), keras.initializers.Constant(
                                          self.read_gate_bias / 2 if self.read_mix == "tanh" else self.read_gate_bias))):
                setattr(self, self._name(name, n), self.add_weight(name=self._name(name, n), shape=shape, initializer=init))
            const = keras.initializers.Constant
            if self.read_mix == "gates":  # keep-the-state gate, from the read logits
                for name, shape, init in (("keep_state_w", (self.read_gate_dim, 1), "zeros"),
                                          ("keep_state_b", (1,), const(self.keep_gate_bias))):
                    setattr(self, self._name(name, n), self.add_weight(name=self._name(name, n), shape=shape,
                                                                       initializer=init))
            if self.mlp_residual == "gates":  # keep-the-input and take-the-MLP gates, from the normalized MLP output
                for name, shape, init in (("keep_input_w", (D, 1), "zeros"), ("keep_input_b", (1,), const(self.keep_gate_bias)),
                                          ("mlp_gate_w", (D, 1), "zeros"), ("mlp_gate_b", (1,), const(self.mlp_mix_bias))):
                    setattr(self, self._name(name, n), self.add_weight(name=self._name(name, n), shape=shape,
                                                                       initializer=init))
            if self.mlp_residual == "mix":  # the output mixing head, on the normalized MLP output
                for name, shape, init in (("mix_head_w", (D, 1), "zeros"),
                                          ("mix_head_b", (1,), keras.initializers.Constant(self.mlp_mix_bias / 2))):
                    setattr(self, self._name(name, n), self.add_weight(name=self._name(name, n), shape=shape,
                                                                       initializer=init))
            if self.write_gate:
                width = D if self.write_gate_input == "state" else self.address_dim
                for name, shape, init in (("write_gate_w", (width, 1), "zeros"),
                                          ("write_gate_b", (1,), keras.initializers.Constant(self.write_gate_bias))):
                    setattr(self, self._name(name, n), self.add_weight(name=self._name(name, n), shape=shape, initializer=init))
        self.built = True

    @staticmethod
    def _name(name: str, n: int) -> str:
        """Parameter name in layer n (0-based): layer 1 keeps the plain names."""
        return name if n == 0 else f"{name}_l{n + 1}"

    def _w(self, name: str, n: int) -> Any:
        return getattr(self, self._name(name, n))

    # -- constants ------------------------------------------------------------

    def _rope(self, t: int, like: Any) -> Tuple[Any, Any]:
        """cos and sin of k theta for k = 0..t: `(t + 1, dim/2)` each (cached per length and device)."""
        cache = self.__dict__.setdefault("_rope_cache", {})
        key = (t, str(like.device), like.dtype)
        if key not in cache:
            half = self.dim // 2
            theta = self.rope_base ** (-np.arange(half, dtype=np.float64) / half)
            ang = np.arange(t + 1, dtype=np.float64)[:, None] * theta[None]
            cache[key] = tuple(torch.as_tensor(f(ang), dtype=like.dtype, device=like.device) for f in (np.cos, np.sin))
        return cache[key]

    def decays(self, n: int = 0) -> Tuple[Any, Any]:
        """(lambda, mu) of layer n: the state's decay per token, and the memory's (a scalar for the linear memory,
        `(num_banks,)` for the banks)."""
        mem = self._w("decay_memory" if self.memory == "linear" else "decay_banks", n).value
        return (torch.exp(-torch.nn.functional.softplus(self._w("decay_state", n).value)),
                torch.exp(-torch.nn.functional.softplus(mem)))

    @property
    def read_gate_dim(self) -> int:
        """Width of the read-side gates' input: dim (the state) or address_dim (the read logits)."""
        return self.dim if self.read_gate_input == "state" else self.address_dim

    def read_gate_in(self, s: Any, read: Any) -> Any:
        """The read-side gates' input from the state after the update s and the read logits `read`."""
        return s if self.read_gate_input == "state" else read

    @property
    def write_gate(self) -> bool:
        """Whether the bank writes pass a learned gate (`memory="banks"` with a `write_gate_bias`)."""
        return self.memory == "banks" and self.write_gate_bias is not None

    @property
    def memory_rows(self) -> int:
        """Rows of the memory state `(B, memory_rows, memory_cols)`: key_dim (linear) or num_banks."""
        return self.key_dim if self.memory == "linear" else self.num_banks

    @property
    def address_dim(self) -> int:
        """Width of the read / write addresses: num_banks (banks) or key_dim (linear)."""
        return self.key_dim if self.memory == "linear" else self.num_banks

    @property
    def memory_cols(self) -> int:
        """Columns of the memory state: dim (linear), or dim + 16 for the banks (the values, then the written
        mass z in column dim, then zero padding to a width the scan kernel takes)."""
        return self.dim + 16 if self.memory == "banks" else self.dim

    @staticmethod
    def _decay_consts(raw: Any, t: int) -> Dict[str, Any]:
        """For a decay e^{-softplus(raw)} and chunk length t: the causal (t x t) matrix rate^(i - j), rate^(i + 1)
        and rate^(t - 1 - j) `(t,)`, rate^t and the LRU factor sqrt(1 - rate^2)."""
        a = torch.nn.functional.softplus(raw)
        pos = torch.arange(t, device=raw.device, dtype=raw.dtype)
        d = pos[:, None] - pos[None, :]
        return {"mat": torch.exp(-a * d.clamp_min(0)) * (d >= 0).to(raw.dtype),
                "next": torch.exp(-a * (pos + 1)), "to_end": torch.exp(-a * (t - 1 - pos)),
                "all": torch.exp(-a * t), "lru": torch.sqrt(-torch.expm1(-2 * a))}

    # -- one layer ------------------------------------------------------------

    def _state_group(self, n: int, x: Any, reset: Any, s0: Any):
        """The state over C chunks of t tokens (torch path): x `(B, C, t, dim)`, reset `(B, C, t)`, from s0
        `(B, dim)`. Returns s `(B, C, t, dim)` and the state after the last chunk."""
        B, C, t, D = x.shape
        cos, sin = self._rope(t, x)
        lam = self._decay_consts(self._w("decay_state", n).value, t)
        seg = torch.cumsum(reset.to(torch.int32), -1)  # document index inside the chunk
        live = (seg == 0).to(x.dtype)  # no marker yet: the carry still counts
        same = (seg[..., :, None] == seg[..., None, :]).to(x.dtype)  # (B, C, t, t)
        # s_i = gamma R_i sum_j lambda^(i - j) [same doc] R_{-j} x_j  +  live_i lambda^(i + 1) R_{i + 1} s_prev
        u = _rot(x, cos[:t], -sin[:t])
        s_in = _rot((lam["mat"] * same) @ u, cos[:t], sin[:t]) * lam["lru"]
        a = live * lam["next"]  # (B, C, t)
        prev, s_prev = [], s0
        for c in range(C):
            prev.append(s_prev)
            s_prev = s_in[:, c, -1] + a[:, c, -1, None] * _rot(s_prev, cos[t], sin[t])
        return s_in + a[..., None] * _rot(torch.stack(prev, 1)[:, :, None], cos[1:], sin[1:]), s_prev

    def _state_torch(self, n: int, x: Any, reset: Any, s0: Any):
        """The state over a row `(B, N, dim)` in chunks of `chunk_size` (whole chunks batched, then the rest)."""
        B, N, D = x.shape
        T = min(self.chunk_size, N)
        main, parts = (N // T) * T, []
        for a, b in ((0, main), (main, N)):
            if b > a:
                t = min(T, b - a)
                sg, s0 = self._state_group(n, x[:, a:b].reshape(B, -1, t, D), reset[:, a:b].reshape(B, -1, t), s0)
                parts.append(sg.reshape(B, -1, D))
        return (torch.cat(parts, 1) if len(parts) > 1 else parts[0]), s0

    @staticmethod
    def _memory_torch(q: Any, k: Any, v: Any, mu: Any, reset: Any, m0: Any):
        """The memory token by token (torch path): M_t[a] = mu_a M_{t-1}[a] keep_t + k_t[a] v_t, o_t = M_t^T q_t."""
        m, out = m0, []
        keep = (~reset).to(v.dtype)
        for t in range(v.shape[1]):
            m = (mu[:, None] * keep[:, t, None, None]) * m + k[:, t, :, None] * v[:, t, None, :]
            out.append(torch.einsum("ba,bad->bd", q[:, t], m))
        return torch.stack(out, 1), m

    @staticmethod
    def _delta_group(q: Any, k: Any, v: Any, beta: Any, a: Any, reset: Any, z: Any):
        """The delta-rule banks over C chunks of t tokens: q, k `(B, C, t, S)` (unit length), v `(B, C, t, D)`, beta
        `(B, C, t)`, a `(S,)` the decay rates (mu = e^{-a}), reset `(B, C, t)`, z `(B, S, D)` the state before.
        Returns the reads `(B, C, t, D)` and the state after the last chunk.

        Local index i in a chunk, the step: Z_i = D Z_{i-1} + k_i e_i^T, e_i = beta_i (v_i - k_i^T D Z_{i-1}).
        Unrolled from the chunk's start state z: k_i^T D Z_{i-1} = k_i^T D^(i+1) z + sum_{j<i} A_ij e_j with
        A_ij = k_i^T D^(i-j) k_j = (k_i mu^i) . (k_j mu^-j), so (I + diag(beta) A) e = diag(beta) (V - K' z)
        (K' rows k_i mu^(i+1)), lower unit-triangular. Documents: no term crosses a marker (A, the reads and
        the carry are masked by document)."""
        B, C, t, S = k.shape
        pos = torch.arange(t, device=k.device, dtype=k.dtype)[:, None]
        e_pos, e_neg = torch.exp(-a * pos), torch.exp(a * pos)  # mu^i, mu^-i: (t, S)
        e_pos1 = e_pos * torch.exp(-a)  # mu^(i + 1)
        e_end = torch.exp(-a * (t - 1 - pos))  # mu^(t - 1 - j)
        seg = torch.cumsum(reset.to(torch.int32), -1)
        live = (seg == 0).to(k.dtype)  # no marker yet in the chunk: the carried state still counts
        same = seg[..., :, None] == seg[..., None, :]
        tri = torch.ones(t, t, device=k.device, dtype=torch.bool)
        kn = k * e_neg
        A = ((k * e_pos) @ kn.transpose(-1, -2)) * (same & tri.tril(-1))
        R = ((q * e_pos) @ kn.transpose(-1, -2)) * (same & tri.tril())  # read weights of the errors
        T = torch.eye(t, device=k.device, dtype=k.dtype) + beta[..., None] * A
        bl = beta[..., None]
        ev = torch.linalg.solve_triangular(T, bl * v, upper=False, unitriangular=True)  # (B, C, t, D)
        ez = torch.linalg.solve_triangular(T, bl * live[..., None] * (k * e_pos1), upper=False, unitriangular=True)
        q_carry = q * e_pos1 * live[..., None]
        w_end = (k * e_end * (seg == seg[..., -1:])[..., None]).transpose(-1, -2)  # (B, C, S, t)
        keep_end = live[..., -1, None] * torch.exp(-a * t)  # (B, C, S)
        outs = []
        for c in range(C):  # only the state crosses chunks
            e = ev[:, c] - ez[:, c] @ z  # (B, t, D)
            outs.append(q_carry[:, c] @ z + R[:, c] @ e)
            z = keep_end[:, c, :, None] * z + w_end[:, c] @ e
        return torch.stack(outs, 1), z

    def _delta_memory(self, q: Any, k: Any, v: Any, beta: Any, a: Any, reset: Any, z: Any):
        """The delta-rule banks over a row `(B, N, ...)`: whole chunks batched, then the shorter last one."""
        B, N, S = k.shape
        K = min(self.delta_chunk, N)
        main, outs = (N // K) * K, []
        for lo, hi in ((0, main), (main, N)):
            if hi > lo:
                t = min(K, hi - lo)
                f = lambda x: x[:, lo:hi].reshape((B, -1, t) + tuple(x.shape[2:]))
                o, z = self._delta_group(f(q), f(k), f(v), f(beta), a, f(reset), z)
                outs.append(o.reshape(B, -1, v.shape[-1]))
        return (torch.cat(outs, 1) if len(outs) > 1 else outs[0]), z

    def _addresses(self, n: int, read: Any, write: Any):
        """From the read / write addresses `(B, N, address_dim)`: what the memory scan takes, (q, k, mu): the read
        weights, the write weights `(B, N, memory_rows)` and the decay of each memory row `(memory_rows,)`."""
        mu = self.decays(n)[1]
        if self.memory == "linear":
            c = torch.sqrt(1 - mu * mu)  # LRU normalization of the memory, folded into the query
            q = torch.nn.functional.normalize(read, dim=-1, eps=1e-6) * c  # unit length: |q . k| <= 1, stable
            k = torch.nn.functional.normalize(write, dim=-1, eps=1e-6)
            return q, k, mu.reshape(1).expand(self.key_dim).contiguous()
        r, w = torch.softmax(read, -1), torch.softmax(write, -1)
        return r, w * (1 - mu), mu  # (1 - mu_b): each bank a running average of its writes

    def _norm(self, layer: Any, x: Any) -> Any:
        """RMSNorm of `x` `(..., dim)` with `layer`'s scale (a fused kernel pass on the XPU)."""
        scale = layer.scale.value
        if self.use_kernels and rsm_scan.rms_ok(x, x.shape[-1]):
            return rsm_scan.rms(x, scale)
        return _rms(x, scale)

    def _gate(self, n: int, q: Any) -> Any:
        """The memory read's weight g `(..., 1)` from the gate input q (`read_gate_in`): sigmoid(w_g . q + b_g)
        (`read_mix` "add" / "gates") or (1 + tanh(w_g . q + b_g)) / 2 (`"tanh"`: h = (1 - g) s + g o)."""
        z = q @ self._w("read_gate_w", n).value + self._w("read_gate_b", n).value
        return 0.5 * (1 + torch.tanh(z)) if self.read_mix == "tanh" else torch.sigmoid(z)

    def _keep_state(self, n: int, q: Any) -> Any:
        """The state's weight a `(..., 1)` in h (`read_mix="gates"`): sigmoid(w_a . q + b_a)."""
        return torch.sigmoid(q @ self._w("keep_state_w", n).value + self._w("keep_state_b", n).value)

    def _join(self, n: int, s: Any, o: Any, read: Any) -> Tuple[Any, Any]:
        """h from the state s and the memory read o (see `read_mix`), and the read as it enters h (g o)."""
        gin = self.read_gate_in(s, read)
        g = self._gate(n, gin)
        go = o * g
        if self.read_mix == "gates":
            return self._keep_state(n, gin) * s + go, go
        return (s + go if self.read_mix == "add" else s - g * s + go), go

    def _write_gate(self, n: int, a: Any) -> Any:
        """Write gate sigmoid(w_w . a + b_w) `(..., 1)` from `a`: the state before the update `(..., dim)` (0 after a
        marker), or the write logits `(..., address_dim)` with `write_gate_input="logits"`."""
        return torch.sigmoid(a @ self._w("write_gate_w", n).value + self._w("write_gate_b", n).value)

    def _out_gates(self, n: int, m: Any) -> Tuple[Any, Any]:
        """`mlp_residual="gates"`: (keep the input a2, take the MLP g2) `(..., 1)` each, from the MLP output m."""
        if self.use_kernels and rsm_scan.rms_ok(m, m.shape[-1]):  # one fused pass each way
            mn = rsm_scan.rms(m, torch.ones(m.shape[-1], device=m.device, dtype=m.dtype))
        else:
            mn = _rms(m, 1.0)
        return (torch.sigmoid(mn @ self._w("keep_input_w", n).value + self._w("keep_input_b", n).value),
                torch.sigmoid(mn @ self._w("mlp_gate_w", n).value + self._w("mlp_gate_b", n).value))

    def _kernels_ok(self, x: Any, fresh: bool) -> bool:
        """The custom scans run on the XPU for these widths; their decay gradients assume a zero start state."""
        # any even width: the state scan is one lane per RoPE pair, the memory keeps its rows (banks / key_dim) on
        # the scan's register-tiled axis (the backward of wide values sums over columns, `outer_scan_cols`)
        return (self.use_kernels and self.dim % 2 == 0 and self.memory_rows in rsm_scan.ROWS
                and self.memory_cols % 2 == 0 and (fresh or not torch.is_grad_enabled()) and rsm_scan.available(x))

    def _layer(self, n: int, x: Any, reset: Any, carry: Tuple[Any, Any], collect: bool = False, fresh: bool = False):
        """Layer n over a row `(B, N, dim)` (the normalized input u): h `(B, N, dim)` (the state and the memory read
        joined, see `read_mix`), the carries (s, memory) after it, (s, g o) if `collect`. `fresh`: the carries are zeros (a row that starts the scan)."""
        s0, m0 = carry
        # training from a carried state (truncated BPTT): the kernels scan from zeros (their decay gradients assume
        # it) and the carry's decayed contribution is added in torch, where autograd gives its decay gradients
        inject = not fresh and torch.is_grad_enabled()
        kfresh = fresh or inject
        kern = self._kernels_ok(x, kfresh)
        r8 = reset.to(torch.uint8).contiguous()
        live = (torch.cumsum(reset.to(torch.int32), 1) == 0).to(x.dtype)  # (B, N): no marker yet, the carry counts
        N = x.shape[1]
        if kern:
            cos, sin = self._rope(1, x)
            alpha = torch.nn.functional.softplus(self._w("decay_state", n).value)
            s, s_end = rsm_scan.state_scan(x, alpha, r8, cos[1].contiguous(), sin[1].contiguous(),
                                           None if kfresh else s0)
            if inject:
                c, sn = self._rope(N, x)
                s = _CarryState.apply(s, s0, live, alpha, c, sn)
                s_end = s[:, -1]
        else:
            s, s_end = self._state_torch(n, x, reset, s0)
        read = s @ self._w("address_q", n).value
        # what reads the state before each update (0 right after a marker): the write logits, and the write gate /
        # write strength when they read the state. All linear in it, so the state is projected first and the narrow
        # projections are shifted by one step, instead of shifting the dim-wide state
        cols = [self._w("address_k", n).value]
        if self.memory == "banks_delta":
            cols.append(self._w("write_strength_w", n).value)
        elif self.write_gate and self.write_gate_input == "state":
            cols.append(self._w("write_gate_w", n).value)
        Wp = torch.cat(cols, 1) if len(cols) > 1 else cols[0]
        proj = torch.cat([(s0 @ Wp)[:, None], (s @ Wp)[:, :-1]], 1)  # project every state, then shift the narrow result
        proj = proj * (~reset)[..., None].to(proj.dtype)  # 0 after a marker (a fresh document)
        S_ = self.address_dim
        write = proj[..., :S_]
        v = None if self.memory == "banks" else x @ self._w("value", n).kernel.value
        if self.memory == "banks_delta":
            q = torch.nn.functional.normalize(torch.softmax(read, -1), dim=-1)
            k = torch.nn.functional.normalize(torch.softmax(write, -1), dim=-1)
            beta = torch.sigmoid(proj[..., S_:S_ + 1] + self._w("write_strength_b", n).value)[..., 0]
            # decay rate a = -ln mu, capped so mu^-i stays finite in float32 inside a chunk: a * chunk <= 80
            # (chunk 64: mu >= 0.29, a half-life of at least 0.55 tokens)
            a = torch.nn.functional.softplus(self._w("decay_banks", n).value).clamp_max(80.0 / self.delta_chunk)
            o, m_end = self._delta_memory(q, k, v, beta, a, reset, m0)
            h, o = self._join(n, s, o, read)
            return h, (s_end, m_end), ((s, o) if collect else None)
        q, k, mu = self._addresses(n, read, write)
        if self.write_gate:  # softmax(key) * sigmoid(gate): the token's whole write (values and mass) attenuated
            if self.write_gate_input == "state":
                k = k * torch.sigmoid(proj[..., S_:S_ + 1] + self._w("write_gate_b", n).value)
            else:
                k = k * self._write_gate(n, write)
        vm = v
        if self.memory == "banks":  # a ones column carries the written mass z: the values come out padded, one matmul
            Wv = self._w("value", n).kernel.value
            Wpad = torch.cat([Wv, Wv.new_zeros(self.dim, self.memory_cols - self.dim)], 1)
            ones = self.__dict__.setdefault("_mass_col", {})
            key = (str(Wv.device), self.memory_cols)
            if key not in ones:
                ones[key] = torch.zeros(self.memory_cols, device=Wv.device, dtype=Wv.dtype)
                ones[key][self.dim] = 1.0
            vm = torch.addmm(ones[key], x.reshape(-1, self.dim), Wpad).reshape(x.shape[:-1] + (self.memory_cols,))
        if kern:
            o, m_end = rsm_scan.memory_scan(q, k, vm, mu, r8, None if kfresh else m0)
            if inject:  # o_t += q_t . (mu^(t + 1) M_0), the memory's end gets mu^N M_0 (while no marker was seen)
                pos = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
                decay = torch.exp(-pos[:, None] * (-torch.log(mu))[None])  # (N, rows)
                o = o + torch.matmul(q * decay * live[..., None], m0)
                m_end = m_end + live[:, -1, None, None] * (mu ** N)[None, :, None] * m0
        else:
            o, m_end = self._memory_torch(q, k, vm, mu, reset, m0)
        if (self.memory == "banks" and kern and not collect and self.read_mix in ("add", "gates")
                and rsm_scan.join_ok(o, self.dim, self.memory_cols)):
            # h = a s + g o with o = the values' sum / the mass: one fused pass each way
            gin = self.read_gate_in(s, read)
            a = self._keep_state(n, gin) if self.read_mix == "gates" else None
            return rsm_scan.join_read(s, o, self._gate(n, gin), a), (s_end, m_end), None
        if self.memory == "banks":  # a weighted average of past values
            o = o[..., :self.dim] / o[..., self.dim:self.dim + 1].clamp_min(1e-12)
        h, o = self._join(n, s, o, read)
        return h, (s_end, m_end), ((s, o) if collect else None)

    def _mlp(self, n: int, h: Any, skip: Any) -> Any:
        """The layer output: y = skip + MLP(RMSNorm(h)), skip = the layer input x (`mlp_residual="input"`) or h
        (`"state"`); `"gates"` (skip = x): y = a2 x + g2 m, a2 / g2 = sigmoid of two heads on RMSNorm(m); `"mix"` (skip = x), y = (1 - c)/2 x + (1 + c)/2 m with m = MLP(RMSNorm(h)) and
        c = tanh(w_m . RMSNorm(m) + b_m). The MLP's hidden layer is recomputed in the backward pass. On the XPU the norm, bias and
        GELU, forward and backward, are fused passes (`rsm_scan.norm_mlp`)."""
        w1, w2, scale = self._w("mlp_in", n), self._w("mlp_out", n), self._w("mlp_norm", n).scale.value
        args = (w1.kernel.value, w1.bias.value, w2.kernel.value, w2.bias.value)
        x = h.reshape(-1, h.shape[-1])
        if self.use_kernels and rsm_scan.norm_mlp_ok(x, self.dim, self.mlp_hidden, self.dim, False):
            y = rsm_scan.norm_mlp(x, scale, *args, keep_hidden=not self.mlp_recompute and torch.is_grad_enabled(),
                                 half=self.mlp_fp16)
        else:
            y = _RecomputeMLP.apply(_rms(x, scale), *args, "tanh", self.mlp_tf32)
        if self.mlp_residual == "gates":
            keep, take = self._out_gates(n, y)
            return (keep * skip.reshape(-1, skip.shape[-1]) + take * y).reshape(h.shape)
        if self.mlp_residual == "mix":
            c = torch.tanh(_rms(y, 1.0) @ self._w("mix_head_w", n).value + self._w("mix_head_b", n).value)  # (R, 1)
            sk = skip.reshape(-1, skip.shape[-1])
            return (0.5 * (sk + y) + 0.5 * c * (y - sk)).reshape(h.shape)  # -1 only the input, +1 only the MLP output
        return skip + y.reshape(h.shape)

    # -- forward --------------------------------------------------------------

    def _run(self, ids: Any, carries: Optional[List[Any]] = None, dense: bool = True, collect: bool = False,
             keep: Optional[Any] = None):
        """Runs `(B, N)` ids from `carries` (one (s, M) per layer, zeros if None).

        Returns (y `(B, N, dim)`, or `(B, 1, dim)` for the last position only if not `dense`, or `(R, dim)` at the
        R positions of the bool mask `keep` `(B, N)`; the carries after the row; per layer (s, o) `(B, N, dim)` if
        `collect`). The last layer's output feeds only the readout, so with `keep` its MLP and output mix run at
        those positions alone (every earlier layer runs everywhere: its output is the next layer's input)."""
        E = self.token_embedding.embeddings.value
        ids = torch.as_tensor(keras.ops.cast(ids, "int32")).to(E.device)
        B = int(ids.shape[0])
        reset = ids == self.reset_token
        # index_select, not embedding: the XPU embedding backward fails for 1025..2048 ids (work-group too large)
        x = E.index_select(0, ids.reshape(-1).long()).reshape(ids.shape + (E.shape[1],))
        fresh = carries is None
        if fresh:
            carries = [(E.new_zeros(B, self.dim), E.new_zeros(B, self.memory_rows, self.memory_cols))] * self.num_layers
        ends, trace = [], []
        for n in range(self.num_layers):
            u = self._norm(self._w("in_norm", n), x) if self.pre_norm else x
            h, end, st = self._layer(n, u, reset, carries[n], collect, fresh)
            ends.append(end)
            trace.append(st)
            if not dense and n == self.num_layers - 1:
                h, x = h[:, -1:], x[:, -1:]
            elif keep is not None and n == self.num_layers - 1:
                # the kept rows by index_select (its backward is index_add): the backward of boolean-mask indexing
                # (index_put with accumulate) launches work-groups larger than the Iris Xe allows at some shapes
                at = keep.reshape(-1).nonzero().squeeze(1)
                h, x = h.reshape(-1, h.shape[-1]).index_select(0, at), x.reshape(-1, x.shape[-1]).index_select(0, at)
            x = self._mlp(n, h, h if self.mlp_residual == "state" else x)
        return x, ends, trace

    def trace(self, ids: Any, carries: Optional[List[Any]] = None) -> List[Tuple[Any, Any]]:
        """Per layer, the state s and the memory read o after every position: `(B, N, dim)` each."""
        return self._run(ids, carries, collect=True)[2]

    def hidden(self, inputs: Any) -> Any:
        """Single target: y `(B, dim)` after the last token of each `(B, N)` input."""
        return self._run(inputs, dense=False)[0][:, 0]

    def dense_hidden(self, inputs: Any) -> Any:
        """Every position of `(B, N)` packed rows: y `(B, N, dim)` (target of i = token i + 1)."""
        return self._run(inputs)[0]

    def dense_hidden_valid(self, inputs: Any, valid: Any, carry: bool = False) -> Any:
        """y at the positions of `(B, N)` packed rows that have a target (`valid`, bool): `(R, dim)`, row-major; the
        last layer's MLP runs at those positions only. `carry` (truncated BPTT): row b continues row b of the previous
        call, from its detached end state; the state after this row is kept for the next call."""
        if not carry:
            return self._run(inputs, keep=valid)[0]
        prev = self.__dict__.get("_train_carry")
        if prev is not None and prev[0][0].shape[0] != inputs.shape[0]:
            prev = None
        y, ends, _ = self._run(inputs, prev, keep=valid)
        self._train_carry = [(s.detach(), m.detach()) for s, m in ends]
        return y

    def readout_input(self, h: Any) -> Any:
        """The final RMSNorm (dense training applies it to the valid rows only); identity without `pre_norm`."""
        return self._norm(self.final_norm, h) if self.pre_norm else h

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    # -- generation -----------------------------------------------------------

    def advance(self, ids: Any, state: Optional[Any] = None):
        """Feeds `(B, N)` ids after `state` (one (s, M) per layer): (logits of the last position, state after it)."""
        y, ends, _ = self._run(ids, state, dense=False)
        return self.readout(self.readout_input(y[:, 0])), ends

    def with_context_length(self, context_length: int) -> "RopeStateMemoryLM":
        """Finite context: a copy that generates and evaluates from the last `context_length` tokens only."""
        new = _context.with_context_length(self, context_length)
        new.stateful_generation = False
        return new

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RopeStateMemoryLM":
        # runs saved before the write gate existed have no `write_gate_bias`: they load without the gate; those
        # saved with the first gate (from the write logits) have no `write_gate_input`
        config = {"write_gate_bias": None, **config}
        if config["write_gate_bias"] is not None:
            config = {"write_gate_input": "logits", **config}
        # runs saved before `read_mix` / `mlp_residual="mix"` existed: the read was added (`mlp_residual` was saved);
        # before `read_gate_input`: the read gates read the logits
        config = {"read_mix": "add", "mlp_residual": "input", "read_gate_input": "logits", "tie_embeddings": False,
                  **config}
        return super().from_config(config)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "window_size": self.window_size, "key_dim": self.key_dim,
                "mlp_hidden": self.mlp_hidden, "chunk_size": self.chunk_size, "reset_token": self.reset_token,
                "num_layers": self.num_layers, "rope_base": self.rope_base, "state_half_life": self.state_half_life,
                "memory_half_life": self.memory_half_life, "read_gate_bias": self.read_gate_bias,
                "write_gate_bias": self.write_gate_bias, "write_gate_input": self.write_gate_input,
                "memory": self.memory, "num_banks": self.num_banks, "delta_chunk": self.delta_chunk, "memory_half_life_min": self.memory_half_life_min,
                "memory_half_life_max": self.memory_half_life_max,
                "mlp_residual": self.mlp_residual, "read_mix": self.read_mix, "mlp_mix_bias": self.mlp_mix_bias,
                "keep_gate_bias": self.keep_gate_bias, "read_gate_input": self.read_gate_input,
                "tie_embeddings": self.tie_embeddings, "mlp_recompute": self.mlp_recompute, "mlp_fp16": self.mlp_fp16,
                "embedding_std": self.embedding_std,
                "pre_norm": self.pre_norm,
                "mlp_tf32": self.mlp_tf32}


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))
