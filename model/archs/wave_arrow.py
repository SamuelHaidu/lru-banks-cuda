"""`wave-arrow`: tokens write bounded complex arrows into L state spaces; the state goes straight to an MLP.

Every complex number is a bounded arrow by construction: length in (0, 1) (sigmoid or softmax), angle =
pi * tanh(raw) in (-pi, pi). So every arrow on the way to the state lies inside the unit circle.

    token        s_{v,k}     = sigmoid(amp_{v,k}) e^{i pi tanh(phase_{v,k})}            64 arrows per token
    projection   u_{v,l,k}   = sum_j P_{l,k,j} s_{v,j}                                     per state space l
                 P_{l,k,j}   = softmax_j(proj_len_{l,k,:}) e^{i pi tanh(proj_ang_{l,k,j})}
                 Each output arrow is a weighted average (weights >= 0, summing to 1) of the token's arrows, each
                 turned by its own angle, so |u| < 1. Starts as the identity (output k = input k). A per-channel
                 gain or turn after it would be redundant: a turn is a shift of the row's angles, and the rows can
                 already shrink (turned arrows partly cancel) and gamma scales each channel.
    normalize    w_{v,l,k}   = gamma_{l,k} u_{v,l,k},  gamma_{l,k} = sigmoid(space_gamma_{l,k})
                 LRU input normalization (Orvieto et al. 2023): a channel that keeps |lambda| per step sums its
                 writes to about 1 / sqrt(1 - |lambda|^2) times their size, so gamma starts at sqrt(1 - |lambda|^2)
                 and every channel's state starts at about the size of one write. Learned from there (sigmoid
                 keeps it in (0, 1), so it only shrinks and every write stays inside the unit circle).
    state        S_{l,k}(i)  = sum_{b <= j <= i} w_{tok j, l, k} lambda_{l,k}^{i - j},  lambda = e^{-alpha_{l,k} + i omega_k}
                 The recurrence S <- lambda S + w with no window: the state covers the whole document. It resets
                 at the document marker (`reset_token`, position b): the marker is the first write of a fresh
                 state. Decay learned per channel, frequency per channel shared by the spaces. Not normalized.
    read         [Re S ; Im S] of all spaces (2 L K numbers) -> MLP -> `dim` (128) -> logits.

Layers (`num_layers` > 1). Layer n + 1 reads space 1 of layer n instead of tokens: its "token" at position i is that
state S_{n,1}(i), squashed to length < 1 with the phase kept (z tanh|z| / |z|, so its writes stay inside the unit
circle like token arrows), then the same projection, gamma, state spaces (own decay and frequency) and resets as
layer 1. Each layer has its own MLP to `dim`; the outputs add like residual branches (h = sum of the layers'
MLP outputs) before the shared readout, so the logits are a sum over layers. Layer 1 keeps its parameter names, so
`num_layers=1` is the single-layer model.

Infinite context. Training reads rows of `window_size` tokens packed from consecutive documents (each row starts
at a document start, `model.data.PackedDocumentDataset`) and predicts every next token. The row is run in chunks
of `chunk_size` tokens. Inside a chunk, every layer's state at every position is one (T x T) Toeplitz matmul per
channel plus the carried state lambda^(t+1) h; resets are exact by subtraction, S_doc(i) = S(i) - lambda^(i-b+1)
S(b-1), b the last marker in the chunk (S(-1) = h). Then the layer's MLP, then the next layer from its space 1.
A chunk hands on only its end states (the carries) and its `dim` outputs. Each MLP keeps only its 256-wide input for
the backward pass and recomputes its 2048-wide hidden layer there (`_RecomputeMLP`): that hidden layer, stored twice
per position by plain autograd, was most of the activation memory.
`window_size` is only the training row length (and the validation window); nothing in the model depends on it but
the decay init. Generation runs the recurrence one token at a time (O(1) per token, unbounded context);
`with_context_length(n)` gives a copy that generates and evaluates from the last n tokens only (finite context).

The token table w (V x 2 L K) is built once per forward; positions gather from it.
"""

from typing import Any, Dict, Optional, Tuple

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

import torch  # noqa: E402

from model.archs import _context  # noqa: E402
from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.archs.rope_token_banks import _const  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


class _tf32:
    """TF32 tensor-core matmuls inside the block, on CUDA only (no effect on other devices)."""

    def __init__(self, on: bool, like: torch.Tensor):
        self.on = bool(on) and like.is_cuda

    def __enter__(self):
        if self.on:
            self.prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *exc):
        if self.on:
            torch.backends.cuda.matmul.allow_tf32 = self.prev


class _RecomputeMLP(torch.autograd.Function):
    """y = gelu(x W1 + b1) W2 + b2 that keeps only `x` for the backward pass.

    The hidden layer (`mlp_hidden` wide, 8x the input) is what makes a plain MLP expensive in memory: autograd
    stores it twice per position (before and after the GELU). Here the backward pass recomputes it from `x`
    (one extra x W1 matmul), a block of `rows` positions at a time, so it is never stored and never whole.
    The products use the layouts the XPU multiplies fastest (a bias-free `mm` where the bias is tiny, the weight
    gradients as `(dpre^T x)^T`, plain `mm` + add instead of the accumulating `addmm_`).
    """

    rows = 16384

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
                approximate: str, tf32: bool):  # type: ignore[override]
        ctx.save_for_backward(x, w1, b1, w2)
        ctx.approximate, ctx.tf32 = approximate, tf32
        out = x.new_empty(x.shape[0], w2.shape[1])
        with _tf32(tf32, x):
            w2t = w2.T.contiguous()
            for a in range(0, x.shape[0], _RecomputeMLP.rows):
                h = torch.nn.functional.gelu(torch.addmm(b1, x[a:a + _RecomputeMLP.rows], w1), approximate=approximate)
                out[a:a + _RecomputeMLP.rows] = torch.nn.functional.linear(h, w2t)
            out += b2
        return out

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):  # type: ignore[override]
        x, w1, b1, w2 = ctx.saved_tensors
        dx, dw1t, dw2, db1 = torch.empty_like(x), None, None, None
        with _tf32(ctx.tf32, x):
            for a in range(0, x.shape[0], _RecomputeMLP.rows):
                xb, dyb = x[a:a + _RecomputeMLP.rows], dy[a:a + _RecomputeMLP.rows]
                pre = torch.addmm(b1, xb, w1)
                g2 = torch.nn.functional.gelu(pre, approximate=ctx.approximate).T @ dyb
                dpre = torch.ops.aten.gelu_backward(dyb @ w2.T, pre, approximate=ctx.approximate)
                g1t, gb1 = dpre.T @ xb, dpre.sum(0)  # dW1^T (H, D): the faster layout
                torch.mm(dpre, w1.T, out=dx[a:a + _RecomputeMLP.rows])
                dw1t, dw2, db1 = (g1t, g2, gb1) if dw2 is None else (dw1t + g1t, dw2 + g2, db1 + gb1)
        return dx, dw1t.T, db1, dw2, dy.sum(0), None, None


@keras.saving.register_keras_serializable(package="model")
class WaveArrowLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        dim: Width of the MLP output the readout reads.
        window_size: Training row length (packed documents) and validation window. The state itself has no window.
        channels: Complex channels K per token and per state space.
        state_spaces: State spaces L (decay initialized per space from short to long, learned per channel).
        mlp_hidden: Hidden width of the MLP between the state and the readout.
        chunk_size: Scan chunk length (speed and memory only, the result is the same for any value).
        reset_token: Document marker id; the state restarts at it (the dataset's `doc_token_id`).
        num_layers: Stacked layers; layer n + 1 writes from space 1 of layer n, every layer's MLP adds to the logits.
        mlp_gelu: "tanh" (default, the tanh approximation of GELU) or "exact" (erf). The two differ by about 1e-4
            relative (a model trained with erf scores the same loss with tanh to 4 decimals). On the Intel XPU erf
            is 3x slower than every other elementwise op and made up 40% of the MLP's time.
        mlp_tf32: Run the MLP matmuls with TF32 tensor cores on CUDA (about 1e-3 relative per product, fp32
            accumulation; the scan and the readout stay fp32). No effect on other devices; False for full fp32.
    """

    # lengths, angles, decays and frequencies: weight decay would pull them to a fixed length or angle 0
    no_weight_decay_names = ("amp_", "phase_", "proj_", "space_", "decay_", "channel_")
    # 64 rows x 1024 tokens: the same ~65k targets per step as dense supervision at 512 x 128
    config_defaults = {"training": {"learning_rate": 0.01, "warmup_steps": 200, "dense_supervision": True},
                       "data": {"batch_size": 64}}
    # training rows are packed documents with a target at every position (see `model.data.build_datasets`)
    packed_documents = True
    # `generate` runs the recurrence token by token over the whole text; False on finite-context copies
    stateful_generation = True

    def __init__(self, vocabulary_size: int = 10000, dim: int = 128, window_size: int = 1024, channels: int = 64,
                 state_spaces: int = 2, mlp_hidden: int = 2048, chunk_size: int = 128, reset_token: int = 0,
                 num_layers: int = 1, mlp_gelu: str = "tanh", mlp_tf32: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.state_spaces = int(state_spaces)
        self.mlp_hidden = int(mlp_hidden)
        self.chunk_size = int(chunk_size)
        self.reset_token = int(reset_token)
        self.num_layers = int(num_layers)
        if mlp_gelu not in ("exact", "tanh"):
            raise ValueError(f"mlp_gelu must be 'exact' or 'tanh', got {mlp_gelu!r}")
        self.mlp_gelu = mlp_gelu
        self.mlp_tf32 = bool(mlp_tf32)
        K, L, V = self.channels, self.state_spaces, self._vocabulary_size
        self.layer_width = 2 * L * K  # [Re S ; Im S] of one layer, the input of its MLP
        self.amp_raw = layers.Embedding(V, K, embeddings_initializer="zeros", name="amp_raw")  # length 0.5 at init
        self.phase_raw = layers.Embedding(V, K, name="phase_raw")  # angles set uniform on the circle in build()
        for n in range(self.num_layers):
            setattr(self, self._name("mlp_in", n), layers.Dense(self.mlp_hidden, activation="gelu", name=self._name("mlp_in", n)))
            # layers 2+ start silent (zero output weights): the model begins as the 1-layer model and a layer
            # grows only where it lowers the loss
            setattr(self, self._name("mlp_out", n), layers.Dense(
                self.dim, kernel_initializer="glorot_uniform" if n == 0 else "zeros", name=self._name("mlp_out", n)))
        self.readout = layers.Dense(V, use_bias=False, name="readout")
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
        K, L, T, V = self.channels, self.state_spaces, self.window_size, self._vocabulary_size
        self.amp_raw.build((None,))
        self.phase_raw.build((None,))
        # token angles uniform on the circle: phase = atanh(u), u ~ U(-0.99, 0.99) -> angle = pi u
        self.phase_raw.embeddings.assign(np.arctanh(np.random.default_rng().uniform(-0.99, 0.99, (V, K))).astype("float32"))
        self.readout.build((None, self.dim))
        alpha = np.log(2.0) / np.geomspace(1.0, T / 2, L)  # decay: half-lives from 1 token to half the window over the spaces
        gamma0 = np.sqrt(1.0 - np.exp(-2.0 * alpha))  # LRU normalization: sqrt(1 - |lambda|^2) at init, |lambda| = e^{-alpha}
        omega = np.geomspace(np.pi / 2, 2 * np.pi / (2 * T), K)  # fast (a few tokens per turn) .. slow (window-long)
        for n in range(self.num_layers):  # the same init in every layer
            getattr(self, self._name("mlp_in", n)).build((None, self.layer_width))
            getattr(self, self._name("mlp_out", n)).build((None, self.mlp_hidden))
            weight = lambda name, shape, init: setattr(self, self._name(name, n), self.add_weight(
                name=self._name(name, n), shape=shape, initializer=init))
            # projection per space: row weights softmax(proj_len), identity at init (diagonal 10 -> weight ~0.997)
            weight("proj_len", (L, K, K), _const(np.tile(10.0 * np.eye(K)[None], (L, 1, 1))))
            weight("proj_ang", (L, K, K), "zeros")
            # decay learned per channel from the per-space init; gamma stored as a sigmoid logit
            weight("decay_raw", (L, K), _const(np.tile(np.log(np.expm1(alpha))[:, None], (1, K))))
            weight("space_gamma", (L, K), _const(np.tile(np.log(gamma0 / (1.0 - gamma0))[:, None], (1, K))))
            weight("channel_omega", (K,), _const(omega))
        self.built = True

    @staticmethod
    def _name(name: str, n: int) -> str:
        """Parameter name in layer n (0-based): layer 1 keeps the single-layer names."""
        return name if n == 0 else f"{name}_l{n + 1}"

    def _w(self, name: str, n: int) -> Any:
        return getattr(self, self._name(name, n))

    # -- write ----------------------------------------------------------------

    def _projection(self, n: int = 0) -> Any:
        """gamma (P z) of layer n as one real matrix `(2 K, 2 L K)`: [Re z ; Im z] @ W = [Re w ; Im w].

        P (L, K out, K in) = averaging weights times bounded turns; gamma (LRU normalization per space and
        channel) is folded into the columns."""
        ops = keras.ops
        K, C = self.channels, self.state_spaces * self.channels
        p_len, p_ang = ops.softmax(self._w("proj_len", n), axis=-1), np.pi * ops.tanh(self._w("proj_ang", n))
        g = ops.sigmoid(self._w("space_gamma", n))[..., None]  # (L, K out, 1)
        pr = (g * p_len * ops.cos(p_ang)).reshape(C, K).T  # (K in, L K out)
        pi = (g * p_len * ops.sin(p_ang)).reshape(C, K).T
        return torch.cat([torch.cat([pr, pi], 1), torch.cat([-pi, pr], 1)], 0)

    def _table(self) -> Any:
        """Every token's writes `(V, 2 L K)` ([real ; imaginary]), all inside the unit circle."""
        ops = keras.ops
        # token arrows s (V, K)
        s_len, s_ang = ops.sigmoid(self.amp_raw.embeddings), np.pi * ops.tanh(self.phase_raw.embeddings)
        return torch.cat([s_len * torch.cos(s_ang), s_len * torch.sin(s_ang)], -1) @ self._projection(0)

    def write_table(self):
        """What every token writes into every state space: real and imaginary `(V, L, K)`."""
        V, L, K, C = self._vocabulary_size, self.state_spaces, self.channels, self.state_spaces * self.channels
        t = self._table()
        return t[:, :C].reshape(V, L, K), t[:, C:].reshape(V, L, K)

    def _writes(self, ids: Any):
        """Writes of every position: real and imaginary `(..., L, K)`."""
        L, K, C = self.state_spaces, self.channels, self.state_spaces * self.channels
        table = self._table()
        w = torch.nn.functional.embedding(self._ids(ids, table), table)
        shape = tuple(w.shape[:-1]) + (L, K)
        return w[..., :C].reshape(shape), w[..., C:].reshape(shape)

    @staticmethod
    def _layer_writes(zr: Any, zi: Any, W: Any):
        """Writes `(B, T, L K)` of a layer from space 1 of the layer below `(B, T, K)` and its projection `W`.
        The state is squashed to length tanh|z| < 1 (phase kept), so the writes stay inside the unit circle."""
        r = torch.sqrt(zr * zr + zi * zi + 1e-12)
        f = torch.tanh(r) / r
        w = torch.cat([zr * f, zi * f], -1) @ W
        C = W.shape[1] // 2
        return w[..., :C], w[..., C:]

    # -- recurrence -------------------------------------------------------------

    def _powers(self, n: Any, layer: int = 0):
        """lambda^n of every channel of a layer: real and imaginary `(*n.shape, L K)`; `n` a float tensor of any shape."""
        ops = keras.ops
        alpha = ops.reshape(ops.softplus(self._w("decay_raw", layer)), (-1,))  # (L K,)
        omega = ops.tile(self._w("channel_omega", layer), (self.state_spaces,))  # (L K,), frequency shared by the spaces
        n = n[..., None]
        mag = torch.exp(-alpha * n)
        return mag * torch.cos(omega * n), mag * torch.sin(omega * n)

    def _kernel(self, layer: int, T: int, like: Any):
        """A layer's chunk constants: the complex Toeplitz matrix lambda^(t - j) (causal) as one real block matrix
        `(L K, 2T, 2T)` = [[Re, -Im], [Im, Re]], and lambda^(t + 1) `(T, L K)` (real and imaginary)."""
        t = torch.arange(T, device=like.device, dtype=like.dtype)
        d = t[:, None] - t[None, :]
        causal = (d >= 0).to(like.dtype)[..., None]
        mr, mi = (m * causal for m in self._powers(d.clamp_min(0), layer))  # (T, T, C)
        mr, mi = mr.permute(2, 0, 1), mi.permute(2, 0, 1)  # (C, T, T)
        big = torch.cat([torch.cat([mr, -mi], -1), torch.cat([mi, mr], -1)], 1)
        pr, pi = self._powers(t + 1, layer)
        return big, pr, pi

    @staticmethod
    def _chunk_scan(wr: Any, wi: Any, reset: Any, h: Tuple[Any, Any], kernel: Tuple[Any, ...]):
        """One layer over one chunk: the state `(B, t, L K)` (real, imaginary) at every position from the writes
        `(B, t, L K)`, the reset flags `(B, t)` and the state `h` before the chunk `(B, L K)`. `kernel` is
        `_kernel`'s, for the full chunk length T; a shorter chunk (t < T) uses its top-left part."""
        big, pr, pi = kernel
        B, t, C = wr.shape
        T = pr.shape[0]
        hr, hi = h
        if t < T:  # the leading t rows and columns of both blocks
            keep_idx = torch.cat([torch.arange(t), T + torch.arange(t)]).to(wr.device)
            big = big[:, keep_idx][:, :, keep_idx]
            pr, pi = pr[:t], pi[:t]
        # S[t] = sum_{j <= t} lambda^(t - j) w[j] + lambda^(t + 1) h: one block matmul per channel gives the sum
        out = big @ torch.cat([wr.permute(2, 1, 0), wi.permute(2, 1, 0)], dim=1)  # (C, 2t, B)
        sr = out[:, :t].permute(2, 1, 0) + pr * hr[:, None] - pi * hi[:, None]
        si = out[:, t:].permute(2, 1, 0) + pr * hi[:, None] + pi * hr[:, None]
        # resets: S_doc(i) = S(i) - lambda^(i - b + 1) S(b - 1), b = the last marker of the chunk at or before i
        pos = torch.arange(t, device=wr.device)
        last = torch.cummax(torch.where(reset, pos, -1), dim=1).values  # (B, t); -1: none in this chunk
        prev = (last - 1).clamp_min(0)[..., None].expand(B, t, C)  # S(b - 1); at b = 0 it is h
        at_start = (last == 0)[..., None]
        br = torch.where(at_start, hr[:, None], torch.gather(sr, 1, prev))
        bi = torch.where(at_start, hi[:, None], torch.gather(si, 1, prev))
        live = (last >= 0)[..., None].to(sr.dtype)
        k = (pos - last).clamp(0, t - 1)  # lambda^(i - b + 1) = pr[i - b]
        qr, qi = pr[k] * live, pi[k] * live
        return sr - (qr * br - qi * bi), si - (qr * bi + qi * br)

    def _mlp(self, n: int, sr: Any, si: Any) -> Any:
        """Layer n's MLP on its state [Re S ; Im S]: `(..., dim)`. The same function as the two Dense layers
        (gelu in between), but the backward pass recomputes the hidden layer instead of storing it."""
        x = torch.cat([sr, si], dim=-1)
        w1, w2 = self._w("mlp_in", n), self._w("mlp_out", n)
        y = _RecomputeMLP.apply(x.reshape(-1, x.shape[-1]), w1.kernel.value, w1.bias.value, w2.kernel.value, w2.bias.value,
                                 "tanh" if self.mlp_gelu == "tanh" else "none", self.mlp_tf32)
        return y.reshape(x.shape[:-1] + (y.shape[-1],))

    def _block(self, ids: Any, table: Any, consts: Any, carries: Any, mlp: bool, collect: bool):
        """Every layer over one chunk `(B, T)` of ids, from the states before it (`carries`, one pair per layer).

        Returns (the sum of the layers' MLP outputs `(B, T, dim)` or None, the end states, the states at every
        position if `collect`). Only the end states and the `dim` outputs leave the chunk."""
        T, K, C = ids.shape[1], self.channels, self.state_spaces * self.channels
        reset = ids == self.reset_token
        w = torch.nn.functional.embedding(ids, table)
        wr, wi = w[..., :C], w[..., C:]
        y, ends, states = None, [], []
        for n in range(len(carries)):
            sr, si = self._chunk_scan(wr, wi, reset, carries[n], consts[n][:3])
            ends.append((sr[:, -1], si[:, -1]))
            if collect:
                states.append((sr, si))
            if mlp:
                yn = self._mlp(n, sr, si)
                y = yn if y is None else y + yn  # residual-like: every layer adds to the same `dim` vector
            if n + 1 < len(carries):
                wr, wi = self._layer_writes(sr[..., :K], si[..., :K], consts[n + 1][3])
        return y, ends, states

    @staticmethod
    def _ids(ids: Any, like: Any) -> Any:
        return torch.as_tensor(keras.ops.cast(ids, "int32")).to(like.device)

    def _run(self, ids: Any, carries: Optional[Any] = None, layers: Optional[int] = None, mlp: bool = False,
             collect: bool = False):
        """Runs `(B, N)` ids chunk by chunk through the first `layers` layers (all by default) from `carries`
        (one (real, imaginary) `(B, L K)` pair per layer, zeros if None).

        Returns (the summed MLP outputs `(B, N, dim)` if `mlp`, the end states, every layer's states
        `(B, N, L K)` if `collect`)."""
        layers = self.num_layers if layers is None else layers
        table = self._table()
        ids = self._ids(ids, table)
        B, N = int(ids.shape[0]), int(ids.shape[1])
        C, T = self.state_spaces * self.channels, min(self.chunk_size, int(ids.shape[1]))
        consts = [self._kernel(n, T, table) + ((self._projection(n) if n else None),) for n in range(layers)]
        if carries is None:
            carries = [(table.new_zeros(B, C), table.new_zeros(B, C)) for _ in range(layers)]
        carries = list(carries)[:layers]
        ys, states = [], [[] for _ in range(layers)]
        for s in range(0, N, T):
            y, carries, st = self._block(ids[:, s:s + T], table, consts, carries, mlp, collect)
            for n, pair in enumerate(st):
                states[n].append(pair)
            if mlp:
                ys.append(y)
        out = torch.cat(ys, dim=1) if mlp else None
        if collect:
            states = [(torch.cat([p[0] for p in st], 1), torch.cat([p[1] for p in st], 1)) for st in states]
        return out, carries, states

    def scan(self, ids: Any, state: Optional[Tuple[Any, Any]] = None):
        """Layer 1 state after every position of `(B, N)` ids: real and imaginary `(B, N, L K)`.

        Starts from `state` (real and imaginary `(B, L K)`, zeros if None) and restarts at every `reset_token`.
        Exact for any N: no window, `chunk_size` only splits the work.
        """
        return self._run(ids, None if state is None else [state], layers=1, collect=True)[2][0]

    def scan_layers(self, ids: Any, states: Optional[Any] = None):
        """Every layer's state after every position: a list of (real, imaginary) `(B, N, L K)`, one per layer."""
        return self._run(ids, states, collect=True)[2]

    # -- forward --------------------------------------------------------------

    def _head(self, ends: Any) -> Any:
        """Sum of the layers' MLPs on their end states: `(B, dim)`."""
        out = None
        for n, (sr, si) in enumerate(ends):
            y = self._mlp(n, sr, si)
            out = y if out is None else out + y
        return out

    def hidden(self, inputs: Any) -> Any:
        """Single target: the summed MLP outputs `(B, dim)` after the last token of each `(B, N)` input."""
        return self._head(self._run(inputs)[1])

    def dense_hidden(self, inputs: Any) -> Any:
        """Every position of `(B, N)` packed rows: the summed MLP outputs `(B, N, dim)` (target of i = token i + 1)."""
        return self._run(inputs, mlp=True)[0]

    def readout_input(self, h: Any) -> Any:
        """`hidden` / `dense_hidden` already end at the `dim` vector the readout reads."""
        return h

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.hidden(inputs))

    # -- generation -----------------------------------------------------------

    def advance(self, ids: Any, state: Optional[Any] = None):
        """Feeds `(B, N)` ids after `state` (one (real, imaginary) pair per layer): `(logits of the last
        position, state after it)`."""
        ends = self._run(ids, state)[1]
        return self.readout(self._head(ends)), ends

    def with_context_length(self, context_length: int) -> "WaveArrowLM":
        """Finite context: a copy that generates and evaluates from the last `context_length` tokens only."""
        new = _context.with_context_length(self, context_length)
        new.stateful_generation = False
        return new

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "window_size": self.window_size, "channels": self.channels, "state_spaces": self.state_spaces,
                "mlp_hidden": self.mlp_hidden, "chunk_size": self.chunk_size, "reset_token": self.reset_token,
                "num_layers": self.num_layers, "mlp_gelu": self.mlp_gelu, "mlp_tf32": self.mlp_tf32}
