"""`wave-field`: tokens write damped complex waves into L state spaces; a cascade reads them back.

Names:
    channel      one complex number of a state space. It turns at its frequency omega_k (channel k has the same
                 frequency in every state space) and fades at its own rate alpha_{l,k}.
    state space  K channels. There are L of them; every token writes into each its own projection of its signature.

Write. Token v has a signature s_v ([Re ; Im] of K complex numbers) and a mix p_v over the state spaces (softmax).
It writes w_{v,l} = p_{v,l} (s_v P_l) into state space l, with P_l a learned 2K x 2K map (identity at init). The
state at position i is everything written in the window before it, carried by the damped waves:

    S_{l,k}(i) = sum_{j <= i, i - j < T} w_{tok j, l, k} e^{(-alpha_{l,k} + i omega_k)(i - j)} / Z_{l,k}

(Z_{l,k} = the channel's decay summed over the window, so every channel starts at a comparable scale.) It is the
recurrence S <- e^{-alpha + i omega} S + w cut at the window; training computes the exact windowed sum for every
query at once with the block-banded matmul (dense supervision).

Read: a cascade from the fastest-fading state space to the slowest. The query q starts as the reader's signature.
For each state space l: filter g_l = 1 + head_l(RMSNorm q) (a complex gain per channel; zero init = hear
everything), reading r_l = g_l S_l, features [Re r_l, Im r_l, |r_l|]; then the reading updates the query through a
residual MLP on the concatenation, q += MLP_l([RMSNorm q ; RMSNorm features_l]) (zero-init output). The readings of
all state spaces are concatenated, RMS-normalized, and go through a linear map plus a residual GELU MLP to the
readout.
"""

from typing import Any, Dict, List, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

from model.archs._dense import DenseSupervisionMixin  # noqa: E402
from model.archs._norm import RMSNorm  # noqa: E402
from model.archs.rope_token_banks import _band_matmul, _const  # noqa: E402
from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class WaveFieldLM(DenseSupervisionMixin, BaseLM):
    """Args:
        vocabulary_size: Vocabulary size.
        dim: Width of the vector the readout reads.
        window_size: Tokens the state covers (the context length).
        channels: Complex channels K per state space (frequencies shared by the state spaces).
        state_spaces: State spaces L, from short to long memory (decay initialized per state space, learned per
            channel).
        reader_hidden: Hidden width of each query-update MLP of the cascade.
        mlp_hidden: Hidden width of the residual GELU MLP before the readout.
    """

    # raw decay and frequencies, the projections (decay would pull them from identity to 0) and the mixes
    no_weight_decay_names = ("channel_", "decay_", "space_")
    # training settings every run of this arch has used, written into config.json by `model create`
    config_defaults = {"training": {"learning_rate": 0.01, "warmup_steps": 200, "dense_supervision": True}}

    def __init__(self, vocabulary_size: int = 10000, dim: int = 128, window_size: int = 128, channels: int = 64,
                 state_spaces: int = 4, reader_hidden: int = 256, mlp_hidden: int = 1024, **kwargs: Any):
        super().__init__(**kwargs)
        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.state_spaces = int(state_spaces)
        self.reader_hidden = int(reader_hidden)
        self.mlp_hidden = int(mlp_hidden)
        K, L, V = self.channels, self.state_spaces, self._vocabulary_size
        self.feature_width = 3 * K * L  # [Re r, Im r, |r|] per state space
        # cascade reader: one filter head per state space, one query update after each but the last
        self.query_norm = [RMSNorm(name=f"query_norm_{l}") for l in range(L)]
        self.filter_head = [layers.Dense(2 * K, kernel_initializer="zeros", name=f"filter_head_{l}") for l in range(L)]
        self.heard_norm = [RMSNorm(name=f"heard_norm_{l}") for l in range(L - 1)]
        self.update_in = [layers.Dense(self.reader_hidden, activation="gelu", name=f"update_in_{l}") for l in range(L - 1)]
        self.update_out = [layers.Dense(2 * K, kernel_initializer="zeros", name=f"update_out_{l}") for l in range(L - 1)]
        self.signature = layers.Embedding(V, 2 * K, name="signature")  # [Re s ; Im s]
        self.space_mix = layers.Embedding(V, L, embeddings_initializer="zeros", name="space_mix")  # softmax: uniform at init
        self.feature_norm = RMSNorm(name="feature_norm")
        self.base = layers.Dense(self.dim, name="base")
        self.mlp_in = layers.Dense(self.mlp_hidden, activation="gelu", name="mlp_in")
        self.mlp_out = layers.Dense(self.dim, kernel_initializer="zeros", name="mlp_out")
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
        K, L, T = self.channels, self.state_spaces, self.window_size
        self.signature.build((None,))
        self.space_mix.build((None,))
        self.feature_norm.build((None, self.feature_width))
        self.base.build((None, self.feature_width))
        self.mlp_in.build((None, self.feature_width))
        self.mlp_out.build((None, self.mlp_hidden))
        self.readout.build((None, self.dim))
        for l in range(L):
            self.query_norm[l].build((None, 2 * K))
            self.filter_head[l].build((None, 2 * K))
        for l in range(L - 1):
            self.heard_norm[l].build((None, 3 * K))
            self.update_in[l].build((None, 2 * K + 3 * K))
            self.update_out[l].build((None, self.reader_hidden))
        # decay: half-lives from 1 token to half the window over the state spaces, then learned per channel
        alpha = np.log(2.0) / np.geomspace(1.0, T / 2, L)
        raw = np.tile(np.log(np.expm1(alpha))[:, None], (1, K))
        self.decay_raw = self.add_weight(name="decay_raw", shape=(L, K), initializer=_const(raw))
        omega = np.geomspace(np.pi / 2, 2 * np.pi / (2 * T), K)  # fast (a few tokens per turn) .. slow (window-long)
        self.channel_omega = self.add_weight(name="channel_omega", shape=(K,), initializer=_const(omega))
        self.space_proj = self.add_weight(name="space_proj", shape=(L, 2 * K, 2 * K),
                                          initializer=_const(np.tile(np.eye(2 * K)[None], (L, 1, 1))))
        self.built = True

    # -- write ----------------------------------------------------------------

    def _kernel(self, like: Any):
        """Damped waves over distance 0..T-1, normalized per channel: real and imaginary parts `(L, K, T)`."""
        ops = keras.ops
        dist = ops.convert_to_tensor(np.arange(self.window_size, dtype="float32"), dtype=like.dtype)
        decay = ops.exp(-ops.softplus(self.decay_raw)[..., None] * dist)  # (L, K, T)
        decay = decay / ops.sum(decay, axis=-1, keepdims=True)
        ang = self.channel_omega[:, None] * dist[None]  # (K, T)
        return decay * ops.cos(ang)[None], decay * ops.sin(ang)[None]

    def _writes(self, ids: Any):
        """What each position writes into each state space: real and imaginary `(..., L, K)`."""
        ops = keras.ops
        K, L = self.channels, self.state_spaces
        # what every token writes, p_{v,l} (s_v P_l): a (V, L, 2K) table built once per forward (the mix depends
        # only on the token, so it goes into the table, not onto every position), then one gather
        table = ops.einsum("vd,lde->vle", self.signature.embeddings, self.space_proj)
        table = table * ops.softmax(self.space_mix.embeddings, axis=-1)[..., None]
        w = ops.take(ops.reshape(table, (self._vocabulary_size, L * 2 * K)), ids, axis=0)
        w = ops.reshape(w, tuple(ids.shape) + (L, 2 * K))
        return w[..., :K], w[..., K:]

    # -- read -----------------------------------------------------------------

    def _read(self, ids: Any, sr: Any, si: Any, trace: Optional[List[Any]] = None) -> Any:
        """The cascade read of states `(..., L, K)` by the tokens `ids` (...): readings `(..., L * 3K)`.

        With `trace` (a list), appends `(query, filter re, filter im)` of every state space, for analysis.
        """
        ops = keras.ops
        K, L = self.channels, self.state_spaces
        q = self.signature(ids)  # the reader's signature starts the query
        feats = []
        for l in range(L):
            g = self.filter_head[l](self.query_norm[l](q))
            gr, gi = 1.0 + g[..., :K], g[..., K:]
            rr = gr * sr[..., l, :] - gi * si[..., l, :]
            ri = gr * si[..., l, :] + gi * sr[..., l, :]
            f = ops.concatenate([rr, ri, ops.sqrt(rr * rr + ri * ri + 1e-8)], axis=-1)
            feats.append(f)
            if trace is not None:
                trace.append((q, gr, gi))
            if l < L - 1:
                h = ops.concatenate([self.query_norm[l](q), self.heard_norm[l](f)], axis=-1)
                q = q + self.update_out[l](self.update_in[l](h))
        return ops.concatenate(feats, axis=-1)

    # -- forward --------------------------------------------------------------

    def hidden(self, inputs: Any) -> Any:
        """Single target: readings of the last token of each `(B, T)` window."""
        ops = keras.ops
        ids = ops.cast(inputs, "int32")
        wr, wi = self._writes(ids)  # (B, T, L, K)
        kr, ki = self._kernel(wr)  # (L, K, T): distance d at index d
        kr, ki = ops.flip(kr, axis=-1), ops.flip(ki, axis=-1)  # index t -> distance T-1-t
        sr = ops.einsum("lkt,btlk->blk", kr, wr) - ops.einsum("lkt,btlk->blk", ki, wi)
        si = ops.einsum("lkt,btlk->blk", kr, wi) + ops.einsum("lkt,btlk->blk", ki, wr)
        return self._read(ids[:, -1], sr, si)

    def dense_hidden(self, inputs: Any) -> Any:
        """All T queries of a `(B, 2T-1)` input (query m sees inputs m .. m+T-1): `(B, T, features)`."""
        ops = keras.ops
        ids = ops.cast(inputs, "int32")
        T, L, K = self.window_size, self.state_spaces, self.channels
        B, S = int(ids.shape[0]), int(ids.shape[1])
        Q = S - T + 1
        wr, wi = self._writes(ids)  # (B, S, L, K)
        kr, ki = self._kernel(wr)
        # compact band rows: [m, r] = weight of input m + r for query m (distance T-1-r), shared by every query
        band = lambda k: ops.broadcast_to(ops.flip(ops.reshape(k, (L * K, T)), axis=-1)[:, None, :], (L * K, Q, T))
        keys = ops.concatenate([ops.reshape(wr, (B, S, L * K)), ops.reshape(wi, (B, S, L * K))], axis=0)  # (2B, S, C)
        keys = ops.transpose(keys, (2, 1, 0))  # (C, S, 2B)
        r = _band_matmul(band(kr), keys)  # (C, Q, 2B): [kr*wr | kr*wi]
        i = _band_matmul(band(ki), keys)  # [ki*wr | ki*wi]
        to_state = lambda s: ops.reshape(ops.transpose(s, (2, 1, 0)), (B, Q, L, K))
        return self._read(ids[:, T - 1:], to_state(r[..., :B] - i[..., B:]), to_state(r[..., B:] + i[..., :B]))

    def readout_input(self, h: Any) -> Any:
        z = self.feature_norm(h)
        return self.base(z) + self.mlp_out(self.mlp_in(z))

    def call(self, inputs: Any, training: bool = False) -> Any:
        return self.readout(self.readout_input(self.hidden(inputs)))

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "vocabulary_size": self._vocabulary_size, "dim": self.dim,
                "window_size": self.window_size, "channels": self.channels, "state_spaces": self.state_spaces,
                "reader_hidden": self.reader_hidden, "mlp_hidden": self.mlp_hidden}
