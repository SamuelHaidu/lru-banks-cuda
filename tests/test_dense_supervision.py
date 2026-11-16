"""Dense supervision: one pass over `2T-1` tokens == `T` single-target windows.

The dense path is a re-derivation of each arch's forward pass in absolute
positions (see `model/archs/_dense.py`), so it is checked position by
position against the single-target `hidden`, and the dense sampler is
checked against the windows the single-target sampler builds.
"""

import unittest

from tests.helpers import VOCAB, temp_workspace, tiny_config

import numpy as np
import torch
from keras import ops

from model import config as cfg_mod
from model.archs import build_model
from model.archs._dense import band_mask, skew_band
from model.data import DenseWindowDataset, _fill_windows
from model.train import train

T = 8
CASES = (
    ("rope-logit-sum", {}),
    ("wave-field", {"channels": 6, "state_spaces": 3, "reader_hidden": 16}),
    ("wave-field", {"channels": 6, "state_spaces": 1, "reader_hidden": 16}),
    ("rope-logit-sum", {"per_dim_kernels": True}),
    ("rope-logit-sum", {"kernel_param": "exp-oscillators", "num_oscillators": 4}),
    ("rope-token-kernel", {"num_oscillators": 3}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3]}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_in_output": True, "memory_hidden": 32}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True,
                          "memory_hidden": 32, "bank_projections": True}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True,
                          "memory_hidden": 32, "general_concat": True}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True,
                          "memory_hidden": 32, "general_concat": True, "general_concat_norm": "joint", "bank_projections": True}),
    ("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True,
                          "memory_hidden": 32, "general_concat": True, "general_concat_norm": "joint", "bank_projections": True,
                          "tie_embeddings": True}),
    ("rope-token-kernel", {"num_oscillators": 3, "normalize": False, "saturation": False}),
    ("rope-logit-sum", {"kernel_param": "log-oscillators", "num_oscillators": 4}),
    ("rope-logit-sum", {"per_dim_kernels": True, "memory_hidden": 32}),
    ("rope-log-gaussian-sum", {"kernel_basis": "log-gaussian"}),
    ("rope-log-gaussian-sum", {"kernel_basis": "damped-oscillator"}),
    ("rope-kernel-attention", {}),
    ("rope-kernel-attention", {"attention_residual": True}),
    ("rope-kernel-hops", {"num_hops": 2}),
    ("rope-kernel-hops", {"num_hops": 3, "kernel_basis": "log-gaussian"}),
    ("rope-kernel-hops", {"num_hops": 2, "hop_kernel_basis": "fourier", "hop_num_centers": 6}),
    ("rope-log-gaussian-sum", {"kernel_basis": "fourier"}),
    ("rope-kernel-hops", {"num_hops": 3, "output_mode": "refine", "hop_kernel_basis": "fourier", "hop_num_centers": 6}),
    ("rope-kernel-hops", {"num_hops": 2, "output_mode": "refine", "readout_hidden": 6}),
    ("rope-log-gaussian-sum", {"readout_hidden": 6}),
    ("rope-kernel-hops", {"num_hops": 3, "output_mode": "refine", "context_dim": 24, "readout_hidden": 6,
                          "hop_kernel_basis": "fourier", "hop_num_centers": 6}),
    ("rope-kernel-attention", {"attention_residual": True, "key_value_projection": True}),
    ("rope-kernel-hops", {"num_hops": 2, "key_value_projection": True}),
    ("rope-kernel-hops", {"num_hops": 3, "output_mode": "refine", "context_dim": 24, "readout_hidden": 6,
                          "key_value_projection": True}),
    ("rope-kernel-hops", {"num_hops": 3, "output_mode": "refine", "context_dim": 24, "readout_hidden": 6,
                          "key_value_projection": True, "state_norm": True}),
)


def _params(arch, extra):
    p = {"vocabulary_size": VOCAB, "dim": 16, "window_size": T, **extra}
    if arch not in ("rope-logit-sum", "rope-token-kernel", "rope-token-banks"):
        p["mlp_hidden"] = 8
    if arch in ("rope-kernel-attention", "rope-kernel-hops"):
        p.update(query_hidden=8, num_heads=2)
    return p


class TestSkewBand(unittest.TestCase):
    def test_matches_explicit_band(self):
        v = np.random.default_rng(0).random((2, T, T)).astype("float32")
        expected = np.zeros((2, T, 2 * T - 1), "float32")
        for m in range(T):
            expected[:, m, m:m + T] = v[:, m]
        np.testing.assert_array_equal(ops.convert_to_numpy(skew_band(ops.convert_to_tensor(v))), expected)
        np.testing.assert_array_equal(band_mask(T), expected[0] != 0)


class TestDenseEqualsSingleTarget(unittest.TestCase):
    def test_every_query_matches_its_window(self):
        rng = np.random.default_rng(0)
        for arch, extra in CASES:
            with self.subTest(arch=arch, **extra):
                m = build_model({"arch": arch, "arch_params": _params(arch, extra)})
                if extra.get("attention_residual") or arch == "rope-kernel-hops":
                    m.attention_gate.assign(0.7)  # make the attention path contribute
                for g in getattr(m, "hop_kernel_gate", []) + getattr(m, "hop_attention_gate", []):
                    g.assign(0.6)  # and every extra hop
                if extra.get("per_dim_kernels"):  # distinct kernel per dimension
                    m.distance_weights.assign(rng.normal(0, 1, tuple(m.distance_weights.shape)).astype("float32"))
                if arch == "rope-token-banks":  # a non-uniform mix
                    m.router.kernel.assign(rng.normal(0, 0.3, tuple(m.router.kernel.shape)).astype("float32"))
                if arch in ("rope-token-kernel", "rope-token-banks"):  # distinct per-token numbers, repeated ids for saturation
                    for name in ("token_prior", "token_lambda", "token_kappa", "bank_prior", "bank_lambda"):
                        if not hasattr(m, name):
                            continue
                        w = getattr(m, name)
                        w.assign(rng.normal(0, 1, tuple(w.shape)).astype("float32"))
                if extra.get("bank_projections"):  # off identity: a distinct view per state
                    m.bank_proj.assign(rng.normal(0, 0.3, tuple(m.bank_proj.shape)).astype("float32"))
                if arch == "wave-field":  # non-uniform mixes, projections and decays; filters and MLPs on
                    for w in [m.space_mix.embeddings, m.mlp_out.kernel] + [l.kernel for l in m.filter_head + m.update_out]:
                        w.assign(rng.normal(0, 0.5, tuple(w.shape)).astype("float32"))
                    m.decay_raw.assign(rng.normal(-1, 0.7, tuple(m.decay_raw.shape)).astype("float32"))
                    m.space_proj.assign(rng.normal(0, 0.3, tuple(m.space_proj.shape)).astype("float32"))
                if extra.get("memory_hidden"):  # zero-init output would hide the MLP
                    m.memory_out.kernel.assign(rng.normal(0, 0.3, tuple(m.memory_out.kernel.shape)).astype("float32"))
                rng_w = np.random.default_rng(7)
                for layer in getattr(m, "key_proj", []) + getattr(m, "value_proj", []):
                    layer.kernel.assign(rng_w.normal(0, 0.3, (16, 16)).astype("float32"))  # off identity
                x = rng.integers(1, VOCAB, (3, 2 * T - 1)).astype("int32")
                x[0, :5] = 0  # document-start padding
                dense = ops.convert_to_numpy(m.readout_input(m.dense_hidden(ops.convert_to_tensor(x))))
                single = np.stack([ops.convert_to_numpy(m.readout_input(m.hidden(ops.convert_to_tensor(x[:, i:i + T]))))
                                   for i in range(T)], axis=1)
                # float32 through several hops, norms and projections: ~1e-6 noise on small entries
                np.testing.assert_allclose(dense, single, rtol=1e-5, atol=1e-5)

    def test_call_is_readout_of_hidden(self):
        for arch, extra in CASES:
            with self.subTest(arch=arch, **extra):
                m = build_model({"arch": arch, "arch_params": _params(arch, extra)})
                x = ops.convert_to_tensor(np.random.default_rng(1).integers(0, VOCAB, (2, T)).astype("int32"))
                np.testing.assert_allclose(ops.convert_to_numpy(m(x)), ops.convert_to_numpy(m.readout(m.readout_input(m.hidden(x)))), rtol=1e-6, atol=1e-7)


class TestTokenBanksProjections(unittest.TestCase):
    def test_identity_init_matches_no_projection(self):
        p = _params("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True,
                                         "general_in_output": True, "memory_hidden": 32})
        base = build_model({"arch": "rope-token-banks", "arch_params": p})
        proj = build_model({"arch": "rope-token-banks", "arch_params": {**p, "bank_projections": True}})
        key = lambda v: v.path.split("/", 1)[1]  # path without the model's own (numbered) name
        by_path = {key(v): v for v in proj.weights}
        for v in base.weights:  # same weights; bank_proj stays at its identity init
            by_path[key(v)].assign(v.value)
        x = ops.convert_to_tensor(np.random.default_rng(2).integers(0, VOCAB, (2, 2 * T - 1)).astype("int32"))
        np.testing.assert_allclose(ops.convert_to_numpy(proj.dense_hidden(x)), ops.convert_to_numpy(base.dense_hidden(x)), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ops.convert_to_numpy(proj(x[:, :T])), ops.convert_to_numpy(base(x[:, :T])), rtol=1e-5, atol=1e-6)


class TestWaveFieldRecurrence(unittest.TestCase):
    def test_state_is_the_damped_rotating_recurrence(self):
        """The windowed sum the model computes equals S <- e^{-alpha + i omega} S + w, run over the window."""
        m = build_model({"arch": "wave-field", "arch_params": _params("wave-field", {"channels": 6, "state_spaces": 3, "reader_hidden": 16})})
        rng = np.random.default_rng(5)
        for w in [m.space_mix.embeddings, m.space_proj, m.decay_raw] + [l.kernel for l in m.filter_head + m.update_out]:
            w.assign(rng.normal(0, 0.5, tuple(w.shape)).astype("float32"))
        x = ops.convert_to_tensor(rng.integers(1, VOCAB, (2, T)).astype("int32"))
        wr, wi = (ops.convert_to_numpy(v) for v in m._writes(x))  # (B, T, L, K)
        alpha = np.log1p(np.exp(ops.convert_to_numpy(m.decay_raw).astype("float64")))  # (L, K)
        omega = ops.convert_to_numpy(m.channel_omega).astype("float64")
        Z = np.exp(-alpha[..., None] * np.arange(T)).sum(-1)  # (L, K)
        S = np.zeros(wr.shape[:1] + wr.shape[2:], complex)
        for t in range(T):
            S = np.exp(-alpha + 1j * omega) * S + (wr[:, t] + 1j * wi[:, t]) / Z
        h = m._read(x[:, -1], ops.convert_to_tensor(S.real.astype("float32")), ops.convert_to_tensor(S.imag.astype("float32")))
        np.testing.assert_allclose(ops.convert_to_numpy(m.hidden(x)), ops.convert_to_numpy(h), rtol=1e-4, atol=1e-5)


class TestWaveArrow(unittest.TestCase):
    def _model(self):
        m = build_model({"arch": "wave-arrow", "arch_params": _params("wave-arrow", {"channels": 6, "state_spaces": 2})})
        rng = np.random.default_rng(5)
        for w in (m.proj_len, m.proj_ang, m.space_gamma, m.amp_raw.embeddings, m.phase_raw.embeddings, m.decay_raw):
            w.assign(rng.normal(0, 2, tuple(w.shape)).astype("float32"))
        return m, rng

    def test_writes_are_bounded_arrows(self):
        m, _ = self._model()
        tr, ti = (ops.convert_to_numpy(t) for t in m.write_table())
        self.assertLess(float(np.hypot(tr, ti).max()), 1.0)
        # w = gamma (P s): bounded projection (rows average turned arrows), then the LRU normalization per channel
        c = lambda x: ops.convert_to_numpy(x).astype("float64")
        sig = lambda x: 1 / (1 + np.exp(-c(x)))
        ang = lambda x: np.pi * np.tanh(c(x))
        s = sig(m.amp_raw.embeddings) * np.exp(1j * ang(m.phase_raw.embeddings))  # (V, K)
        e = np.exp(c(m.proj_len) - c(m.proj_len).max(-1, keepdims=True))
        P = e / e.sum(-1, keepdims=True) * np.exp(1j * ang(m.proj_ang))  # (L, K, K)
        G = sig(m.space_gamma)  # (L, K): gamma only
        np.testing.assert_allclose(tr + 1j * ti, G[None] * np.einsum("lkj,vj->vlk", P, s), rtol=1e-4, atol=1e-6)

    def test_projection_starts_as_identity(self):
        m = build_model({"arch": "wave-arrow", "arch_params": _params("wave-arrow", {"channels": 6, "state_spaces": 2})})
        tr, ti = (ops.convert_to_numpy(t) for t in m.write_table())
        sig = lambda x: 1 / (1 + np.exp(-ops.convert_to_numpy(x)))
        s = sig(m.amp_raw.embeddings) * np.exp(1j * np.pi * np.tanh(ops.convert_to_numpy(m.phase_raw.embeddings)))
        # identity projection: each space writes gamma * signature, gamma = sqrt(1 - |lambda|^2)
        lam = np.exp(-np.log1p(np.exp(ops.convert_to_numpy(m.decay_raw).astype("float64"))))
        gamma = np.sqrt(1 - lam ** 2)
        np.testing.assert_allclose(tr + 1j * ti, gamma[None] * s[:, None, :], atol=0.01)

    def _recurrence(self, m, x, state=None):
        """Reference: S <- lambda S + w token by token, S <- w at the reset token (numpy, float64)."""
        wr, wi = (ops.convert_to_numpy(v).astype("float64") for v in m._writes(ops.convert_to_tensor(x)))
        alpha = np.log1p(np.exp(ops.convert_to_numpy(m.decay_raw).astype("float64")))
        omega = ops.convert_to_numpy(m.channel_omega).astype("float64")
        S = np.zeros(wr.shape[:1] + wr.shape[2:], complex) if state is None else state.copy()
        out = []
        for t in range(x.shape[1]):  # not normalized, no window
            S = np.exp(-alpha + 1j * omega) * S * (x[:, t] != m.reset_token)[:, None, None] + (wr[:, t] + 1j * wi[:, t])
            out.append(S.reshape(len(x), -1))
        return np.stack(out, axis=1)  # (B, N, L K)

    def test_scan_is_the_damped_rotating_recurrence_with_resets(self):
        """Every position, any chunk size (3 does not divide N), resets mid-row and from a carried state."""
        m, rng = self._model()
        N = 3 * T + 2
        x = rng.integers(1, VOCAB, (3, N)).astype("int32")
        x[0, 0] = 0  # row starts at a document
        x[0, 9] = x[1, 4] = x[1, 5] = 0  # packed documents (and an empty one)
        S0 = rng.normal(0, 1, (3, 2, 6)) + 1j * rng.normal(0, 1, (3, 2, 6))
        for chunk in (3, T, N):
            m.chunk_size = chunk
            for state in (None, S0):
                with self.subTest(chunk=chunk, carried=state is not None):
                    st = None if state is None else tuple(ops.convert_to_tensor(v.reshape(3, -1).astype("float32")) for v in (state.real, state.imag))
                    sr, si = (ops.convert_to_numpy(v) for v in m.scan(ops.convert_to_tensor(x), st))
                    np.testing.assert_allclose(sr + 1j * si, self._recurrence(m, x, state), rtol=1e-4, atol=1e-5)

    def test_dense_is_every_prefix(self):
        m, rng = self._model()
        m.chunk_size = 3
        x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
        x[:, 0] = 0
        x[1, 6] = 0
        dense = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
        for i in range(x.shape[1]):  # left-padded with markers, as the validation windows are
            window = np.zeros_like(x)
            window[:, -(i + 1):] = x[:, :i + 1]
            np.testing.assert_allclose(dense[:, i], ops.convert_to_numpy(m.hidden(ops.convert_to_tensor(window))), rtol=1e-4, atol=1e-5)

    def test_stateful_generation_matches_full_context(self):
        """Token-by-token state == re-reading the whole text; a finite copy reads only its window."""
        m, _ = self._model()
        prompt = [0, 5, 9, 3]
        full = m.with_context_length(64)  # window longer than the text: same context as the recurrence
        stateful = m.generate(prompt, max_new_tokens=12)
        self.assertEqual(stateful, full.generate(prompt, max_new_tokens=12))
        self.assertFalse(full.stateful_generation)
        self.assertTrue(m.stateful_generation)


class TestWaveArrowLayers(unittest.TestCase):
    """`num_layers=2`: layer 2 writes from layer 1's space 1 (squashed to length tanh|z|), logits add over layers."""

    def _model(self, num_layers=2):
        m = build_model({"arch": "wave-arrow", "arch_params": _params(
            "wave-arrow", {"channels": 6, "state_spaces": 2, "num_layers": num_layers})})
        rng = np.random.default_rng(7)
        for v in m.trainable_weights:
            if not v.path.startswith(("mlp_", "readout")) and "/mlp_" not in v.path:
                v.assign(rng.normal(0, 2, tuple(v.shape)).astype("float32"))
            elif "/mlp_out_l" in v.path:  # layers 2+ start with zero output weights; make them count here
                v.assign(rng.normal(0, 0.1, tuple(v.shape)).astype("float32"))
        return m, rng

    def test_later_layers_start_silent(self):
        m = build_model({"arch": "wave-arrow", "arch_params": _params(
            "wave-arrow", {"channels": 6, "state_spaces": 2, "num_layers": 2})})
        self.assertEqual(float(np.abs(ops.convert_to_numpy(m.mlp_out_l2.kernel)).max()), 0.0)
        self.assertGreater(float(np.abs(ops.convert_to_numpy(m.mlp_out.kernel)).max()), 0.0)
        x = ops.convert_to_tensor(np.random.default_rng(0).integers(1, VOCAB, (2, 9)).astype("int32"))
        sr, si = m.scan(x)  # layer 1 alone gives the whole output
        np.testing.assert_allclose(ops.convert_to_numpy(m.hidden(x)),
                                   ops.convert_to_numpy(m._mlp(0, sr[:, -1], si[:, -1])), atol=1e-6)

    def test_single_layer_keeps_its_parameter_names(self):
        one = {v.path.split("/", 1)[1] for v in self._model(1)[0].weights}
        two = {v.path.split("/", 1)[1] for v in self._model(2)[0].weights}
        self.assertTrue(one < two)
        self.assertTrue(all("_l2" in p for p in two - one))

    def test_layer_two_is_the_recurrence_over_squashed_layer_one_states(self):
        m, rng = self._model()
        m.chunk_size = 3
        x = rng.integers(1, VOCAB, (2, 2 * T + 1)).astype("int32")
        x[:, 0] = 0
        x[1, 7] = 0
        (s1r, s1i), (s2r, s2i) = (tuple(ops.convert_to_numpy(v).astype("float64") for v in pair)
                                  for pair in m.scan_layers(ops.convert_to_tensor(x)))
        c = lambda v: ops.convert_to_numpy(v).astype("float64")
        z = (s1r + 1j * s1i)[..., :6]  # space 1 of layer 1
        z = z * np.tanh(np.abs(z)) / np.abs(z)
        e = np.exp(c(m.proj_len_l2) - c(m.proj_len_l2).max(-1, keepdims=True))
        P = e / e.sum(-1, keepdims=True) * np.exp(1j * np.pi * np.tanh(c(m.proj_ang_l2)))
        w = (1 / (1 + np.exp(-c(m.space_gamma_l2))))[None, None] * np.einsum("lkj,bnj->bnlk", P, z)
        self.assertLess(float(np.abs(w).max()), 1.0)
        lam = np.exp(-np.log1p(np.exp(c(m.decay_raw_l2))) + 1j * c(m.channel_omega_l2)[None])
        S, ref = np.zeros((2, 2, 6), complex), []
        for t in range(x.shape[1]):
            S = lam * S * (x[:, t] != 0)[:, None, None] + w[:, t]
            ref.append(S.reshape(2, -1))
        np.testing.assert_allclose(s2r + 1j * s2i, np.stack(ref, 1), rtol=1e-4, atol=1e-5)

    def test_dense_is_every_prefix(self):
        m, rng = self._model()
        m.chunk_size = 3
        x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
        x[:, 0] = 0
        x[1, 6] = 0
        dense = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
        self.assertEqual(dense.shape[-1], m.dim)  # only the summed `dim` outputs leave the layers
        for i in range(x.shape[1]):
            window = np.zeros_like(x)
            window[:, -(i + 1):] = x[:, :i + 1]
            np.testing.assert_allclose(dense[:, i], ops.convert_to_numpy(m.hidden(ops.convert_to_tensor(window))), rtol=1e-4, atol=1e-5)

    def test_stateful_generation_matches_full_context(self):
        m, _ = self._model()
        prompt = [0, 5, 9, 3]
        self.assertEqual(m.generate(prompt, max_new_tokens=12), m.with_context_length(64).generate(prompt, max_new_tokens=12))

    def test_recompute_mlp_matches_plain_autograd(self):
        """Same outputs and gradients as gelu(x W1 + b1) W2 + b2 by plain autograd, for both GELUs, with more rows
        than one recompute block."""
        import torch

        from model.archs.wave_arrow import _RecomputeMLP

        saved, _RecomputeMLP.rows = _RecomputeMLP.rows, 4
        try:
            for mode, approx in (("exact", "none"), ("tanh", "tanh")):
                with self.subTest(mlp_gelu=mode):
                    m, rng = self._model(1)
                    m.mlp_gelu = mode
                    x = torch.as_tensor(rng.normal(0, 1, (11, 24)).astype("float32"), device=m.mlp_in.kernel.value.device)
                    wts = (m.mlp_in.kernel, m.mlp_in.bias, m.mlp_out.kernel, m.mlp_out.bias)
                    got = []
                    for plain in (False, True):
                        xx = x.clone().requires_grad_()
                        for v in wts:
                            v.value.grad = None
                        if plain:
                            w1, b1, w2, b2 = (v.value for v in wts)
                            y = torch.nn.functional.gelu(xx @ w1 + b1, approximate=approx) @ w2 + b2
                        else:
                            y = m._mlp(0, xx[:, :12], xx[:, 12:])
                        (y * torch.linspace(-1, 1, y.shape[-1], device=y.device)).sum().backward()
                        got.append([y.detach()] + [v.value.grad for v in wts] + [xx.grad])
                    for a, b in zip(*got):
                        np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-4, atol=1e-5)
        finally:
            _RecomputeMLP.rows = saved

    def test_gelu_option_is_validated_and_saved(self):
        with self.assertRaises(ValueError):
            build_model({"arch": "wave-arrow", "arch_params": _params("wave-arrow", {"channels": 6, "mlp_gelu": "relu"})})
        m = build_model({"arch": "wave-arrow", "arch_params": _params("wave-arrow", {"channels": 6, "mlp_gelu": "tanh"})})
        self.assertEqual(m.get_config()["mlp_gelu"], "tanh")
        self.assertTrue(m.get_config()["mlp_tf32"])  # CUDA-only: a no-op on CPU/XPU, so results match fp32 there

    def test_training_run_and_reload(self):
        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7, 20], validation_doc_lengths=[6, 11]):
            cfg = tiny_config("packed2", arch="wave-arrow")
            cfg["arch_params"].update({"channels": 6, "dim": 16, "mlp_hidden": 8, "chunk_size": 3, "num_layers": 2})
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("packed2", cfg)
            history = train("packed2", verbose=0)
            self.assertTrue(np.isfinite(history.history["val_loss"][-1]))
            history = train("packed2", epochs=2, verbose=0)  # resume: reloads the 2-layer model.keras
            self.assertTrue(np.isfinite(history.history["val_loss"][-1]))


class TestPackedDocuments(unittest.TestCase):
    def test_rows_start_at_documents_and_skip_marker_targets(self):
        from model.data import PackedDocumentDataset

        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7]) as root:
            path = str(root / "datasets" / "tiny" / "train.npy")
            ds = PackedDocumentDataset(path, window_size=6, batch_size=32, vocabulary_size=VOCAB, doc_token_id=0, seed=3)
            tokens = np.load(path)
            starts = set(np.flatnonzero(tokens == 0).tolist())
            x, y = ds[0]
            self.assertEqual(x.shape, (32, 6))
            self.assertEqual(y.shape, (32, 6))
            padded = np.append(tokens, np.zeros(8, np.int32))  # past the corpus end: markers, no targets
            for row_x, row_y in zip(x, y):
                s = [p for p in starts if np.array_equal(padded[p:p + 6], row_x)]
                self.assertTrue(s, "every row starts at a document start")
                nxt = padded[s[0] + 1:s[0] + 7]
                np.testing.assert_array_equal(row_y, np.where(nxt == 0, -1, nxt))

    def test_training_run_on_packed_rows(self):
        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7, 20], validation_doc_lengths=[6, 11]):
            cfg = tiny_config("packed", arch="wave-arrow")
            cfg["arch_params"].update({"channels": 6, "dim": 16, "mlp_hidden": 8, "chunk_size": 3})
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("packed", cfg)
            history = train("packed", verbose=0)
            self.assertTrue(np.isfinite(history.history["loss"][-1]))
            self.assertTrue(np.isfinite(history.history["val_loss"][-1]))

    def test_corruption_never_touches_markers(self):
        from model.data import PackedDocumentDataset

        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7]) as root:
            path = str(root / "datasets" / "tiny" / "train.npy")
            clean = PackedDocumentDataset(path, window_size=8, batch_size=64, vocabulary_size=VOCAB, doc_token_id=0, seed=3)
            noisy = PackedDocumentDataset(path, window_size=8, batch_size=64, vocabulary_size=VOCAB, doc_token_id=0, seed=3,
                                          corrupt_prob=0.5)
            (xc, yc), (xn, yn) = clean[0], noisy[0]
            np.testing.assert_array_equal(xc == 0, xn == 0)
            np.testing.assert_array_equal(yc, yn)
            self.assertTrue((xc != xn).any())


class TestTokenBanksTiedEmbeddings(unittest.TestCase):
    def test_readout_is_the_embedding_table(self):
        p = _params("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "memory_hidden": 8, "tie_embeddings": True})
        m = build_model({"arch": "rope-token-banks", "arch_params": p})
        E = ops.convert_to_numpy(m.token_embedding.embeddings)
        np.testing.assert_array_equal(ops.convert_to_numpy(m.readout.kernel), E.T)
        untied = build_model({"arch": "rope-token-banks", "arch_params": {**p, "tie_embeddings": False}})
        self.assertEqual(untied.count_params() - m.count_params(), VOCAB * 16)  # one table fewer
        self.assertEqual(len({id(v) for v in m.weights}), len(m.weights))
        h = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(3, 16)).astype("float32"))
        np.testing.assert_allclose(ops.convert_to_numpy(m.readout(h)), ops.convert_to_numpy(h) @ E.T, rtol=1e-5, atol=1e-5)

    def test_tied_run_trains_saves_and_reloads(self):
        from model.evaluate import load_checkpoint

        with temp_workspace(train_doc_lengths=[40, 25, 60, 30], validation_doc_lengths=[30, 18]):
            cfg = tiny_config(run_name="tied", arch="rope-token-banks")
            cfg["arch_params"] = _params("rope-token-banks", {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True,
                                                              "general_in_output": True, "memory_hidden": 8, "general_concat": True,
                                                              "general_concat_norm": "joint", "bank_projections": True, "tie_embeddings": True})
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("tied", cfg)
            history = train("tied", verbose=0)
            self.assertTrue(np.isfinite(history.history["loss"][-1]))
            m = load_checkpoint("tied")
            self.assertTrue(m.tie_embeddings)
            x = ops.convert_to_tensor(np.random.default_rng(4).integers(1, VOCAB, (2, T)).astype("int32"))
            E = ops.convert_to_numpy(m.token_embedding.embeddings)
            np.testing.assert_array_equal(ops.convert_to_numpy(m.readout.kernel), E.T)  # still one shared table after loading
            h = m.readout_input(m.hidden(x))
            np.testing.assert_allclose(ops.convert_to_numpy(m(x)), ops.convert_to_numpy(h) @ E.T, rtol=1e-5, atol=1e-5)


class TestTokenBanksComplexRope(unittest.TestCase):
    def test_complex_and_stacked_paths_match_the_real_rotation(self):
        rng = np.random.default_rng(3)
        base = {"num_oscillators": 3, "bank_lambdas": [0.9, 0.3], "general_pair_kernels": True, "general_in_output": True, "memory_hidden": 32}
        for extra in ({}, {"bank_projections": True}, {"general_concat": True}, {"memory_heads": 2, "memory_window": 4, "memory_value_range": 3, "memory_mlp_hidden": 8, "general_concat": True}, {"general_concat": True, "general_concat_norm": "joint", "memory_heads": 2, "memory_window": 4, "memory_value_range": 3, "memory_mlp_hidden": 8},
                      {"memory_heads": 2, "memory_window": 4, "memory_value_range": 3, "memory_mlp_hidden": 8},
                      {"bank_projections": True, "memory_heads": 2, "memory_window": 4, "memory_value_range": 3, "memory_mlp_hidden": 8}):
            with self.subTest(**extra):
                m = build_model({"arch": "rope-token-banks", "arch_params": _params("rope-token-banks", {**base, **extra})})
                for v in m.trainable_weights:  # generic, non-identity weights everywhere
                    v.assign(ops.convert_to_tensor(np.asarray(ops.convert_to_numpy(v.value)) + rng.normal(0, 0.1, tuple(v.shape)).astype("float32")))
                x = ops.convert_to_tensor(rng.integers(1, VOCAB, (2, 2 * m.context_length - 1)).astype("int32"))
                outs = []
                for flags in ((False, False), (True, False), (True, True)):
                    m._complex_rope, m._stacked_lookup = flags
                    y = m.readout_input(m.dense_hidden(x))
                    grads = torch.autograd.grad(y.square().sum(), [v.value for v in m.trainable_weights], allow_unused=True)
                    outs.append((ops.convert_to_numpy(y), [None if g is None else g.detach().cpu().numpy() for g in grads]))
                for y, grads in outs[1:]:
                    np.testing.assert_allclose(y, outs[0][0], rtol=1e-4, atol=1e-5)
                    for g, g0 in zip(grads, outs[0][1]):
                        if g0 is not None:
                            np.testing.assert_allclose(g, g0, rtol=1e-3, atol=1e-4 * (np.abs(g0).max() + 1e-12))


class TestTokenBanksMemory(unittest.TestCase):
    """rope-token-banks with associative memories: context = window + memory_window - 1."""

    def _model(self, **extra):
        p = {"vocabulary_size": VOCAB, "dim": 16, "window_size": 5, "num_oscillators": 3, "bank_lambdas": [0.9, 0.3],
             "general_pair_kernels": True, "general_in_output": True, "memory_hidden": 32, "memory_heads": 2,
             "memory_window": 4, "memory_value_range": 3, "memory_oscillators": 2, "memory_mlp_hidden": 8, **extra}
        m = build_model({"arch": "rope-token-banks", "arch_params": p})
        rng = np.random.default_rng(3)
        for w in m.weights:  # move everything off its init so every path matters
            if "token_emb" in w.path or "readout" in w.path:
                continue
            w.assign(ops.convert_to_numpy(w) + rng.normal(0, 0.3, tuple(w.shape)).astype("float32"))
        return m

    def test_context_length(self):
        self.assertEqual(self._model().context_length, 5 + 4 - 1)

    def test_every_query_matches_its_window(self):
        m = self._model()
        C = m.context_length
        x = np.random.default_rng(0).integers(1, VOCAB, (3, 2 * C - 1)).astype("int32")
        x[0, :4] = 0
        x[1, 5:9] = x[1, 0]  # repeated tokens
        dense = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
        single = np.stack([ops.convert_to_numpy(m.hidden(ops.convert_to_tensor(x[:, i:i + C]))) for i in range(C)], axis=1)
        self.assertEqual(dense.shape, (3, C, 16 * 3))
        np.testing.assert_allclose(dense, single, rtol=1e-4, atol=1e-5)

    def test_no_future_leak(self):
        m = self._model()
        C = m.context_length
        x = np.random.default_rng(1).integers(1, VOCAB, (2, 2 * C - 1)).astype("int32")
        y = x.copy()
        y[:, C + 2:] = (y[:, C + 2:] + 7) % (VOCAB - 1) + 1  # change inputs after query m = 2
        hx = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
        hy = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(y)))
        np.testing.assert_allclose(hx[:, :3], hy[:, :3], rtol=1e-5, atol=1e-6)  # queries 0..2 end at <= C+1
        self.assertGreater(np.abs(hx[:, 3:] - hy[:, 3:]).max(), 1e-4)

    def test_call_and_readout(self):
        m = self._model()
        x = ops.convert_to_tensor(np.random.default_rng(2).integers(0, VOCAB, (2, m.context_length)).astype("int32"))
        np.testing.assert_allclose(ops.convert_to_numpy(m(x)), ops.convert_to_numpy(m.readout(m.readout_input(m.hidden(x)))), rtol=1e-6, atol=1e-6)


class TestDenseWindowDataset(unittest.TestCase):
    def test_queries_see_the_single_target_windows(self):
        with temp_workspace(train_doc_lengths=[40, 25, 60, 12, 5]):
            path = str(cfg_mod.dataset_split_path("tiny", "train"))
            meta = cfg_mod.load_dataset_metadata("tiny")
            ds = DenseWindowDataset(path, window_size=T, batch_size=16, vocabulary_size=VOCAB,
                                    doc_token_id=meta["doc_token_id"], steps_per_epoch=2, use_cache=False)
            ds.tokens = np.asarray(ds.tokens).astype(np.uint16)  # real corpora are stored uint16
            x, y = ds[0]
            self.assertEqual((x.dtype, y.dtype), (np.int32, np.int32))
            self.assertEqual(x.shape, (16, 2 * T - 1))
            self.assertEqual(y.shape, (16, T))
            # Recover each sample's document and first target from its rows, then compare.
            rng = np.random.default_rng(42)  # same draws as the dataset's first batch
            docs = rng.integers(0, ds.num_docs, size=16)
            starts, ends = ds.doc_starts[docs], ds.doc_ends[docs]
            p0 = rng.integers(starts + 1, np.maximum(starts + 2, ends - T + 1))
            for b in range(16):
                for m in range(T):
                    target = p0[b] + m
                    if target < ends[b]:
                        window, _ = _fill_windows(ds.tokens, np.array([target]), np.array([starts[b]]), T)
                        np.testing.assert_array_equal(x[b, m:m + T], window[0])
                        self.assertEqual(y[b, m], ds.tokens[target])
                    else:
                        self.assertEqual(y[b, m], -1)


class TestDenseTraining(unittest.TestCase):
    def test_dense_run_trains_and_validates_single_target(self):
        with temp_workspace(train_doc_lengths=[40, 25, 60, 30], validation_doc_lengths=[30, 18]):
            cfg = tiny_config(run_name="dense", arch="rope-kernel-attention")
            cfg["arch_params"].update(dim=16, mlp_hidden=8, query_hidden=8, num_heads=2, attention_residual=True)
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("dense", cfg)
            history = train("dense", verbose=0)
            self.assertTrue(np.isfinite(history.history["loss"][-1]))
            self.assertIn("val_loss", history.history)
            self.assertTrue((cfg_mod.run_dir("dense") / "model.keras").exists())

    def test_window_tied_arch_is_rejected(self):
        with temp_workspace():
            cfg = tiny_config(run_name="bad", arch="dual-sphere")
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("bad", cfg)
            with self.assertRaises(ValueError):
                train("bad", verbose=0)


if __name__ == "__main__":
    unittest.main()
