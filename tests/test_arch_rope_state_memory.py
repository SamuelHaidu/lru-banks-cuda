"""`rope-state-memory`: the chunkwise parallel path == the token-by-token recurrence."""

import unittest

from tests.helpers import VOCAB, temp_workspace, tiny_config

import numpy as np
import torch
from keras import ops

from model import config as cfg_mod
from model.archs import build_model
from model.train import train

T = 8


def _model(num_layers=1, seed=5, **extra):
    p = {"vocabulary_size": VOCAB, "dim": 16, "window_size": T, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8,
         "num_layers": num_layers, "rope_base": 10.0, **extra}
    m = build_model({"arch": "rope-state-memory", "arch_params": p})
    rng = np.random.default_rng(seed)
    m.token_embedding.embeddings.assign(rng.normal(0, 1, tuple(m.token_embedding.embeddings.shape)).astype("float32"))
    for n in range(num_layers):  # decays far from the init, so a wrong power shows
        m._w("decay_state", n).assign(rng.normal(0, 1))
        if m.memory == "linear":
            m._w("decay_memory", n).assign(rng.normal(-1, 1))
        else:
            v = m._w("decay_banks", n)
            v.assign(rng.normal(-1, 1, tuple(v.shape)).astype("float32"))
        for name in ("address_q", "address_k"):  # non-uniform addressing (the bank projections start at 0)
            v = m._w(name, n)
            v.assign(rng.normal(0, 0.7, tuple(v.shape)).astype("float32"))
        if m.memory == "banks_delta":  # a token-dependent write strength
            v = m._w("write_strength_w", n)
            v.assign(rng.normal(0, 0.5, tuple(v.shape)).astype("float32"))
            m._w("write_strength_b", n).assign(np.array([0.3], "float32"))
        for v in (m._w("mlp_out", n).kernel, m._w("mlp_out", n).bias):  # a live MLP (its output starts at zero)
            v.assign(rng.normal(0, 0.3, tuple(v.shape)).astype("float32"))
        if m.read_mix == "gates":  # a live keep-the-state gate
            v = m._w("keep_state_w", n)
            v.assign(rng.normal(0, 1, tuple(v.shape)).astype("float32"))
            m._w("keep_state_b", n).assign(np.array([1.0], "float32"))
        if m.mlp_residual == "gates":  # live keep / MLP gates
            for name, bias in (("keep_input", 1.0), ("mlp_gate", -0.5)):
                v = m._w(name + "_w", n)
                v.assign(rng.normal(0, 0.5, tuple(v.shape)).astype("float32"))
                m._w(name + "_b", n).assign(np.array([bias], "float32"))
        if m.mlp_residual == "mix":  # a live mixing head leaning to the input (b = -1): random inputs and MLP outputs
            v = m._w("mix_head_w", n)  # mixed half and half can nearly cancel, and the next RMSNorm magnifies rounding
            v.assign(rng.normal(0, 0.3, tuple(v.shape)).astype("float32"))
            m._w("mix_head_b", n).assign(np.array([-1.0], "float32"))
        if m.write_gate:  # a token-dependent write gate (it starts nearly closed, the same for every token)
            v = m._w("write_gate_w", n)
            v.assign(rng.normal(0, 1, tuple(v.shape)).astype("float32"))
            m._w("write_gate_b", n).assign(np.array([-0.5], "float32"))
        g = m._w("read_gate_w", n)  # an open, query-dependent gate (it starts nearly closed)
        g.assign(rng.normal(0, 1, tuple(g.shape)).astype("float32"))
        m._w("read_gate_b", n).assign(np.array([0.5], "float32"))
        for name in ("in_norm", "mlp_norm")[0 if m.pre_norm else 1:]:  # non-trivial norm scales
            v = m._w(name, n).scale
            v.assign(rng.uniform(0.5, 1.5, tuple(v.shape)).astype("float32"))
    if m.pre_norm:
        m.final_norm.scale.assign(rng.uniform(0.5, 1.5, (m.dim,)).astype("float32"))
    return m, rng


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _reference(m, x, n_layers=None):
    """Token by token (float64), written out independently of the model's code:
    u = RMSNorm(x); s <- lambda R_1 s + gamma u; write address a = W_k s_before (0 after a marker), read a' = W_q s;
    banks: w = beta softmax(a) (beta = sigmoid(w_w . s_before + b_w), the write gate, 1 without it; from the
    logits a with write_gate_input "logits"), r = softmax(a'),
    bank_b <- mu_b bank_b + (1 - mu_b) w_b v, z_b <- mu_b z_b + (1 - mu_b) w_b, o = (r . banks) / (r . z);
    banks_delta: k, q = unit softmaxes, beta = sigmoid(w . s_before + b), Z <- D Z, e = beta (v - k^T Z), Z += k e^T,
    o = Z^T q;
    linear: k, q = normalized addresses, M <- mu M + k v^T, o = sqrt(1 - mu^2) M^T q; then the read's weight
    g = sigmoid(w_g . a' + b) (read_mix "add": h = s + g o; "gates": h = sigmoid(w_a . a' + b_a) s + g o) or
    (1 + tanh(w_g . a' + b)) / 2 ("tanh": h = (1 - g) s + g o); returns (s, g o) per layer."""
    c = lambda v: ops.convert_to_numpy(v).astype("float64")
    dev = m.token_embedding.embeddings.value.device
    B, N = x.shape
    D, half = m.dim, m.dim // 2
    theta = m.rope_base ** (-np.arange(half) / half)
    rot = lambda s: np.concatenate([s[:, :half] * np.cos(theta) - s[:, half:] * np.sin(theta),
                                    s[:, :half] * np.sin(theta) + s[:, half:] * np.cos(theta)], -1)
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-6)
    inp = c(m.token_embedding.embeddings)[x]
    out = []
    for n in range(n_layers or m.num_layers):
        lam, mu = (c(d) for d in m.decays(n))
        lam = float(lam)
        Wq, Wk = (c(m._w(name, n)) for name in ("address_q", "address_k"))
        Wv = c(m._w("value", n).kernel)
        wg, bg = c(m._w("read_gate_w", n)), c(m._w("read_gate_b", n))
        gate_in = (lambda s_, a_r_: s_) if m.read_gate_input == "state" else (lambda s_, a_r_: a_r_)
        u = (inp / np.sqrt((inp ** 2).mean(-1, keepdims=True) + 1e-6) * c(m._w("in_norm", n).scale)
             if m.pre_norm else inp)
        s, M, ss, oo, gg, kk = np.zeros((B, D)), np.zeros((B, m.memory_rows, D)), [], [], [], []
        Z = np.zeros((B, m.memory_rows))
        for t in range(N):
            keep = (x[:, t] != m.reset_token)[:, None]
            pre = s * keep
            a_w = pre @ Wk
            s = lam * rot(pre) + np.sqrt(1 - lam ** 2) * u[:, t]
            a_r = s @ Wq
            v = u[:, t] @ Wv
            if m.memory == "banks_delta":
                k, q = unit(_softmax(a_w)), unit(_softmax(a_r))
                beta = 1 / (1 + np.exp(-(pre @ c(m._w("write_strength_w", n)) + c(m._w("write_strength_b", n)))))
                M = mu[None, :, None] * M * keep[..., None]
                e = beta * (v - np.einsum("bs,bsd->bd", k, M))
                M = M + k[:, :, None] * e[:, None, :]
                o = np.einsum("bs,bsd->bd", q, M)
            elif m.memory == "banks":
                w, r = _softmax(a_w), _softmax(a_r)
                if m.write_gate:
                    g_in = pre if m.write_gate_input == "state" else a_w
                    w = w / (1 + np.exp(-(g_in @ c(m._w("write_gate_w", n)) + c(m._w("write_gate_b", n)))))
                M = mu[None, :, None] * M * keep[..., None] + ((1 - mu) * w)[:, :, None] * v[:, None, :]
                Z = mu[None] * Z * keep + (1 - mu) * w
                o = np.einsum("bs,bsd->bd", r, M) / (r * Z).sum(-1, keepdims=True)
            else:
                mu = float(mu)
                M = mu * M * keep[..., None] + unit(a_w)[:, :, None] * v[:, None, :]
                o = np.sqrt(1 - mu ** 2) * np.einsum("bf,bfd->bd", unit(a_r), M)
            z = gate_in(s, a_r) @ wg + bg
            g = 0.5 * (1 + np.tanh(z)) if m.read_mix == "tanh" else 1 / (1 + np.exp(-z))
            oo.append(o * g)
            gg.append(g)
            if m.read_mix == "gates":
                kk.append(1 / (1 + np.exp(-(gate_in(s, a_r) @ c(m._w("keep_state_w", n)) + c(m._w("keep_state_b", n))))))
            ss.append(s)
        s, o, g = np.stack(ss, 1), np.stack(oo, 1), np.stack(gg, 1)
        out.append((s, o))
        dt = m.token_embedding.embeddings.value.dtype  # the model's own dtype (float64 runs stay exact)
        hs = {"add": lambda: s + o, "tanh": lambda: (1 - g) * s + o, "gates": lambda: np.stack(kk, 1) * s + o}
        h = torch.as_tensor(hs[m.read_mix](), dtype=dt, device=dev)
        x_in = torch.as_tensor(inp, dtype=dt, device=dev)
        inp = c(m._mlp(n, h, h if m.mlp_residual == "state" else x_in))
    return out


class TestRopeStateMemory(unittest.TestCase):
    def test_parallel_matches_recurrence(self):
        """Every position, chunk sizes that do and do not divide N, resets mid-row, 1 and 2 layers, with and
        without the pre-norm."""
        for layers, pre_norm, memory, extra in ((1, True, "banks", {}), (2, True, "banks", {}), (2, False, "banks", {}),
                                                (1, True, "banks", {"write_gate_input": "logits"}),
                                                (2, True, "banks", {"read_mix": "add", "mlp_residual": "input"}),
                                                (2, True, "banks", {"read_mix": "tanh", "mlp_residual": "mix"}),
                                                (1, True, "banks", {"read_gate_input": "logits"}),
                                                (2, True, "linear", {}), (1, True, "banks_delta", {}),
                                                (2, True, "banks_delta", {})):
            m, rng = _model(layers, pre_norm=pre_norm, memory=memory, **extra)
            N = 3 * T + 2
            x = rng.integers(1, VOCAB, (3, N)).astype("int32")
            x[0, 0] = 0
            x[0, 9] = x[1, 4] = x[1, 5] = x[2, 8] = 0  # packed documents, an empty one, a marker at a chunk start
            ref = _reference(m, x)
            for chunk in (1, 3, T, N):
                m.chunk_size = m.delta_chunk = chunk
                with self.subTest(layers=layers, pre_norm=pre_norm, memory=memory, chunk=chunk, **extra):
                    for li, ((s, o), (rs, ro)) in enumerate(zip(m.trace(ops.convert_to_tensor(x)), ref)):
                        # layer 1: float32 rounding; later layers read a float32 layer output through an RMSNorm,
                        # which magnifies it where the random test network's output is small (exactness: float64 test)
                        rtol, atol = (1e-4, 1e-5) if li == 0 else (2e-3, 5e-4)
                        np.testing.assert_allclose(ops.convert_to_numpy(s), rs, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(ops.convert_to_numpy(o), ro, rtol=rtol, atol=atol)

    def test_exact_in_float64(self):
        """The default mixing (tanh read mix, MLP mix head), every memory, 2 layers, in float64: the parallel paths
        equal the token-by-token reference and give the same gradients for any chunk size, to rounding (float32
        tests compare at float32 precision, which deep random networks magnify)."""
        import keras

        from model import device as dev_mod

        before = dev_mod._resolved_device
        dev_mod.setup_device("cpu")  # the Iris Xe has no float64
        keras.config.set_dtype_policy("float64")
        try:
            for memory in ("banks", "linear", "banks_delta"):
                m, rng = _model(2, memory=memory)
                self.assertEqual(m.token_embedding.embeddings.value.dtype, torch.float64)
                x = rng.integers(1, VOCAB, (3, 3 * T + 2)).astype("int32")
                x[0, 0] = 0
                x[0, 9] = x[1, 4] = x[1, 5] = x[2, 8] = 0
                ref = _reference(m, x)
                grads = []
                for chunk in (1, 3, T):
                    m.chunk_size = m.delta_chunk = chunk
                    with self.subTest(memory=memory, chunk=chunk):
                        for (s_, o), (rs, ro) in zip(m.trace(ops.convert_to_tensor(x)), ref):
                            np.testing.assert_allclose(ops.convert_to_numpy(s_), rs, rtol=1e-9, atol=1e-11)
                            np.testing.assert_allclose(ops.convert_to_numpy(o), ro, rtol=1e-9, atol=1e-11)
                    for v in m.trainable_weights:
                        v.value.grad = None
                    y = m.dense_hidden(ops.convert_to_tensor(x))
                    (y * torch.linspace(-1, 1, y.shape[-1], device=y.device, dtype=y.dtype)).sum().backward()
                    grads.append([None if v.value.grad is None else v.value.grad.cpu().clone() for v in m.trainable_weights])
                for v, *gs in zip(m.trainable_weights, *grads):
                    if gs[0] is not None:
                        for g in gs[1:]:
                            np.testing.assert_allclose(g.numpy(), gs[0].numpy(), rtol=1e-8, atol=1e-10)
        finally:
            keras.config.set_dtype_policy("float32")
            dev_mod.setup_device(before)

    def test_read_gate_starts_closed(self):
        """Fresh model: w_g = 0, so the memory read's weight starts at sigmoid(-6) = 0.0025 (b_g = -3 under tanh)."""
        m = build_model({"arch": "rope-state-memory", "arch_params": {"vocabulary_size": VOCAB, "dim": 16, "window_size": T,
                                                                       "key_dim": 6, "mlp_hidden": 8}})
        self.assertEqual(float(np.abs(ops.convert_to_numpy(m.read_gate_w)).max()), 0.0)
        self.assertEqual(tuple(m.read_gate_w.shape), (m.dim, 1))  # from the state after the update
        q = torch.randn(3, m.read_gate_dim, device=m.read_gate_w.value.device)
        np.testing.assert_allclose(m._gate(0, q).detach().cpu().numpy(), 1 / (1 + np.exp(6.0)), rtol=1e-5)
        # excluded from weight decay the way model.train does it (Keras matches the variable name)
        import keras

        opt = keras.optimizers.AdamW(weight_decay=0.01)
        opt.exclude_from_weight_decay(var_names=list(m.no_weight_decay_names))
        opt.build(m.trainable_weights)
        self.assertFalse(opt._use_weight_decay(m.read_gate_w))
        self.assertFalse(opt._use_weight_decay(m.read_gate_b))
        self.assertTrue(opt._use_weight_decay(m.value.kernel))

    def test_write_gate(self):
        """Banks: the write gate starts closed (w_w = 0, b_w = -6) and out of weight decay; the read is divided by
        the written mass, so a gate equal for every token changes nothing (the fresh gate is a no-op; a token-
        dependent gate is checked against the reference in `test_parallel_matches_recurrence`). A model saved
        before the gate existed (no `write_gate_bias` in its config) loads without it."""
        p = {"vocabulary_size": VOCAB, "dim": 16, "window_size": T, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8}
        m = build_model({"arch": "rope-state-memory", "arch_params": p})
        self.assertTrue(m.write_gate)
        self.assertEqual(float(np.abs(ops.convert_to_numpy(m.write_gate_w)).max()), 0.0)
        self.assertEqual(tuple(m.write_gate_w.shape), (m.dim, 1))  # its own projection of the state before the update
        a = torch.randn(3, m.dim, device=m.write_gate_w.value.device)
        np.testing.assert_allclose(m._write_gate(0, a).detach().cpu().numpy(), 1 / (1 + np.exp(6.0)), rtol=1e-5)
        import keras

        opt = keras.optimizers.AdamW(weight_decay=0.01)
        opt.exclude_from_weight_decay(var_names=list(m.no_weight_decay_names))
        opt.build(m.trainable_weights)
        self.assertFalse(opt._use_weight_decay(m.write_gate_w))
        self.assertFalse(opt._use_weight_decay(m.write_gate_b))

        m, rng = _model(1)
        m.write_gate_w.assign(np.zeros(tuple(m.write_gate_w.shape), "float32"))  # the same gate for every token
        x = rng.integers(1, VOCAB, (2, 3 * T)).astype("int32")
        x[:, 0] = 0
        o_gated = ops.convert_to_numpy(m.trace(ops.convert_to_tensor(x))[0][1])
        cfg = {**m.get_config(), "write_gate_bias": None}
        plain = type(m).from_config(cfg)
        for v in plain.weights:  # same weights, no gate
            v.assign(next(w for w in m.weights if w.path == v.path))
        np.testing.assert_allclose(ops.convert_to_numpy(plain.trace(ops.convert_to_tensor(x))[0][1]), o_gated,
                                   rtol=1e-4, atol=1e-6)
        old = {k: v for k, v in m.get_config().items() if k not in ("write_gate_bias", "write_gate_input")}
        self.assertFalse(type(m).from_config(old).write_gate)
        first = {**old, "write_gate_bias": -6.0}  # saved with the first gate (from the write logits)
        self.assertEqual(type(m).from_config(first).write_gate_input, "logits")
        self.assertEqual(type(m).from_config(m.get_config()).write_gate_input, "state")

    def test_fresh_model_is_the_bigram(self):
        """Zero-init MLP outputs: a fresh model of any depth is logits = W_out^T RMSNorm(embedding(last token))
        (W_out^T embedding, pure linear, without `pre_norm`); embedding std 0.02; the norm scales and the read gate
        are out of weight decay."""
        for layers, pre_norm in ((1, True), (2, True), (4, True), (4, False)):
            m = build_model({"arch": "rope-state-memory", "arch_params": {
                "vocabulary_size": VOCAB, "dim": 16, "window_size": T, "key_dim": 6, "mlp_hidden": 8,
                "num_layers": layers, "pre_norm": pre_norm, "mlp_residual": "input"}})
            x = np.random.default_rng(layers).integers(0, VOCAB, (3, T)).astype("int32")
            E = ops.convert_to_numpy(m.token_embedding.embeddings)
            W = ops.convert_to_numpy(m.readout.kernel)
            e = E[x[:, -1]]
            if pre_norm:
                e = e / np.sqrt((e ** 2).mean(-1, keepdims=True) + 1e-6)
            np.testing.assert_allclose(ops.convert_to_numpy(m(ops.convert_to_tensor(x))), e @ W, rtol=1e-4, atol=1e-5)
            self.assertAlmostEqual(float(np.std(E)), 0.02, delta=0.004)
        import keras

        opt = keras.optimizers.AdamW(weight_decay=0.01)
        opt.exclude_from_weight_decay(var_names=list(m.no_weight_decay_names))
        opt.build(m.trainable_weights)
        for v in m.trainable_weights:
            with self.subTest(weight=v.path):
                decayed = not any(k in v.path.split("/")[-1]
                                  for k in ("decay_", "read_gate", "write_gate", "mix_head", "keep_", "mlp_gate",
                                            "scale", "address_"))
                self.assertEqual(opt._use_weight_decay(v), decayed)

    def test_mlp_residual_carries_the_input(self):
        """y = x + MLP(RMSNorm(s + g o)): with the MLP output zeroed, every layer passes its input through unchanged
        ("input"); with "state" it passes h = s + g o."""
        for mode in ("input", "state"):
            m, rng = _model(2, mlp_residual=mode, read_mix="add")
            for n in range(2):
                for v in (m._w("mlp_out", n).kernel, m._w("mlp_out", n).bias):
                    v.assign(np.zeros(tuple(v.shape), "float32"))
            x = rng.integers(1, VOCAB, (2, T)).astype("int32")
            x[:, 0] = 0
            y = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
            if mode == "input":
                np.testing.assert_allclose(y, ops.convert_to_numpy(m.token_embedding.embeddings)[x], rtol=1e-5, atol=1e-6)
            else:
                s, o = (ops.convert_to_numpy(v) for v in m.trace(ops.convert_to_tensor(x))[1])
                np.testing.assert_allclose(y, s + o, rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            _model(1, mlp_residual="both")
        with self.assertRaises(ValueError):
            _model(1, read_mix="max")
        with self.assertRaises(ValueError):
            _model(1, memory="delta")

    def test_tanh_read_mix_and_mlp_mix(self):
        """read_mix "tanh": c = -1 gives h = s, c = +1 gives h = o (the read replaces the state). mlp_residual "mix":
        y = (1 - c2)/2 x + (1 + c2)/2 m, x the layer input, m = MLP(RMSNorm(h)), c2 = tanh(w_m . RMSNorm(m) + b_m);
        a fresh model starts with the MLP's weight at 0.0025: y = (1 - tanh(-3))/2 x (m = 0). Configs saved before these options load as "add" / "input"."""
        m, rng = _model(1, read_mix="tanh", mlp_residual="mix")
        x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
        x[:, 0] = 0
        ids = torch.as_tensor(x, device=m.token_embedding.embeddings.value.device)
        E = m.token_embedding.embeddings.value
        with torch.no_grad():
            u = m._norm(m.in_norm, torch.nn.functional.embedding(ids.long(), E))
            zero = (E.new_zeros(2, m.dim), E.new_zeros(2, m.memory_rows, m.memory_cols))
            m.read_gate_w.assign(np.zeros(tuple(m.read_gate_w.shape), "float32"))
            for bias, want in ((30.0, "o"), (-30.0, "s")):
                m.read_gate_b.assign(np.array([bias], "float32"))
                h, _, (s, o) = m._layer(0, u, ids == 0, zero, collect=True, fresh=True)
                np.testing.assert_allclose(h.cpu().numpy(), (o if want == "o" else s).cpu().numpy(), rtol=1e-5, atol=1e-6)
            # the MLP mix, against the MLP written out
            hr = torch.randn(2, T, m.dim, device=E.device)
            xr = torch.randn(2, T, m.dim, device=E.device)
            z = hr * torch.rsqrt(hr.pow(2).mean(-1, keepdim=True) + 1e-6) * m.mlp_norm.scale.value
            out = torch.nn.functional.gelu(z @ m.mlp_in.kernel.value + m.mlp_in.bias.value, approximate="tanh")
            out = out @ m.mlp_out.kernel.value + m.mlp_out.bias.value
            c = torch.tanh(out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6) @ m.mix_head_w.value
                           + m.mix_head_b.value)
            want = (1 - c) / 2 * xr + (1 + c) / 2 * out
            np.testing.assert_allclose(m._mlp(0, hr, xr).cpu().numpy(), want.cpu().numpy(), rtol=1e-4, atol=1e-5)
        fresh = build_model({"arch": "rope-state-memory", "arch_params": {
            "vocabulary_size": VOCAB, "dim": 16, "window_size": T, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8,
            "read_mix": "tanh", "mlp_residual": "mix"}})
        self.assertEqual(tuple(fresh.mlp_out.kernel.shape), (8, 16))
        self.assertEqual(tuple(fresh.mix_head_w.shape), (16, 1))
        with torch.no_grad():
            ids = torch.as_tensor(x, device=fresh.token_embedding.embeddings.value.device)
            Ef = fresh.token_embedding.embeddings.value
            xf = torch.nn.functional.embedding(ids.long(), Ef)
            np.testing.assert_allclose(ops.convert_to_numpy(fresh.dense_hidden(ops.convert_to_tensor(x))),
                                       0.5 * (1 - np.tanh(-3.0)) * xf.cpu().numpy(), rtol=1e-5, atol=1e-6)
            gw = fresh._gate(0, torch.randn(3, fresh.read_gate_dim, device=Ef.device))  # the memory's weight in h
            np.testing.assert_allclose(gw.cpu().numpy(), 1 / (1 + np.exp(6.0)), rtol=1e-5)
        old = {k: v for k, v in m.get_config().items() if k not in ("read_mix", "mlp_residual", "read_gate_input")}
        loaded = type(m).from_config(old)
        self.assertEqual((loaded.read_mix, loaded.mlp_residual, loaded.read_gate_input), ("add", "input", "logits"))
        self.assertEqual(type(m).from_config(m.get_config()).read_gate_input, "state")
        old_state = {**old, "mlp_residual": "state"}
        self.assertEqual(type(m).from_config(old_state).mlp_residual, "state")

    def test_gates(self):
        """The default: two independent gates per mix. h = a s + g o (a keeps the state, g takes the read; a -> 0,
        g -> 1 replaces the state), y = a2 x + g2 m (a2 keeps the input, g2 takes the MLP output, both from
        RMSNorm(m)). Fresh: a, a2 = sigmoid(6), g, g2 = sigmoid(-6), m = 0, so y = sigmoid(6) x."""
        m, rng = _model(1)
        self.assertEqual((m.read_mix, m.mlp_residual), ("gates", "gates"))
        x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
        x[:, 0] = 0
        E = m.token_embedding.embeddings.value
        ids = torch.as_tensor(x, device=E.device)
        with torch.no_grad():
            u = m._norm(m.in_norm, torch.nn.functional.embedding(ids.long(), E))
            zero = (E.new_zeros(2, m.dim), E.new_zeros(2, m.memory_rows, m.memory_cols))
            for v in (m.read_gate_w, m.keep_state_w):
                v.assign(np.zeros(tuple(v.shape), "float32"))
            for keep, take, want in ((30.0, -30.0, "s"), (-30.0, 30.0, "o"), (30.0, 30.0, "s+o")):
                m.keep_state_b.assign(np.array([keep], "float32"))
                m.read_gate_b.assign(np.array([take], "float32"))
                h, _, (s_, o) = m._layer(0, u, ids == 0, zero, collect=True, fresh=True)
                ref = {"s": s_, "o": o, "s+o": s_ + o}[want]
                np.testing.assert_allclose(h.cpu().numpy(), ref.cpu().numpy(), rtol=1e-5, atol=1e-6)
            hr = torch.randn(2, T, m.dim, device=E.device)
            xr = torch.randn(2, T, m.dim, device=E.device)
            z = hr * torch.rsqrt(hr.pow(2).mean(-1, keepdim=True) + 1e-6) * m.mlp_norm.scale.value
            out = torch.nn.functional.gelu(z @ m.mlp_in.kernel.value + m.mlp_in.bias.value, approximate="tanh")
            out = out @ m.mlp_out.kernel.value + m.mlp_out.bias.value
            on = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)
            a2 = torch.sigmoid(on @ m.keep_input_w.value + m.keep_input_b.value)
            g2 = torch.sigmoid(on @ m.mlp_gate_w.value + m.mlp_gate_b.value)
            np.testing.assert_allclose(m._mlp(0, hr, xr).cpu().numpy(), (a2 * xr + g2 * out).cpu().numpy(),
                                       rtol=1e-4, atol=1e-5)
        fresh = build_model({"arch": "rope-state-memory", "arch_params": {
            "vocabulary_size": VOCAB, "dim": 16, "window_size": T, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8}})
        Ef = fresh.token_embedding.embeddings.value
        ids = torch.as_tensor(x, device=Ef.device)
        with torch.no_grad():
            want = 1 / (1 + np.exp(-6.0)) * torch.nn.functional.embedding(ids.long(), Ef).cpu().numpy()
            np.testing.assert_allclose(ops.convert_to_numpy(fresh.dense_hidden(ops.convert_to_tensor(x))), want,
                                       rtol=1e-5, atol=1e-6)
            q = torch.randn(3, fresh.read_gate_dim, device=Ef.device)
            np.testing.assert_allclose(fresh._gate(0, q).cpu().numpy(), 1 / (1 + np.exp(6.0)), rtol=1e-5)
            np.testing.assert_allclose(fresh._keep_state(0, q).cpu().numpy(), 1 / (1 + np.exp(-6.0)), rtol=1e-5)

    def test_bank_init_is_uniform_and_power_law(self):
        """Fresh banks: W_q = W_k = 0 (both softmaxes exactly uniform), half-lives log-spaced from min to max, so
        the memory's weight on a token k back, sum_b (1 - mu_b) mu_b^k / S, falls like 1/k in between; the state
        starts at a half-life of 2 tokens."""
        m = build_model({"arch": "rope-state-memory", "arch_params": {"vocabulary_size": VOCAB, "dim": 16,
                                                                       "window_size": T, "key_dim": 6}})
        self.assertEqual(m.memory, "banks")
        lam, mu = (ops.convert_to_numpy(d).astype("float64") for d in m.decays())
        self.assertAlmostEqual(float(lam) ** 2, 0.5, places=5)
        np.testing.assert_allclose(np.log(0.5) / np.log(mu), np.geomspace(1, 1024, 128), rtol=1e-4)
        self.assertEqual(float(np.abs(ops.convert_to_numpy(m.address_q)).max()), 0.0)
        self.assertEqual(tuple(m.address_k.shape), (16, 128))
        dev = m.address_q.value.device
        read, write = (torch.zeros(2, 5, 128, device=dev) for _ in range(2))
        r, w, _ = m._addresses(0, read, write)
        np.testing.assert_allclose(r.detach().cpu().numpy(), 1 / 128, rtol=1e-6)
        np.testing.assert_allclose((w / (1 - torch.as_tensor(mu, dtype=w.dtype, device=w.device))).detach().cpu().numpy(),
                                   1 / 128, rtol=1e-5)
        # the read is normalized: at init a power-law-weighted average of the past values, at their scale
        x = np.random.default_rng(1).integers(1, VOCAB, (2, 40)).astype("int32")
        x[:, 0] = 0
        for v in (m.read_gate_b,):
            v.assign(np.array([30.0], "float32"))  # gate open
        s, o = (ops.convert_to_numpy(a) for a in m.trace(ops.convert_to_tensor(x))[0])
        u = ops.convert_to_numpy(m._norm(m.in_norm, torch.as_tensor(ops.convert_to_numpy(m.token_embedding.embeddings)[x],
                                                                      device=dev)))
        vv = u @ ops.convert_to_numpy(m.value.kernel)
        self.assertLess(np.linalg.norm(o, axis=-1).max(), np.linalg.norm(vv, axis=-1).max() + 1e-4)
        self.assertGreater(np.linalg.norm(o[:, -1], axis=-1).min(), 0.1 * np.linalg.norm(vv, axis=-1).mean())
        k = np.array([8, 32, 128])
        kernel = ((1 - mu)[None] * mu[None] ** k[:, None]).mean(1)
        self.assertLess(np.ptp(k * kernel) / np.mean(k * kernel), 0.3)  # k * weight(k) nearly constant: ~ 1/k

    def test_lru_normalization(self):
        m, _ = _model(memory="linear")
        lam, mu = (float(ops.convert_to_numpy(d)) for d in m.decays())
        fresh = build_model({"arch": "rope-state-memory", "arch_params": {
            "vocabulary_size": VOCAB, "dim": 16, "window_size": T, "state_half_life": 4.0, "memory_half_life": 32.0,
            "memory": "linear"}})
        lam0, mu0 = (float(ops.convert_to_numpy(d)) for d in fresh.decays())
        self.assertAlmostEqual(lam0 ** 4, 0.5, places=5)
        self.assertAlmostEqual(mu0 ** 32, 0.5, places=5)
        # a constant input sums to about its own size: gamma = sqrt(1 - lambda^2) of a real decay
        x = np.full((1, 400), 7, "int32")
        s, _ = m.trace(ops.convert_to_tensor(x))[0]
        e = ops.convert_to_numpy(m.token_embedding.embeddings)[7]
        u = e / np.sqrt((e ** 2).mean() + 1e-6) * ops.convert_to_numpy(m.in_norm.scale)  # what the state reads
        self.assertLess(np.linalg.norm(ops.convert_to_numpy(s)[0, -1]), np.linalg.norm(u) * np.sqrt((1 + lam) / (1 - lam)) + 1e-5)

    def test_dense_is_every_prefix(self):
        for layers in (1, 2):
            m, rng = _model(layers)
            m.chunk_size = 3
            x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
            x[:, 0] = 0
            x[1, 6] = 0
            dense = ops.convert_to_numpy(m.dense_hidden(ops.convert_to_tensor(x)))
            for i in range(x.shape[1]):  # left-padded with markers, as the validation windows are
                window = np.zeros_like(x)
                window[:, -(i + 1):] = x[:, :i + 1]
                np.testing.assert_allclose(dense[:, i], ops.convert_to_numpy(m.hidden(ops.convert_to_tensor(window))),
                                           rtol=1e-4, atol=1e-5)

    def test_dense_hidden_valid_is_dense_at_targets(self):
        """The last layer's output mix at the target positions only == the dense output there, for every
        `mlp_residual` (the earlier layers still run everywhere)."""
        for residual in ("gates", "input", "state", "mix"):
            m, rng = _model(2, mlp_residual=residual)
            x = ops.convert_to_tensor(rng.integers(1, VOCAB, (2, 2 * T)).astype("int32"))
            valid = torch.as_tensor(rng.random((2, 2 * T)) < 0.4, device=m.token_embedding.embeddings.value.device)
            with torch.no_grad():
                want = m.dense_hidden(x)[valid].cpu().numpy()
                got = m.dense_hidden_valid(x, valid).cpu().numpy()
            with self.subTest(residual=residual):
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_gradient_accumulation_is_one_batch(self):
        """4 rows in one step == 4 one-row micro-batches accumulated (rows with different target counts): the same
        update (SGD, so the update is the gradient itself)."""
        import keras as K

        from model.losses import SparseCategoricalCrossentropyFromLogits

        rng = np.random.default_rng(1)
        x = rng.integers(1, VOCAB, (4, 2 * T)).astype("int32")
        x[:, 0] = 0
        y = rng.integers(1, VOCAB, (4, 2 * T)).astype("int32")
        y[rng.random(y.shape) < 0.5] = -1
        y[3, :-2] = -1  # a row with 2 targets
        y[3, -2:] = (5, 7)
        after, init = [], None
        for accum in (1, 4):
            m, _ = _model(2)
            if init is None:  # _model seeds only some weights: start both runs from the same ones
                init = [ops.convert_to_numpy(v.value).copy() for v in m.weights]
            else:
                for v, a in zip(m.weights, init):
                    v.assign(a)
            m.compile(optimizer=K.optimizers.SGD(0.5), loss=SparseCategoricalCrossentropyFromLogits())
            m.dense_supervision_enabled = True
            m.grad_accumulation = accum
            batches = [(x, y)] if accum == 1 else [(x[i:i + 1], y[i:i + 1]) for i in range(4)]
            for xb, yb in batches:
                m.train_step((ops.convert_to_tensor(xb), ops.convert_to_tensor(yb)))
            self.assertEqual(int(m.optimizer.iterations.value), 1)
            after.append([ops.convert_to_numpy(v.value).copy() for v in m.trainable_weights])
        for a, b in zip(*after):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
        # a micro-batch without targets: no gradient, but it still counts toward the update
        m, _ = _model(2)
        m.compile(optimizer=K.optimizers.SGD(0.5), loss=SparseCategoricalCrossentropyFromLogits())
        m.dense_supervision_enabled, m.grad_accumulation = True, 2
        before = [ops.convert_to_numpy(v.value).copy() for v in m.trainable_weights]
        m.train_step((ops.convert_to_tensor(x[:1]), ops.convert_to_tensor(np.full_like(y[:1], -1))))
        self.assertEqual(int(m.optimizer.iterations.value), 0)
        m.train_step((ops.convert_to_tensor(x[1:2]), ops.convert_to_tensor(y[1:2])))
        self.assertEqual(int(m.optimizer.iterations.value), 1)
        self.assertFalse(all(np.array_equal(a, ops.convert_to_numpy(v.value)) for a, v in zip(before, m.trainable_weights)))

    def test_stateful_generation_matches_full_context(self):
        for layers in (1, 2):
            m, _ = _model(layers)
            prompt = [0, 5, 9, 3]
            full = m.with_context_length(64)
            self.assertEqual(m.generate(prompt, max_new_tokens=12), full.generate(prompt, max_new_tokens=12))
            self.assertFalse(full.stateful_generation)

    def test_gradients_match_recurrence(self):
        """Chunked and one-token-chunk runs give the same gradients for every weight (the delta rule's chunked
        WY form included)."""
        for memory in ("banks", "banks_delta"):
            self._gradients_chunks(memory)

    def _gradients_chunks(self, memory):
        m, rng = _model(2, memory=memory)
        x = rng.integers(1, VOCAB, (2, 2 * T + 1)).astype("int32")
        x[:, 0] = 0
        x[0, 5] = 0
        grads = []
        for chunk in (1, 4):
            m.chunk_size = m.delta_chunk = chunk
            for v in m.trainable_weights:
                v.value.grad = None
            y = m.dense_hidden(ops.convert_to_tensor(x))
            (y * torch.linspace(-1, 1, y.shape[-1], device=y.device)).sum().backward()
            grads.append([v.value.grad.cpu().clone() if v.value.grad is not None else None for v in m.trainable_weights])
        for v, a, b in zip(m.trainable_weights, *grads):
            with self.subTest(memory=memory, weight=v.path):
                if a is None:
                    self.assertIsNone(b)
                    continue
                # float32 on the XPU: reductions sum in a varying order (exactness across chunk sizes: float64 test)
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3,
                                           atol=1e-4 * max(1.0, float(np.abs(b.numpy()).max())))

    def test_training_run_and_reload(self):
        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7, 20], validation_doc_lengths=[6, 11]):
            cfg = tiny_config("rsm", arch="rope-state-memory")
            cfg["arch_params"].update({"dim": 16, "key_dim": 6, "mlp_hidden": 8, "chunk_size": 3, "num_layers": 2,
                                       "num_banks": 8})
            cfg["training"]["dense_supervision"] = True
            cfg_mod.save("rsm", cfg)
            history = train("rsm", verbose=0)
            self.assertTrue(np.isfinite(history.history["val_loss"][-1]))
            history = train("rsm", epochs=2, verbose=0)  # resume from model.keras
            self.assertTrue(np.isfinite(history.history["val_loss"][-1]))


    def test_tied_embeddings(self):
        """tie_embeddings: the readout is the embedding table transposed, no readout weights; the fused readout loss puts
        both gradients into the one table (= autograd of plain logits); saved and loaded; old configs load untied."""
        from model.fused_linear_cross_entropy import fused_linear_cross_entropy

        m, rng = _model(1, tie_embeddings=True)
        self.assertTrue(m.tie_embeddings)
        self.assertFalse(any("readout" in v.path for v in m.weights))
        self.assertEqual(sum("embeddings" in v.path for v in m.weights), 1)
        E = m.token_embedding.embeddings
        np.testing.assert_array_equal(ops.convert_to_numpy(m.readout.kernel), ops.convert_to_numpy(E).T)
        x = rng.integers(1, VOCAB, (2, 2 * T)).astype("int32")
        x[:, 0] = 0
        y = torch.as_tensor(rng.integers(1, VOCAB, (2 * 2 * T,)), device=E.value.device)

        def grad(fused):
            for v in m.trainable_weights:
                v.value.grad = None
            h = m.readout_input(m.dense_hidden(ops.convert_to_tensor(x))).reshape(-1, m.dim)
            if fused:
                loss = fused_linear_cross_entropy(h, m.readout.kernel, y, chunk_size=7)
            else:
                loss = torch.nn.functional.cross_entropy(m.readout(h), y)
            loss.backward()
            return E.value.grad.cpu().clone(), float(loss)

        (g1, l1), (g2, l2) = grad(True), grad(False)
        self.assertAlmostEqual(l1, l2, places=4)
        np.testing.assert_allclose(g1.numpy(), g2.numpy(), rtol=1e-4, atol=1e-6)
        old = {k: v for k, v in m.get_config().items() if k != "tie_embeddings"}
        self.assertFalse(type(m).from_config(old).tie_embeddings)
        import os
        import tempfile

        import keras

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            m.save(path)
            back = keras.saving.load_model(path)
            self.assertTrue(back.tie_embeddings)
            np.testing.assert_array_equal(ops.convert_to_numpy(back.readout.kernel), ops.convert_to_numpy(m.readout.kernel))
            np.testing.assert_allclose(ops.convert_to_numpy(back(ops.convert_to_tensor(x))),
                                       ops.convert_to_numpy(m(ops.convert_to_tensor(x))), rtol=1e-5, atol=1e-6)

    def test_init_weights_from(self):
        """`init_weights_from` copies the named layers from another run (different arch config) into a fresh run."""
        from model.train import build_or_load

        with temp_workspace(train_doc_lengths=[5, 3, 9, 4, 7, 20], validation_doc_lengths=[6, 11]):
            src = tiny_config("src", arch="rope-state-memory")
            src["arch_params"].update({"dim": 16, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8,
                                       "read_mix": "add", "mlp_residual": "input", "tie_embeddings": False})
            src["training"]["dense_supervision"] = True
            cfg_mod.save("src", src)
            train("src", verbose=0)
            dst = tiny_config("dst", arch="rope-state-memory")
            dst["arch_params"].update({"dim": 16, "key_dim": 6, "mlp_hidden": 8, "num_banks": 8, "tie_embeddings": False})
            dst["init_weights_from"] = {"run": "src", "checkpoint": "latest", "weights": ["token_embedding", "readout"]}
            cfg_mod.save("dst", dst)
            from model.evaluate import load_checkpoint

            m, epoch = build_or_load("dst", cfg_mod.load("dst"), verbose=0)
            s_ = load_checkpoint("src", "latest")
            self.assertEqual(epoch, 0)
            for a_, b_ in ((m.token_embedding.embeddings, s_.token_embedding.embeddings), (m.readout.kernel, s_.readout.kernel)):
                np.testing.assert_array_equal(ops.convert_to_numpy(a_), ops.convert_to_numpy(b_))
            self.assertEqual(float(np.abs(ops.convert_to_numpy(m.address_k)).max()), 0.0)  # the rest: own init
            dst["init_weights_from"]["weights"] = ["no_such_layer"]
            cfg_mod.save("dst", dst)
            with self.assertRaises(ValueError):
                build_or_load("dst", cfg_mod.load("dst"), verbose=0)


class TestRopeStateMemoryKernels(unittest.TestCase):
    """The custom XPU scans == the chunked torch path: outputs, end carries and every gradient."""

    def setUp(self):
        from model.device import setup_device
        from model.kernels import rsm_scan

        setup_device("xpu")  # a training test before this one may have pinned the CPU; falls back to CPU if no XPU
        m, _ = _model(2, key_dim=16, dim=32, num_banks=16)
        if not rsm_scan.available(m.token_embedding.embeddings.value):
            self.skipTest(f"XPU kernels unavailable: {rsm_scan.error()}")

    def _pair(self, layers=2, **extra):
        # dim 32: the bank memory's padded values are 48 wide (the 3-row-block kernel) and its dq / dk scans 48 rows.
        # The linear and delta memories under the default tanh read mix: the read can replace the state, and their
        # small reads then magnify float32 rounding ~25x / ~2.5x through the random 2-layer net (kernel vs torch
        # 4e-4 / 5e-5 of a gradient's size, against 1-2e-5 with "add"); the kernels are the same either way, so
        # they are compared under the added read, the bank memory under the default
        if extra.get("memory") in ("linear", "banks_delta"):
            extra = {"read_mix": "add", "mlp_residual": "input", **extra}
        m, rng = _model(layers, **{"key_dim": 16, "dim": 32, "num_banks": 16, **extra})
        x = rng.integers(1, VOCAB, (3, 5 * T + 3)).astype("int32")
        x[:, 0] = 0
        x[0, 9] = x[1, 4] = x[1, 5] = x[2, 16] = 0
        return m, ops.convert_to_tensor(x)

    def _both(self, m, f):
        out = []
        for kernels in (True, False):
            m.use_kernels = kernels
            out.append(f())
        m.use_kernels = True
        return out

    def test_trace_matches_torch_path(self):
        for memory in ("banks", "linear", "banks_delta"):
            m, x = self._pair(memory=memory)
            m.chunk_size = 4
            got, ref = self._both(m, lambda: [tuple(t.detach().cpu().numpy() for t in st) for st in m.trace(x)])
            for (s, o), (rs, ro) in zip(got, ref):
                with self.subTest(memory=memory):
                    np.testing.assert_allclose(s, rs, rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(o, ro, rtol=1e-4, atol=1e-5)

    def test_carries_match_torch_path(self):
        """Generation: from a non-zero carried state, token by token and in blocks."""
        for memory in ("banks", "linear", "banks_delta"):
            self._carries(memory)

    def _carries(self, memory):
        m, x = self._pair(memory=memory)
        with torch.no_grad():
            def run():
                _, state = m.advance(x[:, :11])
                logits, state = m.advance(x[:, 11:12], state)
                logits2, state = m.advance(x[:, 12:20], state)
                return [logits.cpu().numpy(), logits2.cpu().numpy()] + [t.cpu().numpy() for pair in state for t in pair]
            got, ref = self._both(m, run)
        for a, b in zip(got, ref):  # float32, XPU reductions in a varying order: relative to each tensor's size
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=2e-5 * max(1.0, float(np.abs(b).max())))

    def test_gradients_match_torch_path(self):
        for memory in ("banks", "linear", "banks_delta"):
            self._gradients(memory)

    def test_wide_dims_match_torch_path(self):
        """dim 320: the row norms take the wide kernel (> 256), the bank values are 336 wide (not a row width of
        `outer_scan`), so dq / dk run as column read-outs (`outer_scan_cols`); 32 banks. A smoke test of the wiring
        (a wrong term is O(1)): sums over 320 columns and the gates leave the kernels ~1e-4 of a bias gradient from
        the torch path, and from themselves run to run (XPU reductions); the scan's precision is
        `test_memory_scans_exact`."""
        for memory in ("banks", "linear"):
            self._gradients(memory, tol=2e-3, dim=320, num_banks=32, mlp_hidden=64)
            m, x = self._pair(memory=memory, dim=320, num_banks=32, mlp_hidden=64)
            got, ref = self._both(m, lambda: [tuple(t.detach().cpu().numpy() for t in st) for st in m.trace(x)])
            for (s_, o), (rs, ro) in zip(got, ref):
                np.testing.assert_allclose(s_, rs, rtol=1e-3, atol=1e-4 * float(np.abs(rs).max()))
                np.testing.assert_allclose(o, ro, rtol=1e-3, atol=1e-4 * float(np.abs(ro).max()))

    def test_join_read_matches_autograd(self):
        """h = a s + g m[:D] / max(m[D], 1e-12), with a (read_mix "gates") and without (a = 1, "add"), a row with no
        written mass (the clamp): output and every gradient as autograd."""
        from model.kernels import rsm_scan

        rng = np.random.default_rng(4)
        t = lambda *shape: torch.as_tensor(rng.normal(size=shape).astype("float32"), device="xpu")
        R, D, M = 37, 320, 336
        s_, m_, g_, a_, dh = t(R, D), t(R, M), torch.rand(R, 1, device="xpu"), torch.rand(R, 1, device="xpu"), t(R, D)
        m_[:, D] = m_[:, D].abs() + 0.1
        m_[5, D] = 0.0
        for with_a in (False, True):
            got = []
            for fused in (True, False):
                leaves = [v.clone().requires_grad_() for v in (s_, m_, g_, a_)]
                ss, mm, gg, aa = leaves
                if fused:
                    h = rsm_scan.join_read(ss, mm, gg, aa if with_a else None)
                else:
                    o = mm[:, :D] / mm[:, D:D + 1].clamp_min(1e-12)
                    h = (aa * ss if with_a else ss) + gg * o
                (h * dh).sum().backward()
                got.append([h.detach()] + [v.grad if v.grad is not None else torch.zeros_like(v) for v in leaves])
            for name, a, b in zip(("h", "ds", "dm", "dg", "da"), *got):
                with self.subTest(with_a=with_a, value=name):
                    np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-4,
                                               atol=1e-5 * max(1.0, float(b.abs().max())))

    def test_memory_scans_exact(self):
        """The memory scans (`outer_scan`: out_t = Z_t^T gamma_t; `outer_scan_cols`: out_t = Z_t gamma_t) against
        the recurrence Z_t = dec Z_{t-1} keep_t + alpha_t beta_t^T in float64 (CPU), forward and reversed, a decay per
        row or per column, with document resets (two on segment boundaries), whole and time-segmented (8 segments),
        with the end state. Widths 336 (4 columns per lane) and 98 (2 per lane, several work-groups)."""
        from model.kernels import rsm_scan

        rng = np.random.default_rng(0)
        saved = rsm_scan.SEGMENT_ROWS, rsm_scan.SEGMENT_MIN
        try:
            for kind, A, C, dec_rows in (("cols", 32, 336, True), ("cols", 16, 98, True), ("rows", 32, 336, True),
                                         ("rows", 16, 48, False)):
                for reverse in (False, True):
                    B, N = 2, 64
                    al, be = rng.normal(size=(B, N, A)), rng.normal(size=(B, N, C))
                    ga = rng.normal(size=(B, N, C if kind == "cols" else A))
                    dec = rng.uniform(0.3, 0.99, A if dec_rows else C)
                    reset = np.zeros((B, N), np.uint8)
                    reset[0, 10] = reset[1, 3] = reset[1, 24] = reset[0, 32] = reset[1, 40] = 1  # 24, 32, 40: boundaries
                    want = np.zeros((B, N, A if kind == "cols" else C))
                    ends = np.zeros((B, A, C))
                    D = dec[:, None] if dec_rows else dec[None, :]
                    for b_ in range(B):
                        Z = np.zeros((A, C))
                        for t in (range(N - 1, -1, -1) if reverse else range(N)):
                            tk = t + 1 if reverse else t  # reversed: the reset of the step after it ends a document
                            keep = 0.0 if (tk < N and reset[b_, tk]) else 1.0
                            Z = D * keep * Z + np.outer(al[b_, t], be[b_, t])
                            want[b_, t] = Z @ ga[b_, t] if kind == "cols" else Z.T @ ga[b_, t]
                        ends[b_] = Z
                    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device="xpu").contiguous()
                    for rows, mn in ((1, 256), (16, 8)):  # whole; 8 segments of 8 steps
                        rsm_scan.SEGMENT_ROWS, rsm_scan.SEGMENT_MIN = rows, mn
                        out, end = rsm_scan._segmented(kind, f(al), f(be), f(ga), torch.as_tensor(reset, device="xpu"),
                                                       f(dec), torch.ones(1, device="xpu"), reverse, dec_rows, True)
                        with self.subTest(kind=kind, A=A, C=C, reverse=reverse, segmented=rows > 1):
                            np.testing.assert_allclose(out.cpu().numpy(), want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
                            np.testing.assert_allclose(end.cpu().numpy(), ends, rtol=1e-4, atol=1e-5 * np.abs(ends).max())
        finally:
            rsm_scan.SEGMENT_ROWS, rsm_scan.SEGMENT_MIN = saved

    def _gradients(self, memory, tol=2e-5, **extra):
        m, x = self._pair(memory=memory, **extra)

        def grads():
            for v in m.trainable_weights:
                v.value.grad = None
            y = m.dense_hidden(x)
            (y * torch.linspace(-1, 1, y.shape[-1], device=y.device)).sum().backward()
            return [None if v.value.grad is None else v.value.grad.cpu().clone() for v in m.trainable_weights]

        got, ref = self._both(m, grads)
        for v, a, b in zip(m.trainable_weights, got, ref):
            with self.subTest(memory=memory, weight=v.path):
                if a is None or b is None:  # the readout is not part of dense_hidden
                    self.assertTrue(a is None and b is None)
                    continue
                # float32 sums in a different order: entries of a gradient differ by up to ~1.2e-5 of its largest
                # entry (measured the same with either read / output mixing; torch vs torch at other chunk sizes: ~8e-6)
                atol = tol * max(1.0, float(np.abs(b.numpy()).max()))
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-3, atol=atol)

    def test_fused_norm_mlp_matches_autograd(self):
        """MLP(RMSNorm(x)), with and without the L2-normalized output, recomputing the hidden layer or keeping its
        pre-activations: same output and gradients as plain autograd, over several row blocks, for row widths 64,
        128, 192 and 320 (the wide kernel)."""
        from model.kernels import rsm_scan

        rng = np.random.default_rng(3)
        t = lambda *shape, scale=1.0: torch.as_tensor(rng.normal(0, scale, shape).astype("float32"), device="xpu")
        saved, rsm_scan.FusedNormMLP.rows = rsm_scan.FusedNormMLP.rows, 8
        try:
            for D, H, F, l2 in ((64, 24, 128, True), (128, 32, 128, False), (128, 36, 64, True), (320, 32, 64, False),
                                (192, 24, 320, True)):  # 320: the wide row-norm kernel (rms, then l2)
                ws = (t(D) + 1, t(D, H, scale=0.2), t(H), t(H, F, scale=0.2), t(F))
                x, dy = t(37, D), t(37, F)
                got = []
                for fused in ("recompute", "keep", None):
                    leaves = [v.clone().requires_grad_() for v in (x,) + ws]
                    if fused:
                        y = rsm_scan.norm_mlp(*leaves, l2_out=l2, keep_hidden=fused == "keep")
                    else:
                        xx, sc, ww1, bb1, ww2, bb2 = leaves
                        z = xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + 1e-6) * sc
                        y = torch.nn.functional.gelu(z @ ww1 + bb1, approximate="tanh") @ ww2 + bb2
                        if l2:
                            y = torch.nn.functional.normalize(y, dim=-1, eps=1e-6)
                    (y * dy).sum().backward()
                    got.append([y.detach()] + [v.grad for v in leaves])
                for k, mode in enumerate(("recompute", "keep")):
                    for a, b in zip(got[k], got[2]):
                        with self.subTest(D=D, l2=l2, mode=mode):
                            np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-4, atol=2e-5)
        finally:
            rsm_scan.FusedNormMLP.rows = saved

    def test_full_width_model_matches_torch_path(self):
        """dim 128, key_dim 64: every fused kernel in the model (norm MLPs too) against the torch path. Under the
        added read (the kernels are the same for every mixing; the default mixing's exactness: the float64 test)."""
        m, rng = _model(2, key_dim=64, dim=128, mlp_hidden=48, num_banks=64, read_mix="add", mlp_residual="input")
        x = rng.integers(1, VOCAB, (2, 3 * T + 1)).astype("int32")
        x[:, 0] = 0
        x[1, 5] = 0
        x = ops.convert_to_tensor(x)

        def run():
            for v in m.trainable_weights:
                v.value.grad = None
            y = m.dense_hidden(x)
            (y * torch.linspace(-1, 1, y.shape[-1], device=y.device)).sum().backward()
            return [y.detach().cpu()] + [None if v.value.grad is None else v.value.grad.cpu().clone()
                                         for v in m.trainable_weights]

        got, ref = self._both(m, run)
        for a, b in zip(got, ref):
            if a is None:
                self.assertIsNone(b)
                continue
            # XPU reductions sum in a varying order (torch and kernel paths alike vary ~1e-5 run to run on the same
            # inputs); through this random 2-layer net kernel vs torch measured up to 1.2e-4 of a tensor's size
            atol = 3e-4 * max(1.0, float(np.abs(b.numpy()).max()))
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-3, atol=atol)

    def test_carried_state_training_gradients(self):
        """Truncated BPTT: a row run from a carried (detached) state under grad == the same tokens inside one long
        row (forward), and its parameter gradients are the same with the custom kernels (carry added in torch) as
        on the pure torch path. A marker inside the second row cuts the carry."""
        rng = np.random.default_rng(3)
        ids = rng.integers(1, VOCAB, (2, 3 * T)).astype("int32")
        ids[0, 0] = ids[1, 0] = 0
        ids[1, 2 * T + 3] = 0
        grads = {}
        m, _ = _model(2)  # one model (its unseeded weights are random per build), the path toggled
        for kern in (False, True):
            m.use_kernels = kern
            with torch.no_grad():
                full = m._run(ids)[0][:, T:2 * T]
                _, ends, _ = m._run(ids[:, :T])
                part = m._run(ids[:, T:2 * T], ends)[0]
            np.testing.assert_allclose(part.cpu().numpy(), full.cpu().numpy(), rtol=2e-3, atol=2e-3)
            carry = [(s.detach(), z.detach()) for s, z in ends]
            y = m._run(ids[:, 2 * T:], [(s, z) for s, z in carry])[0]
            m.zero_grad()
            (y ** 2).sum().backward()
            grads[kern] = [np.zeros(v.shape) if v.value.grad is None else v.value.grad.cpu().numpy()
                           for v in m.trainable_weights]
        for a, b in zip(grads[False], grads[True]):
            np.testing.assert_allclose(b, a, rtol=3e-3, atol=3e-3 * max(1.0, float(np.abs(a).max())))


if __name__ == "__main__":
    unittest.main()
