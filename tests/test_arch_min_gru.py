"""`min-gru`: the chunked scan against a step-by-step loop (values and gradients), document resets, the valid-position path, the carried state (truncated BPTT) and token-by-token generation. CPU."""

import unittest

import numpy as np

from model.device import setup_device

setup_device("cpu")

import torch  # noqa: E402

from model.archs.min_gru import MinGRULM, linear_scan  # noqa: E402


def step_by_step(keep, write, initial):
    state, out = initial, []
    for t in range(keep.shape[1]):
        state = keep[:, t] * state + write[:, t]
        out.append(state)
    return torch.stack(out, 1)


def tiny(**kw):
    torch.manual_seed(0)
    return MinGRULM(**{"vocabulary_size": 50, "dim": 16, "hidden_size": 12, "num_layers": 2, "window_size": 40, **kw})


def ids(B=2, T=40, seed=0):
    return np.random.default_rng(seed).integers(1, 50, (B, T)).astype(np.int32)


class TestScan(unittest.TestCase):
    def test_values_and_gradients_match_the_loop(self):
        """Lengths around the chunk size (sqrt(T)) and not multiples of it, a carried initial state; float64."""
        g = torch.Generator().manual_seed(0)
        for T in (1, 2, 7, 50, 64, 65, 300, 1021, 1024, 4099):  # 1021, 4099: primes (prefix + rest)
            keep = torch.rand(2, T, 5, generator=g, dtype=torch.float64).requires_grad_()
            write = torch.randn(2, T, 5, generator=g, dtype=torch.float64).requires_grad_()
            initial = torch.randn(2, 5, generator=g, dtype=torch.float64).requires_grad_()
            d = torch.randn(2, T, 5, generator=g, dtype=torch.float64)
            a = linear_scan(keep, write, initial)
            b = step_by_step(keep, write, initial)
            self.assertLess(float((a - b).abs().max()), 1e-12, T)
            for x, y in zip(torch.autograd.grad(a, (keep, write, initial), d),
                            torch.autograd.grad(b, (keep, write, initial), d)):
                self.assertLess(float((x - y).abs().max()), 1e-12, T)

    def test_head_keep_factors_match_expanded_ones(self):
        """A keep factor per head (gate_heads) equals the same factor repeated over the head's channels: values and
        gradients (the head's gradient is the sum over its channels), around the chunk sizes; float64."""
        g = torch.Generator().manual_seed(1)
        for T in (1, 7, 64, 65, 300):
            keep = torch.rand(2, T, 3, generator=g, dtype=torch.float64).requires_grad_()
            write = torch.randn(2, T, 12, generator=g, dtype=torch.float64).requires_grad_()
            initial = torch.randn(2, 12, generator=g, dtype=torch.float64).requires_grad_()
            d = torch.randn(2, T, 12, generator=g, dtype=torch.float64)
            a = linear_scan(keep, write, initial)
            b = step_by_step(keep.repeat_interleave(4, -1), write, initial)
            self.assertLess(float((a - b).abs().max()), 1e-12, T)
            for x, y in zip(torch.autograd.grad(a, (keep, write, initial), d),
                            torch.autograd.grad(b, (keep, write, initial), d)):
                self.assertLess(float((x - y).abs().max()), 1e-12, T)

    def test_long_rows_with_tiny_keep_factors_stay_finite(self):
        """Products of many keep factors underflow to 0 (no log space, nothing to overflow)."""
        keep = torch.full((1, 5000, 3), 1e-3)
        write = torch.ones(1, 5000, 3)
        states = linear_scan(keep, write)
        self.assertTrue(torch.isfinite(states).all())
        self.assertAlmostEqual(float(states[0, -1, 0]), 1.0 / (1 - 1e-3), places=5)


SYSTEM, TOOLS, ASSISTANT = (12, 13), (2, 3), (6, 7)


def agent_doc(rng, memory=True):
    """<bos>[<system>..</system>][<tools>..</tools>]<user>..</user><assistant>..</assistant> (tag ids as the agent
    datasets: system 12/13, tools 2/3, user 4/5, assistant 6/7; other tokens 16..49)."""
    words = lambda lo, hi: list(rng.integers(16, 50, int(rng.integers(lo, hi))))
    doc = [0]
    if memory and rng.random() < 0.8:
        doc += [12] + words(1, 8) + [13]
    if memory and rng.random() < 0.8:
        doc += [2] + words(1, 12) + [3]
    for _ in range(int(rng.integers(1, 3))):
        doc += [4] + words(1, 5) + [5, 6] + words(1, 6) + [7]
    return doc


def agent_rows(B=2, T=70, seed=3):
    """Rows cut from a stream of agent documents (the first one starts at a document start)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):
        stream = []
        while len(stream) < T:
            stream += agent_doc(rng)
        rows.append(stream[:T])
    return np.asarray(rows, np.int32)


def reference(m, tokens):
    """The static-memory model on one raw row, a token at a time with plain loops: (stream at each position the
    layers see, those positions)."""
    w = lambda v: v.value.detach().double().cpu()
    E = w(m.token_embedding.embeddings)
    rms = lambda x: x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
    H, h, d, n = m.hidden_size, m.memory_heads, m.memory_dim, 2
    # parse: which memory each token belongs to (0: none), tags, reading positions, documents
    kind, tag, read, doc = [], [], [], []
    inside, reading, number = 0, False, 0
    for t in tokens:
        if t == 0:
            number, inside, reading = number + 1, 0, False
        is_tag = t in (12, 13, 2, 3)
        if t in (13, 3):
            inside = 0
        kind.append(inside if not is_tag else 0)
        tag.append(is_tag)
        if t == 12:
            inside = 1
        elif t == 2:
            inside = 2
        if t == 6:  # the opening tag and the tokens inside read (they predict the assistant's tokens)
            reading = True
        if t == 7:
            reading = False
        read.append(reading)
        doc.append(number)
    # each document's memories: the encoder over its tags and contents (a marker first), per layer
    enc_p = {k: w(v) for k, v in m._mem_enc.items()}
    memories = {}
    for g in sorted(set(doc)):
        seq = [(0, 0)] + [(t, k) for t, k, a, dg in zip(tokens, kind, tag, doc) if dg == g and (k or a)]
        state = torch.zeros(H, dtype=torch.float64)
        entries = []
        for t, k in seq:
            x = rms(E[t])
            if k:
                entries.append((rms(state), x, k))
            gl, raw = ((x * enc_p["norm_in"]) @ enc_p["in_proj"] + enc_p["in_bias"]).split([H, H])
            u = torch.sigmoid(gl)
            state = (1 - u) * state * (t != 0) + u * torch.tanh(raw)
        memories[g] = entries
    # the layers over the kept tokens
    kept = [i for i, k in enumerate(kind) if k == 0]
    stream = [E[tokens[i]].clone() for i in kept]
    for layer in m._params:
        p = {k: w(v) for k, v in layer.items()}
        state = torch.zeros(H, dtype=torch.float64)
        for j, i in enumerate(kept):
            x = rms(stream[j]) * p["norm_in"]
            gl, raw = (x @ p["in_proj"] + p["in_bias"]).split([H, H])
            u = torch.sigmoid(gl)
            state = (1 - u) * state * (tokens[i] != 0) + u * torch.tanh(raw)
            sn = rms(state)
            add = torch.zeros(m.dim, dtype=torch.float64)
            if read[i] and memories[doc[i]]:
                q = (torch.nn.functional.elu(sn @ p["mem_q"]) + 1).reshape(h, d)
                gate = torch.sigmoid(sn @ p["mem_gate"] + p["mem_gate_bias"]).reshape(n, h)
                reads = torch.zeros(n, h, d, dtype=torch.float64)
                for mem in (1, 2):
                    num, den = torch.zeros(h, d, dtype=torch.float64), torch.zeros(h, dtype=torch.float64)
                    for before, value, k in memories[doc[i]]:
                        if k == mem:
                            key = (torch.nn.functional.elu(before @ p["mem_k"]) + 1).reshape(h, d)
                            s = (q * key).sum(-1)
                            num += s[:, None] * (value @ p["mem_v"]).reshape(h, d)
                            den += s
                    reads[mem - 1] = num / den.clamp_min(1e-6)[:, None]
                add = (reads * gate[..., None]).reshape(-1) @ p["mem_out"]
            if m.mlp_mult is None:
                write = torch.tanh((sn * p["norm_state"]) @ p["out_proj"])
            else:
                hid = torch.nn.functional.gelu((sn * p["norm_state"]) @ p["mlp_in"] + p["mlp_bias"], approximate="tanh")
                write = hid @ p["mlp_out"]
            stream[j] = stream[j] + add + p["write_scale"] * write
    return torch.stack(stream), kept


def memory_model(**kw):
    m = tiny(static_memory=True, memory_heads=2, memory_dim=4, **kw)
    g = torch.Generator().manual_seed(5)
    for p in m._params:  # make the memory matter
        p["mem_out"].value.data.copy_(torch.randn(p["mem_out"].shape, generator=g) * 0.5)
        p["mem_gate_bias"].value.data.fill_(1.0)
    return m


class TestStaticMemory(unittest.TestCase):
    def test_matches_the_loop_reference(self):
        m = memory_model()
        x = agent_rows()
        got = m.dense_hidden(x).double().cpu()  # rows without the span contents
        for b in range(x.shape[0]):
            want, kept = reference(m, list(x[b]))
            self.assertLess(float((got[b, :len(kept)] - want).abs().max()), 1e-5)

    def test_memory_gradients_match_plain_autograd(self):
        """RepeatSorted's cumsum backward against autograd's gather backward, on rows with documents that miss one of
        the memories (z = 0 there: dividing by a 1e-6 clamp gave 1e7-sized gradient rows that wrecked the cumsum)."""
        import model.archs.min_gru as mg

        m, x = memory_model(), agent_rows(B=3, T=128)
        grads = []
        for plain in (False, True):
            for w in m.trainable_weights:
                w.value.grad = None
            saved = mg.RepeatSorted
            if plain:
                mg.RepeatSorted = type("Plain", (), {"apply": staticmethod(lambda C, idx: C.index_select(0, idx))})
            try:
                m.dense_hidden(x).square().sum().backward()
            finally:
                mg.RepeatSorted = saved
            grads.append({w.path: w.value.grad for w in m.trainable_weights})
        for k, a in grads[0].items():
            if a is not None:
                self.assertLess(float((a - grads[1][k]).abs().max() / (a.abs().max() + 1e-12)), 1e-4, k)

    def test_mlp_matches_the_loop_reference(self):
        """`mlp_mult`: the GELU MLP write instead of the tanh projection; the full row, the valid positions only and
        generation token by token agree with the loop."""
        m = memory_model(mlp_mult=4)
        self.assertNotIn("out_proj", m._params[0])
        x = agent_rows(B=1, T=90)
        got = m.dense_hidden(x).double().cpu()
        want, kept = reference(m, list(x[0]))
        self.assertLess(float((got[0, :len(kept)] - want).abs().max()), 1e-5)
        with torch.no_grad():
            state, steps = None, []
            for t in range(x.shape[1]):
                out, state = m.advance(x[:, t:t + 1], state)
                steps.append(out[0])
        full = m.readout(m.readout_input(m.dense_hidden(x)))[0].detach()
        from model.memory_rows import span_masks
        last = np.cumsum(span_masks(x[0], [SYSTEM, TOOLS], ASSISTANT, 0)[0] == 0) - 1
        for t, o in enumerate(steps):
            self.assertLess(float((o - full[last[t]]).abs().max()), 1e-4, t)

    def test_starts_as_the_model_without_memory(self):
        """mem_out 0: the layers on the rows without span contents."""
        m = tiny(static_memory=True, memory_heads=2, memory_dim=4)
        x = agent_rows()
        plain = tiny()
        for v in plain.weights:
            v.assign(next(u for u in m.weights if u.path.split("/")[-1] == v.path.split("/")[-1]))
        from model.memory_rows import span_masks
        for b in range(x.shape[0]):
            kind = span_masks(x[b], [SYSTEM, TOOLS], ASSISTANT, 0)[0]
            kept = x[b][kind == 0][None]
            self.assertLess(float((m.dense_hidden(x)[b, :kept.shape[1]] - plain.dense_hidden(kept)[0]).abs().max()), 1e-6)

    def test_generation_matches_the_full_row(self):
        """Token by token and in random chunks (spans opened in one call close in a later one)."""
        m = memory_model()
        x = agent_rows(B=1, T=90)
        full = m.readout(m.readout_input(m.dense_hidden(x)))[0]
        from model.memory_rows import span_masks
        kept = np.cumsum(span_masks(x[0], [SYSTEM, TOOLS], ASSISTANT, 0)[0] == 0) - 1  # last kept position so far
        rng = np.random.default_rng(0)
        for sizes in ([1] * 90, list(rng.integers(1, 12, 30))):
            state, t = None, 0
            with torch.no_grad():
                for size in sizes:
                    if t >= 90:
                        break
                    out, state = m.advance(x[:, t:t + size], state)
                    t = min(90, t + size)
                    self.assertLess(float((out[0] - full[kept[t - 1]]).abs().max()), 1e-5, t)

    def test_inspection_rebuilds_the_memory_reads(self):
        """inspect_memory: the read weights over the entries (q . k_j, per memory and head, normalized), the entries'
        values from their tokens and the gates rebuild every reading position's memory_add."""
        m = memory_model()
        x = agent_rows(B=1, T=90)
        infos, meta = m.inspect_memory(x)
        rms = lambda v: v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-6)
        E = m.token_embedding.embeddings.value.detach().cpu().numpy()
        h, d, nm = m.memory_heads, m.memory_dim, 2
        tokens = x[0][meta["entry_pos"]]
        self.assertTrue(len(tokens) > 0 and (meta["kept"].size + tokens.size) <= x.shape[1])
        for n, (info, p) in enumerate(zip(infos, m._params)):
            values = (rms(E[tokens]) @ p["mem_v"].value.detach().cpu().numpy()).reshape(-1, h, d)
            keys = meta["entry_keys"][:, n]  # (entries, h, d)
            for t, g in zip(meta["read_pos"], meta["read_doc"]):
                q = info["memory_query"][0, t].reshape(h, d)
                gate = info["memory_gate"][0, t].reshape(nm, h)
                read = np.zeros((nm, h, d))
                for i in range(nm):
                    mine = (meta["entry_doc"] == g) & (meta["entry_kind"] == i)
                    if mine.any():
                        w = np.einsum("hd,jhd->jh", q, keys[mine])
                        read[i] = np.einsum("jh,jhd->hd", w / w.sum(0), values[mine])
                add = (read * gate[..., None]).reshape(-1) @ p["mem_out"].value.detach().cpu().numpy()
                self.assertLess(float(np.abs(add - info["memory_add"][0, t]).max()), 1e-4, (n, t))

    def test_gradients_reach_the_memory(self):
        m = memory_model()
        x = agent_rows()
        m.dense_hidden(x).square().sum().backward()
        for name in ("mem_q", "mem_k", "mem_v", "mem_gate", "mem_gate_bias", "mem_out"):
            self.assertGreater(float(m._params[1][name].value.grad.abs().sum()), 0, name)
        for name, v in m._mem_enc.items():
            self.assertGreater(float(v.value.grad.abs().sum()), 0, name)

    def test_loader_rows_match_the_raw_rows(self):
        """Rows from document starts (validation): the loader's batch gives the raw rows' outputs; random windows:
        every row is the corpus without span contents, and each read document brings its whole memory."""
        import tempfile
        from pathlib import Path

        from model.data import PackedDocumentDataset
        from model.memory_rows import span_masks

        rng = np.random.default_rng(1)
        corpus = np.asarray(sum((agent_doc(rng) for _ in range(300)), []), np.int32)
        spec = {"spans": [SYSTEM, TOOLS], "read": ASSISTANT}
        m = memory_model()
        kind = span_masks(corpus, [SYSTEM, TOOLS], ASSISTANT, 0)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "train.npy")
            np.save(path, corpus)
            val = PackedDocumentDataset(path, 40, 4, 50, 0, steps_per_epoch=3, fixed=True, use_cache=False,
                                        static_memory=spec)
            batch, _ = val[0]
            got = m.dense_hidden(batch)
            k0 = np.flatnonzero(kind == 0)
            stream = corpus[k0]  # the corpus without span contents
            for b in range(4):  # the raw row: from its document start to its last kept token
                s = next(int(s) for s in np.searchsorted(k0, np.flatnonzero(corpus == 0))
                         if np.array_equal(stream[s:s + 40], batch["ids"][b]))
                raw = corpus[k0[s]:k0[s + 39] + 1][None]
                self.assertLess(float((got[b] - m.dense_hidden(raw)[0]).abs().max()), 1e-5, b)
            doc = np.cumsum(corpus == 0)
            contents = {tuple(corpus[(doc == g) & (kind > 0)]) for g in np.unique(doc)}
            train = PackedDocumentDataset(path, 40, 4, 50, 0, random_windows=True, use_cache=False, static_memory=spec)
            self.assertEqual(train.steps_per_pass, ((kind == 0).sum() - 40) // 40 // 4)
            for i in range(3):
                batch, y = train[i]
                rows = batch["ids"]
                stream = corpus[kind == 0]
                for b in range(4):  # a window of the stream without span contents
                    self.assertTrue(any(np.array_equal(stream[s:s + 40], rows[b]) for s in range(len(stream) - 40)), b)
                flat = batch["enc"].reshape(-1)
                doc_of = batch["mem_seg"][batch["mem_slot"] // 32] // 2  # each entry's document
                for g in range(batch["mem_count"].shape[0]):  # each memory: a whole document's span contents
                    self.assertIn(tuple(flat[batch["mem_pos"][doc_of == g]]), contents)
                m.dense_hidden_valid(batch, torch.as_tensor(y >= 0)).sum()


class TestMinGRU(unittest.TestCase):
    def test_state_restarts_at_the_document_marker(self):
        """After a marker, the stream equals that of a row that starts at the marker."""
        m = tiny()
        x = ids()
        x[:, 15] = 0
        full = m.dense_hidden(x)
        fresh = m.dense_hidden(x[:, 15:])
        self.assertLess(float((full[:, 15:] - fresh).abs().max()), 1e-5)

    def test_valid_positions_only(self):
        m = tiny()
        x = ids()
        valid = torch.as_tensor(np.random.default_rng(1).random((2, 40)) < 0.4)
        self.assertLess(float((m.dense_hidden_valid(x, valid) - m.dense_hidden(x)[valid]).abs().max()), 1e-6)

    def test_carried_state_continues_the_row(self):
        """Two halves with `carry=True` (truncated BPTT) give the same stream as the whole row."""
        m = tiny()
        x = ids()
        valid = torch.ones(2, 20, dtype=torch.bool)
        m.__dict__.pop("_train_carry", None)
        first = m.dense_hidden_valid(x[:, :20], valid, carry=True)
        second = m.dense_hidden_valid(x[:, 20:], valid, carry=True)
        both = torch.cat([first.reshape(2, 20, -1), second.reshape(2, 20, -1)], 1)
        self.assertLess(float((both - m.dense_hidden(x)).abs().max()), 1e-5)

    def test_generation_state_matches_the_full_row(self):
        m = tiny()
        x = ids()
        state, logits = None, []
        for t in range(40):
            out, state = m.advance(x[:, t:t + 1], state)
            logits.append(out)
        full = m.readout(m.readout_input(m.dense_hidden(x)))
        self.assertLess(float((torch.stack(logits, 1) - full).abs().max()), 1e-5)

    def test_recompute_gives_the_same_gradients(self):
        x = ids()
        m = tiny()  # one model, toggled (keras draws each new model's weights from its own RNG)
        grads = []
        for recompute in (False, True):
            m.recompute = recompute
            for v in m.trainable_weights:
                v.value.grad = None
            m.dense_hidden(x).square().sum().backward()
            grads.append([v.value.grad.clone() for v in m.trainable_weights if v.value.grad is not None])
        for a, b in zip(*grads):
            self.assertLess(float((a - b).abs().max()), 1e-5)

    def test_gate_heads_model(self):
        """gate_heads: valid positions, carried state and generation agree with the full row, as per channel."""
        m = tiny(gate_heads=3)
        x = ids()
        x[:, 15] = 0
        full = m.dense_hidden(x)
        self.assertLess(float((full[:, 15:] - m.dense_hidden(x[:, 15:])).abs().max()), 1e-5)
        state, logits = None, []
        for t in range(40):
            out, state = m.advance(x[:, t:t + 1], state)
            logits.append(out)
        self.assertLess(float((torch.stack(logits, 1) - m.readout(m.readout_input(full))).abs().max()), 1e-5)
        self.assertEqual(tuple(m._params[0]["in_proj"].value.shape), (16, 3 + 12))
        self.assertEqual(m.inspect(x)[0]["update_gate"].shape, (2, 40, 12))

    def test_inspect_shapes_and_ranges(self):
        m = tiny()
        info = m.inspect(ids())
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0]["state"].shape, (2, 40, 12))
        self.assertEqual(info[0]["layer_write"].shape, (2, 40, 16))
        self.assertTrue((np.abs(info[0]["state"]) <= 1).all())  # tanh candidates: the state stays in -1..1
        self.assertTrue(((info[0]["update_gate"] >= 0) & (info[0]["update_gate"] <= 1)).all())
        hl = m.timescales()[0]
        self.assertAlmostEqual(float(hl.min()), 1.0, places=3)
        self.assertAlmostEqual(float(hl.max()), 1024.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
