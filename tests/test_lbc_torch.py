"""model/lbc_torch.py (lbc checkpoints in torch) on a tiny random checkpoint written in the engine's format: the
recording pass matches the decode path, a <bos> restarts the document, a session continues like a full feed, and the
banks follow their definition (running averages of decayed writes) against a float64 replay of the recorded writes."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import lbc_torch as lt

V, D, L, N, H, DH = 64, 32, 2, 8, 2, 32
F = 4 * D
HD = H * DH


def tensors():
    """(name, rows, cols) in Model::Model's order for the tiny config (hops [2, 1], second FFN on layer 0)."""
    ldPV, ldG, ldG2 = 2 * N + HD, (2 * HD + 2 * H + 7) // 8 * 8, (HD + H + 7) // 8 * 8
    out = [("E", V, D), ("final_norm", 1, D)]
    for n, hops in enumerate([2, 1]):
        p = f"l{n}_"
        out += [(p + "norm1", 1, D), (p + "Wa", D, ldPV), (p + "nu", 1, N), (p + "theta", 1, N), (p + "Ws", 2 * N, ldG),
                (p + "bk", 1, HD), (p + "brg", 1, H), (p + "bwg", 1, H), (p + "R", H, DH * 128), (p + "log_tau", 1, H)]
        if hops == 2:
            out.append((p + "log_tau2", 1, H))
        out.append((p + "Wo", HD, D))
        if hops == 2:
            out.append((p + "Wo2", HD, D))
        out += [(p + "ffn_norm", 1, D), (p + "W1", D, F), (p + "b1", 1, F), (p + "W2", F, D), (p + "b2", 1, D)]
        if hops == 2:
            out += [(p + "q2_norm", 1, D), (p + "Wq2", D, ldG2), (p + "brg2", 1, H)]
            out += [(p + "ffn2_norm", 1, D), (p + "W1b", D, F), (p + "b1b", 1, F), (p + "W2b", F, D), (p + "b2b", 1, D)]
    return out


def write_checkpoint(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    manifest, chunks, off = [], [], 0
    for name, r, c in tensors():
        if "norm" in name:
            x = 1 + 0.1 * rng.standard_normal((r, c))
        elif name.endswith("_nu"):
            x = np.log(rng.uniform(0.05, 1.0, (r, c)))
        elif name.endswith(("brg", "brg2")):
            x = np.full((r, c), 0.5)  # reads mostly open
        elif name.endswith("bwg"):
            x = np.zeros((r, c))  # writes about half open
        else:
            x = rng.standard_normal((r, c)) / np.sqrt(r)
        manifest.append([name, off, r, c])
        chunks.append(x.astype(np.float32).ravel())
        off += r * c
    params = np.concatenate(chunks)
    arch = {"vocabulary_size": V, "dim": D, "embed_dim": D, "factorized": False, "tied_embeddings": True,
            "num_layers": L, "hops": [2, 1], "lru_dim": N, "num_heads": H, "head_dim": DH, "ffn_mult": 4,
            "slot_half_life": [1, 64], "slot_half_life_max": [64, 16], "hop2_ffn": True}
    head = json.dumps({"format": 1, "epoch": 1, "step": 0, "num_params": off, "tensors": manifest,
                       "config": {"arch_params": arch}}).encode()
    data_off = (16 + len(head) + lt.ALIGN - 1) // lt.ALIGN * lt.ALIGN
    with open(path, "wb") as f:
        f.write(lt.MAGIC + struct.pack("<Q", len(head)) + head + b"\0" * (data_off - 16 - len(head)))
        f.write(np.concatenate([params, np.zeros(2 * off, np.float32)]).tobytes())  # + the Adam moments


class TestLbcTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        path = Path(cls.dir.name) / "tiny.lbc"
        write_checkpoint(path)
        cls.m = lt.LbcModel(path, "cpu")
        cls.ids = [0] + list(np.random.default_rng(1).integers(3, V, 40))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def feed_all(self, ids):
        dec = lt.Decoder(self.m, graph=False)
        return np.stack([dec.feed(i).numpy().copy() for i in ids])

    def test_trace_matches_decode(self):
        r = lt.trace(self.m, self.ids, max_tokens=64)
        np.testing.assert_allclose(r["logits"], self.feed_all(self.ids), atol=1e-4)
        self.assertGreater(r["g"][:, :, 0].mean(), 0.1)  # some writes open, some closed
        self.assertLess(r["g"][:, :, 0].mean(), 0.9)
        self.assertEqual(float(np.abs(r["p"][:, 1, 1]).max()), 0.0)  # layer 1 has no second read

    def test_bos_restarts_the_document(self):
        doc2 = [0, 5, 9, 11, 7]
        both = self.feed_all(self.ids + doc2)
        np.testing.assert_allclose(both[len(self.ids):], self.feed_all(doc2), atol=1e-5)

    def test_session_continues_like_a_full_feed(self):
        from model.generation import Session

        s = Session()
        first = list(lt.stream(self.m, self.ids[:20], max_new_tokens=4, session=s))
        cont = self.ids[:20] + first + [7, 8]
        a = list(lt.stream(self.m, cont, max_new_tokens=6, session=s))
        self.assertEqual(a, list(lt.stream(self.m, cont, max_new_tokens=6)))

    def test_banks_are_decayed_averages_of_the_writes(self):
        """The read rebuilt in float64 from the recorded writes: slot m of head h at t holds
        sum_j w_jm a_m^(t-j) v_j / sum_j w_jm a_m^(t-j), and the read is sum_m p_tm of those."""
        r = lt.trace(self.m, self.ids, max_tokens=64)
        a = np.exp(self.m.loga)  # (L, H, M)
        for li in range(L):
            for t in (10, len(self.ids) - 1):
                age = t - np.arange(t + 1)
                W = r["w"][:t + 1, li].astype(np.float64) * a[li][None] ** age[:, None, None]  # (t+1, H, M)
                z = W.sum(0)
                avg = np.einsum("jhm,jhd->hmd", W, r["v"][:t + 1, li]) / np.maximum(z, 1e-30)[..., None]
                o = np.einsum("hm,hmd->hd", r["p"][t, li, 0] * (z > lt.BANK_EPS), avg)
                np.testing.assert_allclose(o, r["o"][t, li, 0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
