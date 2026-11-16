"""Properties specific to `AlibiRetrieverSphereLM`."""

import unittest

from tests.helpers import VOCAB, tiny_arch_params

import numpy as np
from keras import ops

from model.archs.alibi_retriever_sphere import AlibiRetrieverSphereLM


def _model(**overrides) -> AlibiRetrieverSphereLM:
    params = tiny_arch_params("alibi-retriever-sphere")
    params.update(overrides)
    return AlibiRetrieverSphereLM(**params)


class TestAlibiRetrieverSphericalInvariants(unittest.TestCase):
    def test_both_tables_are_exactly_unit_norm_after_projection(self):
        m = _model()
        for table in (m.normalized_input_embeddings(), m.normalized_target_embeddings()):
            norms = np.linalg.norm(ops.convert_to_numpy(table), axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_logits_are_bounded_by_tau(self):
        tau = 4.0
        m = _model(initial_tau=tau)
        x = np.random.randint(0, VOCAB, (8, m.context_length)).astype("int32")
        logits = ops.convert_to_numpy(m(x))
        self.assertLessEqual(float(np.abs(logits).max()), tau + 1e-4)

    def test_query_is_on_the_unit_sphere(self):
        m = _model(initial_tau=1.0)
        x = np.random.randint(0, VOCAB, (8, m.context_length)).astype("int32")
        logits = ops.convert_to_numpy(m(x))
        self.assertLessEqual(float(logits.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(logits.min()), -1.0 - 1e-5)


class TestAlibiRetrieverParameterization(unittest.TestCase):
    def test_no_bias_parameters_anywhere(self):
        m = _model()
        for var in m.weights:
            self.assertNotIn("bias", var.name.lower(), f"unexpected bias parameter: {var.name}")

    def test_parameter_count_matches_the_analytic_formula(self):
        v, di, do, w, h, nh = 200, 8, 16, 32, 64, 2
        hd = di // nh
        qk = do // di
        m = AlibiRetrieverSphereLM(
            vocabulary_size=v,
            dim_in=di,
            dim_out=do,
            window_size=w,
            hidden_dim=h,
            num_heads=nh,
        )
        # Embeddings:
        expected_emb = v * di + v * do
        # Taus: tau (1) + retriever_tau (1)
        taus = 2
        # Block 1 (MHA + MLP):
        mha = 3 * (di * (nh * hd)) + (nh * hd) * di
        mlp = di * (4 * di) + (4 * di) * di
        block1 = mha + mlp
        # Block 2 (MHA + MLP):
        block2 = mha + mlp
        # Sequence projection:
        seq_proj = (w + qk) * w
        # Readout FFN:
        ffn = (w * di) * h + h * (h // 2) + (h // 2) * do

        expected_total = expected_emb + taus + block1 + block2 + seq_proj + ffn

        self.assertEqual(m.parameter_breakdown()["total"], expected_total)
        self.assertEqual(m.parameter_breakdown()["embeddings"], expected_emb)
        self.assertEqual(m.parameter_breakdown()["mechanism"], expected_total - expected_emb)

    def test_learnable_tau_toggles_trainability(self):
        for learnable in (True, False):
            with self.subTest(learnable=learnable):
                m = _model(learnable_tau=learnable)
                names = [v.name for v in m.trainable_variables]
                self.assertEqual("tau" in names, learnable)
                self.assertEqual("retriever_tau" in names, learnable)

    def test_context_length_equals_window_size(self):
        m = _model(window_size=11)
        self.assertEqual(m.context_length, 11)

    def test_activation_selection(self):
        for act in ("gelu", "tanh"):
            m = _model(activation=act)
            self.assertEqual(m.activation, act)
            self.assertEqual(m.get_config()["activation"], act)
        with self.assertRaises(ValueError):
            _model(activation="relu")

    def test_invalid_arguments_are_rejected(self):
        for bad in (
            {"vocabulary_size": 0},
            {"dim_in": 0},
            {"dim_out": -1},
            {"window_size": 0},
            {"dim_out": 13, "dim_in": 4},
            {"dim_out": 64, "dim_in": 4, "window_size": 8},
            {"hidden_dim": 1},
            {"num_heads": 0},
            {"dim_in": 7, "num_heads": 2},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _model(**bad)


class TestAlibiRetrieverPaddingBehaviour(unittest.TestCase):
    def test_all_padding_window_produces_finite_logits_and_gradients(self):
        import torch

        m = _model()
        x = ops.convert_to_tensor(np.zeros((2, m.context_length), dtype="int32"))
        logits = m(x, training=True)
        self.assertTrue(bool(torch.isfinite(logits).all()))

        loss = ops.mean(logits**2)
        grads = torch.autograd.grad(loss, [v.value for v in m.trainable_variables], allow_unused=True)
        for var, grad in zip(m.trainable_variables, grads):
            if grad is not None:
                self.assertTrue(bool(torch.isfinite(grad).all()), f"non-finite grad for {var.name}")

    def test_padding_changes_the_prediction(self):
        m = _model()
        rng = np.random.default_rng(0)
        full = rng.integers(1, VOCAB, (1, m.context_length)).astype("int32")
        padded = full.copy()
        padded[0, : m.context_length // 2] = 0
        self.assertFalse(
            np.allclose(ops.convert_to_numpy(m(full)), ops.convert_to_numpy(m(padded))),
            "left padding must affect the output",
        )


if __name__ == "__main__":
    unittest.main()
