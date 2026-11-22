"""Training, checkpointing and resume.

The property that matters most here is that resume is *exact*: reloading
`model.keras` must restore Adam's moment estimates, not just the weights.
When it does not, training visibly spikes for a few hundred steps after every
restart, which is easy to miss and hard to attribute.
"""

import unittest

from tests.helpers import temp_workspace, tiny_config

import numpy as np
from keras import ops

from model import config as cfg_mod
from model.losses import SparseCategoricalCrossentropyFromLogits
from model.train import build_or_load, last_completed_epoch, resolve_optimizer, train


class TestOptimizerResolution(unittest.TestCase):
    def test_supported_optimizers(self):
        for name, expected in (("adam", "Adam"), ("adamw", "AdamW"), ("sgd", "SGD")):
            with self.subTest(name=name):
                cfg = tiny_config()
                cfg["training"]["optimizer"] = name
                self.assertEqual(type(resolve_optimizer(cfg)).__name__, expected)

    def test_unknown_optimizer_raises(self):
        cfg = tiny_config()
        cfg["training"]["optimizer"] = "lbfgs"
        with self.assertRaises(ValueError):
            resolve_optimizer(cfg)

    def test_learning_rate_comes_from_config(self):
        cfg = tiny_config()
        cfg["training"]["learning_rate"] = 0.0007
        opt = resolve_optimizer(cfg)
        self.assertAlmostEqual(float(ops.convert_to_numpy(opt.learning_rate)), 0.0007, places=6)


class TestLoss(unittest.TestCase):
    def test_matches_analytic_cross_entropy(self):
        """The XPU-safe loss must equal -log(softmax)[label], exactly."""
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(6, 17)).astype("float32")
        labels = rng.integers(0, 17, size=(6,)).astype("int32")

        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        expected = -log_probs[np.arange(6), labels]

        actual = ops.convert_to_numpy(
            SparseCategoricalCrossentropyFromLogits(reduction=None)(labels, logits)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_accepts_column_shaped_labels(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(4, 9)).astype("float32")
        labels = rng.integers(0, 9, size=(4,)).astype("int32")
        loss = SparseCategoricalCrossentropyFromLogits()
        flat = float(ops.convert_to_numpy(loss(labels, logits)))
        column = float(ops.convert_to_numpy(loss(labels.reshape(-1, 1), logits)))
        self.assertAlmostEqual(flat, column, places=6)


class TestResumeBookkeeping(unittest.TestCase):
    def test_last_completed_epoch_on_missing_file(self):
        with temp_workspace() as tmp:
            self.assertEqual(last_completed_epoch(tmp / "nope.csv"), 0)

    def test_last_completed_epoch_reads_the_final_row(self):
        with temp_workspace() as tmp:
            path = tmp / "metrics.csv"
            path.write_text("epoch,loss\n0,9.0\n1,8.0\n2,7.0\n")
            # Keras writes zero-based epochs, so a final row of 2 means 3 done.
            self.assertEqual(last_completed_epoch(path), 3)

    def test_corrupt_metrics_file_does_not_crash(self):
        with temp_workspace() as tmp:
            path = tmp / "metrics.csv"
            path.write_text("not,a,valid\nheader\n")
            self.assertEqual(last_completed_epoch(path), 0)


class TestTrainingLoop(unittest.TestCase):
    def _run(self, run_name="r", epochs=2, **cfg_overrides):
        cfg = tiny_config(run_name)
        for section, values in cfg_overrides.items():
            cfg[section].update(values)
        cfg_mod.save(run_name, cfg)
        return train(run_name, epochs=epochs, verbose=0), cfg

    def test_trains_and_writes_all_artifacts(self):
        with temp_workspace():
            self._run("r", epochs=2)
            run_path = cfg_mod.run_dir("r")
            self.assertTrue((run_path / "model.keras").exists())
            self.assertTrue((run_path / "metrics.csv").exists())
            self.assertTrue((run_path / "checkpoints" / "checkpoint_00001.keras").exists())
            self.assertTrue((run_path / "checkpoints" / "checkpoint_00002.keras").exists())
            self.assertTrue((run_path / "checkpoints" / "best.keras").exists())
            self.assertEqual(last_completed_epoch(run_path / "metrics.csv"), 2)

    def test_resume_continues_from_the_recorded_epoch(self):
        with temp_workspace():
            self._run("r", epochs=2)
            history = train("r", epochs=4, verbose=0)
            self.assertEqual(history.epoch, [2, 3])
            self.assertEqual(last_completed_epoch(cfg_mod.run_dir("r") / "metrics.csv"), 4)

    def test_resume_restores_optimizer_moments(self):
        """Adam's m/v must survive the `.keras` round-trip, not reset to zero."""
        with temp_workspace():
            self._run("r", epochs=2)
            cfg = cfg_mod.load("r")
            model, initial_epoch = build_or_load("r", cfg, verbose=0)

            self.assertEqual(initial_epoch, 2)
            steps = cfg["data"]["steps_per_epoch"] * 2
            self.assertEqual(int(ops.convert_to_numpy(model.optimizer.iterations)), steps)

            moments = [v for v in model.optimizer.variables if "momentum" in v.name or "velocity" in v.name]
            self.assertGreater(len(moments), 0)
            magnitudes = [float(np.abs(ops.convert_to_numpy(v)).max()) for v in moments]
            self.assertTrue(all(m > 0 for m in magnitudes), "optimizer moments were reset to zero")

    def test_resume_preserves_weights_exactly(self):
        with temp_workspace():
            self._run("r", epochs=1)
            cfg = cfg_mod.load("r")
            first, _ = build_or_load("r", cfg, verbose=0)
            x = np.random.randint(0, first.vocabulary_size, (3, first.context_length)).astype("int32")
            before = ops.convert_to_numpy(first(x))
            second, _ = build_or_load("r", cfg, verbose=0)
            np.testing.assert_array_equal(before, ops.convert_to_numpy(second(x)))

    def test_completed_run_is_a_no_op(self):
        with temp_workspace():
            self._run("r", epochs=2)
            history = train("r", epochs=2, verbose=0)
            self.assertEqual(getattr(history, "epoch", []), [])

    def test_fresh_run_starts_at_epoch_zero(self):
        with temp_workspace():
            cfg = tiny_config("r")
            cfg_mod.save("r", cfg)
            _, initial_epoch = build_or_load("r", cfg, verbose=0)
            self.assertEqual(initial_epoch, 0)


if __name__ == "__main__":
    unittest.main()
