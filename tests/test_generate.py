"""Generation and evaluation.

Generation is implemented once, against the `BaseLM` interface, so these
tests double as a check that the interface exposes enough for inference
without any architecture-specific knowledge.
"""

import unittest
from unittest import mock

from tests.helpers import VOCAB, temp_workspace, tiny_arch_params, tiny_config

import numpy as np

from model import config as cfg_mod
from model.archs.dual_sphere import DualSphereLM
from model.evaluate import evaluate, resolve_checkpoint
from model.generation import generate
from model.train import train


def _model(**overrides) -> DualSphereLM:
    params = tiny_arch_params("dual-sphere")
    params.update(overrides)
    return DualSphereLM(**params)


class TestGenerate(unittest.TestCase):
    def test_produces_exactly_max_new_tokens(self):
        m = _model()
        for n in (1, 5, 12):
            with self.subTest(n=n):
                out = generate(m, [1, 2, 3], max_new_tokens=n)
                self.assertEqual(len(out.split()), 3 + n)

    def test_greedy_is_deterministic(self):
        m = _model()
        first = generate(m, [4, 5, 6], max_new_tokens=8, temperature=0.0)
        second = generate(m, [4, 5, 6], max_new_tokens=8, temperature=0.0)
        self.assertEqual(first, second)

    def test_greedy_picks_the_argmax(self):
        m = _model()
        prompt = [7, 8]
        out = [int(t) for t in generate(m, prompt, max_new_tokens=1).split()]

        from keras import ops

        x = np.zeros((1, m.context_length), dtype=np.int32)
        x[0, -len(prompt) :] = prompt
        expected = int(np.argmax(ops.convert_to_numpy(m(x))[0]))
        self.assertEqual(out[-1], expected)

    def test_prompt_longer_than_context_is_truncated_not_rejected(self):
        m = _model()
        prompt = list(np.random.randint(1, VOCAB, m.context_length * 3))
        out = generate(m, prompt, max_new_tokens=2)
        self.assertEqual(len(out.split()), len(prompt) + 2)

    def test_sampling_respects_top_k(self):
        """With top_k=1, sampling must collapse onto the greedy choice."""
        m = _model()
        greedy = generate(m, [1, 2], max_new_tokens=6, temperature=0.0)
        sampled = generate(m, [1, 2], max_new_tokens=6, temperature=1.0, top_k=1)
        self.assertEqual(greedy, sampled)

    def test_sampling_can_differ_from_greedy(self):
        m = _model()
        greedy = generate(m, [1, 2], max_new_tokens=25, temperature=0.0)
        np.random.seed(0)
        sampled = generate(m, [1, 2], max_new_tokens=25, temperature=2.0)
        self.assertNotEqual(greedy, sampled)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(_model(), [], max_new_tokens=1)

    def test_string_prompt_without_tokenizer_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(_model(), "hello", max_new_tokens=1)

    def test_stop_token_ends_generation_early(self):
        m = _model()
        prompt = [1, 2]
        first = int(generate(m, prompt, max_new_tokens=1).split()[-1])

        tokenizer = mock.Mock()
        tokenizer.encode.return_value = mock.Mock(ids=[first])
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(str(i) for i in ids)

        out = generate(m, prompt, tokenizer=tokenizer, max_new_tokens=50, stop_tokens=["<eos>"])
        self.assertEqual(len(out.split()), len(prompt) + 1)

    def test_baselm_generate_delegates_to_the_same_implementation(self):
        m = _model()
        self.assertEqual(
            m.generate([3, 4], max_new_tokens=5),
            generate(m, [3, 4], max_new_tokens=5),
        )


class TestEvaluate(unittest.TestCase):
    def test_reports_consistent_derived_metrics(self):
        import math

        with temp_workspace(avg_chars_per_token=4.0):
            cfg_mod.save("r", tiny_config("r"))
            train("r", epochs=1, verbose=0)
            results = evaluate("r", checkpoint="latest", verbose=0)

            # Relative tolerance: the reported values are rounded to 4 dp, so
            # recomputing from the rounded loss cannot match to absolute places.
            self.assertAlmostEqual(
                results["perplexity"],
                math.exp(results["loss"]),
                delta=results["perplexity"] * 1e-3,
            )
            self.assertAlmostEqual(
                results["bpc"], results["loss"] / math.log(2) / 4.0, places=3
            )
            self.assertEqual(results["parameters"]["total"], sum(
                (results["parameters"]["embeddings"], results["parameters"]["mechanism"])
            ))
            self.assertTrue((cfg_mod.run_dir("r") / "eval_latest.json").exists())

    def test_bpc_follows_the_dataset_not_the_run(self):
        """Two runs on one corpus must share `avg_chars_per_token`."""
        with temp_workspace(avg_chars_per_token=2.0):
            cfg_mod.save("r", tiny_config("r"))
            train("r", epochs=1, verbose=0)
            self.assertEqual(evaluate("r", verbose=0)["avg_chars_per_token"], 2.0)

    def test_checkpoint_resolution(self):
        with temp_workspace():
            cfg_mod.save("r", tiny_config("r"))
            train("r", epochs=2, verbose=0)
            self.assertEqual(resolve_checkpoint("r", "latest").name, "model.keras")
            self.assertEqual(resolve_checkpoint("r", "best").name, "best.keras")
            self.assertEqual(resolve_checkpoint("r", 1).name, "checkpoint_00001.keras")
            self.assertEqual(resolve_checkpoint("r", "2").name, "checkpoint_00002.keras")

    def test_missing_checkpoint_raises(self):
        with temp_workspace():
            cfg_mod.save("r", tiny_config("r"))
            train("r", epochs=1, verbose=0)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint("r", 99)

    def test_invalid_checkpoint_spec_raises(self):
        with temp_workspace():
            cfg_mod.save("r", tiny_config("r"))
            with self.assertRaises(ValueError):
                resolve_checkpoint("r", "newest")


if __name__ == "__main__":
    unittest.main()
