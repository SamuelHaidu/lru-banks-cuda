"""Dataset invariants.

The sampling code is the part of this project that has actually crashed live
training, so its boundary behaviour is pinned down here explicitly rather
than assumed.
"""

import unittest

from tests.helpers import DOC_TOKEN_ID, VOCAB, make_corpus, temp_workspace, tiny_config

import numpy as np

from model import config as cfg_mod
from model.data import (
    FullEvalDataset,
    PackedDocumentDataset,
    VariablePrefixDataset,
    build_datasets,
    build_doc_index,
    build_full_eval_dataset,
)


class TestDocumentIndex(unittest.TestCase):
    def test_index_finds_every_marker(self):
        with temp_workspace(train_doc_lengths=[10, 7, 13]) as tmp:
            path = str(tmp / "datasets" / "tiny" / "train.npy")
            tokens = np.load(path, mmap_mode="r")
            starts = build_doc_index(path, tokens, DOC_TOKEN_ID, use_cache=False)
            np.testing.assert_array_equal(starts, [0, 10, 17])

    def test_cache_is_written_and_reused(self):
        with temp_workspace(train_doc_lengths=[10, 7]) as tmp:
            path = str(tmp / "datasets" / "tiny" / "train.npy")
            tokens = np.load(path, mmap_mode="r")
            first = build_doc_index(path, tokens, DOC_TOKEN_ID, use_cache=True)
            self.assertTrue((tmp / "datasets" / "tiny" / "train.npy.doc_idx.npy").exists())
            second = build_doc_index(path, tokens, DOC_TOKEN_ID, use_cache=True)
            np.testing.assert_array_equal(first, second)

    def test_cache_built_for_a_different_marker_is_rejected(self):
        """A stale cache must be detected, not silently trusted."""
        with temp_workspace(train_doc_lengths=[10, 7]) as tmp:
            path = str(tmp / "datasets" / "tiny" / "train.npy")
            tokens = np.load(path, mmap_mode="r")
            build_doc_index(path, tokens, DOC_TOKEN_ID, use_cache=True)
            # Marker 5 never sits at position 0, so the cached [0, 10] is invalid for it.
            rebuilt = build_doc_index(path, tokens, 5, use_cache=True)
            self.assertTrue(all(int(tokens[i]) == 5 for i in rebuilt))


class TestVariablePrefixDataset(unittest.TestCase):
    def _dataset(self, tmp, **kw):
        params = dict(
            path=str(tmp / "datasets" / "tiny" / "train.npy"),
            window_size=8,
            batch_size=16,
            vocabulary_size=VOCAB,
            doc_token_id=DOC_TOKEN_ID,
            steps_per_epoch=4,
        )
        params.update(kw)
        return VariablePrefixDataset(**params)

    def test_shapes_and_dtypes(self):
        with temp_workspace() as tmp:
            ds = self._dataset(tmp)
            x, y = ds[0]
            self.assertEqual(x.shape, (16, 8))
            self.assertEqual(y.shape, (16,))
            self.assertEqual(x.dtype, np.int32)
            self.assertEqual(y.dtype, np.int32)
            self.assertEqual(len(ds), 4)

    def test_windows_never_cross_a_document_boundary(self):
        """Everything left of a document's start must be zero padding."""
        with temp_workspace(train_doc_lengths=[30, 30, 30]) as tmp:
            path = str(tmp / "datasets" / "tiny" / "train.npy")
            tokens = np.asarray(np.load(path))
            ds = self._dataset(tmp, batch_size=64, steps_per_epoch=1)
            for _ in range(20):
                x, y = ds[0]
                for row, target in zip(x, y):
                    # Reconstruct the target position from the (unique) suffix.
                    real = row[row != 0] if (row == 0).any() else row
                    # Any padding must be a strict left-hand prefix: once a real
                    # token appears there can be no padding after it.
                    nonzero = np.flatnonzero(row != 0)
                    if nonzero.size:
                        self.assertTrue(
                            np.all(np.diff(nonzero) == 1),
                            "padding must be contiguous on the left only",
                        )
                        self.assertEqual(nonzero[-1], len(row) - 1, "the window must end at the target")
                    del real, target

    def test_padding_is_left_aligned_for_short_prefixes(self):
        """A document shorter than the window must be left-padded, not wrapped.

        Uses marker id 1 with bodies drawn from `[2, VOCAB)`, so 0 means
        padding and nothing else -- with the real corpus's marker (`<bos>`,
        id 0) a zero in the window is genuinely ambiguous.
        """
        with temp_workspace(train_doc_lengths=[4, 4, 4], doc_token_id=1, body_low=2) as tmp:
            ds = self._dataset(tmp, window_size=16, batch_size=32, steps_per_epoch=1, doc_token_id=1)
            x, _ = ds[0]
            for row in x:
                nonzero = np.flatnonzero(row != 0)
                self.assertGreater(nonzero.size, 0)
                self.assertEqual(nonzero[-1], len(row) - 1, "the window must end at the target")
                self.assertTrue(np.all(np.diff(nonzero) == 1), "padding is a left prefix only")
                # Document is [marker, b1, b2, b3]; the longest possible
                # context is marker + 2 body tokens (target is the last one).
                self.assertLessEqual(nonzero.size, 3)

    def test_last_document_boundary_does_not_overrun(self):
        """Regression: `doc_ends` is exclusive.

        With a corpus whose final document runs to the very last token,
        sampling with an inclusive upper bound reads `tokens[len(tokens)]` and
        raises IndexError. This crashed live training in the previous
        codebase; the bound must stay exclusive.
        """
        with temp_workspace(train_doc_lengths=[6]) as tmp:
            ds = self._dataset(tmp, window_size=4, batch_size=8, steps_per_epoch=1)
            for _ in range(200):
                x, y = ds[0]
                self.assertEqual(x.shape, (8, 4))
                self.assertEqual(y.shape, (8,))

    def test_targets_are_never_the_document_marker(self):
        """Targets start one past the marker, so a marker is never predicted."""
        with temp_workspace(train_doc_lengths=[20, 20]) as tmp:
            ds = self._dataset(tmp, batch_size=64, steps_per_epoch=1)
            for _ in range(20):
                _, y = ds[0]
                self.assertFalse((y == DOC_TOKEN_ID).any())

    def test_corruption_only_touches_valid_positions_and_never_the_target(self):
        """Padding stays zero, targets stay clean, valid positions do change.

        Marker id 1 (bodies from `[2, VOCAB)`) so a zero unambiguously means
        padding; with marker 0 a legitimate `<bos>` in the window is
        indistinguishable from padding and the assertion would be meaningless.
        """
        with temp_workspace(train_doc_lengths=[40, 40], doc_token_id=1, body_low=2) as tmp:
            common = dict(batch_size=64, steps_per_epoch=1, seed=7, doc_token_id=1)
            clean = self._dataset(tmp, corrupt_prob=0.0, **common)
            dirty = self._dataset(tmp, corrupt_prob=0.9, **common)
            x_clean, y_clean = clean[0]
            x_dirty, y_dirty = dirty[0]
            # Same seed -> same documents, positions and targets.
            np.testing.assert_array_equal(y_clean, y_dirty)
            # Padding positions must remain exactly zero.
            pad = x_clean == 0
            self.assertTrue(np.all(x_dirty[pad] == 0))
            # With p=0.9 the valid positions must actually change.
            self.assertGreater(int((x_clean != x_dirty).sum()), 0)

    def test_corrupt_prob_is_range_checked(self):
        with temp_workspace() as tmp:
            with self.assertRaises(ValueError):
                self._dataset(tmp, corrupt_prob=1.0)
            with self.assertRaises(ValueError):
                self._dataset(tmp, corrupt_prob=-0.1)

    def test_missing_marker_raises_clearly(self):
        with temp_workspace() as tmp:
            with self.assertRaises(ValueError):
                self._dataset(tmp, doc_token_id=999)


class TestFullEvalDataset(unittest.TestCase):
    def test_covers_every_document_exactly_once(self):
        with temp_workspace(validation_doc_lengths=[20, 15, 30, 25, 11]) as tmp:
            ds = FullEvalDataset(
                path=str(tmp / "datasets" / "tiny" / "validation.npy"),
                window_size=8,
                batch_size=2,
                doc_token_id=DOC_TOKEN_ID,
            )
            self.assertEqual(ds.num_docs, 5)
            self.assertEqual(len(ds), 3)
            total = sum(len(ds[i][1]) for i in range(len(ds)))
            self.assertEqual(total, 5)

    def test_is_reproducible_across_instances(self):
        with temp_workspace(validation_doc_lengths=[20, 15, 30]) as tmp:
            path = str(tmp / "datasets" / "tiny" / "validation.npy")
            kw = dict(window_size=8, batch_size=4, doc_token_id=DOC_TOKEN_ID, seed=3)
            a = FullEvalDataset(path=path, **kw)
            b = FullEvalDataset(path=path, **kw)
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.y, b.y)


class TestBuilders(unittest.TestCase):
    def test_dataset_facts_come_from_metadata_not_config(self):
        """`doc_token_id`/`vocabulary_size` must be read from the dataset."""
        with temp_workspace() as tmp:
            cfg = tiny_config()
            train, val = build_datasets(cfg, model_context_length=8)
            self.assertEqual(train.window_size, 8)
            self.assertEqual(train.vocabulary_size, VOCAB)
            self.assertEqual(len(train), cfg["data"]["steps_per_epoch"])
            self.assertEqual(len(val), cfg["training"]["validation_steps"])
            # Validation must never be corrupted.
            self.assertEqual(val.corrupt_prob, 0.0)
            del tmp

    def test_full_eval_builder_uses_the_validation_split(self):
        with temp_workspace(validation_doc_lengths=[20, 15, 30, 12]) as tmp:
            ds = build_full_eval_dataset(tiny_config(), model_context_length=8)
            self.assertEqual(ds.num_docs, 4)
            del tmp

    def test_unknown_dataset_raises(self):
        with temp_workspace():
            with self.assertRaises(cfg_mod.ConfigError):
                cfg_mod.load_dataset_metadata("nope")


class TestLossSpans(unittest.TestCase):
    """`loss_spans`: only tokens inside open ... close (the close included) are targets, in every sampler."""

    OPEN, CLOSE, BOS = 6, 7, 0

    def _corpus(self, tmp):
        import os

        # <bos> 20 21 <open> 30 31 <close> 22 <open> 32 <close> | <bos> 23 <open> 33 34 35 <close> 24
        toks = [0, 20, 21, 6, 30, 31, 7, 22, 6, 32, 7, 0, 23, 6, 33, 34, 35, 7, 24]
        want = [False, False, False, False, True, True, True, False, False, True, True,
                False, False, False, True, True, True, True, False]
        path = os.path.join(tmp, "toks.npy")
        np.save(path, np.asarray(toks * 30, dtype=np.int32))
        return path, np.asarray(want * 30)

    def test_mask_and_samplers(self):
        import tempfile

        from model.data import PackedDocumentDataset, loss_target_mask

        with tempfile.TemporaryDirectory() as tmp:
            path, want = self._corpus(tmp)
            tokens = np.load(path)
            mask = loss_target_mask(path, tokens, [[self.OPEN, self.CLOSE]], self.BOS)
            np.testing.assert_array_equal(mask, want)
            np.testing.assert_array_equal(loss_target_mask(path, tokens, [[self.OPEN, self.CLOSE]], self.BOS), want)  # cache
            self.assertIsNone(loss_target_mask(path, tokens, None, self.BOS))
            spans = [[self.OPEN, self.CLOSE]]
            packed = PackedDocumentDataset(path, 8, 64, 40, self.BOS, steps_per_epoch=1, loss_spans=spans, use_cache=False)
            x, y = packed[0]
            for r in range(len(x)):  # every row starts at a marker: recover its offset in the 19-token cycle
                start = 0 if x[r, 1] == 20 else 11
                for k in range(8):
                    p = (start + k + 1) % 19
                    self.assertEqual(y[r, k] >= 0, bool(want[p]) and tokens[p] != self.BOS)
            single = VariablePrefixDataset(path, 8, 256, 40, self.BOS, steps_per_epoch=1, loss_spans=spans, use_cache=False)
            _, ys = single[0]
            self.assertTrue(set(ys.tolist()) <= {30, 31, 32, 33, 34, 35, self.CLOSE})
            full = FullEvalDataset(path, 8, 16, self.BOS, loss_spans=spans, use_cache=False)
            self.assertEqual(full.num_docs, 60)
            self.assertTrue(set(full.y.tolist()) <= {30, 31, 32, 33, 34, 35, self.CLOSE})


class TestRandomWindows(unittest.TestCase):
    def test_each_epoch_is_one_pass_in_shuffled_windows(self):
        """Every window of an epoch is distinct and consecutive from the epoch's offset; together they cover every
        token but the < 2 T at the ends; the next epoch has another offset and order; steps are windows / batch."""
        with temp_workspace(train_doc_lengths=[37, 50, 23, 61, 44, 29, 55]) as tmp:
            T, B = 8, 3
            ds = PackedDocumentDataset(path=str(tmp / "datasets" / "tiny" / "train.npy"), window_size=T, batch_size=B,
                                       vocabulary_size=VOCAB, doc_token_id=DOC_TOKEN_ID, random_windows=True,
                                       steps_multiple=2)
            n = len(ds.tokens)
            self.assertEqual(len(ds), ((n - T) // T // B) // 2 * 2)
            seen = []
            for epoch in range(2):
                ds._epoch = epoch
                offset, order = ds._window_order(epoch)
                xs = np.concatenate([ds[i][0] for i in range(len(ds))])
                starts = offset + order[:len(ds) * B] * T
                self.assertEqual(len(set(starts.tolist())), len(starts))
                for row, s in zip(xs, starts):
                    np.testing.assert_array_equal(row, np.asarray(ds.tokens[s:s + T]))
                covered = np.zeros(n, bool)
                for s in offset + order * T:
                    covered[s:s + T] = True
                self.assertGreaterEqual(covered.sum(), n - 2 * T)
                seen.append((offset, order[:5].tolist()))
            self.assertNotEqual(seen[0], seen[1])

    def test_quarter_pass_epochs_continue_the_pass(self):
        """epoch_passes 0.25: four consecutive epochs serve one pass's windows, none twice."""
        with temp_workspace(train_doc_lengths=[37, 50, 23, 61, 44, 29, 55, 70, 33]) as tmp:
            T, B = 4, 2
            ds = PackedDocumentDataset(path=str(tmp / "datasets" / "tiny" / "train.npy"), window_size=T, batch_size=B,
                                       vocabulary_size=VOCAB, doc_token_id=DOC_TOKEN_ID, random_windows=True,
                                       epoch_passes=0.25)
            self.assertEqual(len(ds), ds.steps_per_pass // 4)
            rows = []
            for epoch in range(4):
                ds._epoch = epoch
                rows += [r.tobytes() for i in range(len(ds)) for r in ds[i][0]]
            offset, order = ds._window_order(0)
            want = [np.asarray(ds.tokens[offset + w * T:offset + w * T + T]).astype(np.int32).tobytes()
                    for w in order[:len(rows)]]
            self.assertEqual(rows, want)


if __name__ == "__main__":
    unittest.main()
