"""Dataset-specific training callbacks.

Kept separate from `model/train.py`'s core loop: every architecture and every
dataset shares the same `fit()` call, and `sparse_categorical_accuracy` (the
only metric wired into `compile()`) is a per-*token* average -- for a doc like
`<bos> a0 a1 op b0 b1 = r0 r1 <eos>` most of those tokens (digits of `a`/`b`,
`op`, `<eos>`) are easy structural predictions, so that average can look fine
while the actual arithmetic (predicting `r0`/`r1` from `=`) is still at
chance. `ModularArithmeticEquationAccuracy` measures exactly that: greedy
next-token accuracy restricted to the result digits, and exact-match on the
full two-digit result, evaluated over the complete `a op b` grid rather than a
sampled batch.

A callback (not a `keras.metrics.Metric`) because `Metric.update_state` only
ever sees `(y_true, y_pred)` -- the per-position sampling in
`VariablePrefixDataset` never tells a metric object whether a given target
token is `r0`, `r1`, or a digit of an operand. Only code with the full
document structure (as this callback builds directly) can compute an
equation-level accuracy.
"""

from typing import Any, Dict, Optional

import numpy as np

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402


class ModularArithmeticEquationAccuracy(keras.callbacks.Callback):
    """Logs whole-equation accuracy for a fixed-modulus `a op b = r` dataset.

    Appends to the epoch's `logs` dict (picked up by `CSVLogger` and the
    default progress bar) rather than printing separately, so it lands in
    `metrics.csv` next to `loss`/`val_loss` with no changes to `train.py`'s
    core loop.

    Assumes the doc layout used by `datasets/build_modular_arithmetic.py`:
    `<bos> a0 a1 op b0 b1 = r0 r1 <eos>` (2-digit zero-padded operands and
    result). Covers the full `modulus x modulus x len(ops)` grid every call --
    cheap here (97*97*3 ~= 28k forward passes) but not assumed to scale to a
    larger modulus.
    """

    def __init__(
        self,
        modulus: int,
        ops: Dict[int, str],
        bos_token_id: int,
        equals_token_id: int,
        log_prefix: str = "equation",
    ):
        super().__init__()
        self.modulus = int(modulus)
        self.ops = dict(ops)
        self.bos_token_id = int(bos_token_id)
        self.equals_token_id = int(equals_token_id)
        self.log_prefix = log_prefix
        self._prefixes: Optional[np.ndarray] = None
        self._true_r0: Optional[np.ndarray] = None
        self._true_r1: Optional[np.ndarray] = None
        self._op_of_row: Optional[np.ndarray] = None

    @staticmethod
    def _digits(n: int) -> tuple:
        return (n // 10, n % 10)

    def _build_grid(self, window: int) -> None:
        rows, r0s, r1s, op_ids = [], [], [], []
        for op_id, fn_name in self.ops.items():
            fn = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}[fn_name]
            for a in range(self.modulus):
                a0, a1 = self._digits(a)
                for b in range(self.modulus):
                    b0, b1 = self._digits(b)
                    r = fn(a, b) % self.modulus
                    r0, r1 = self._digits(r)
                    rows.append((self.bos_token_id, a0, a1, op_id, b0, b1, self.equals_token_id))
                    r0s.append(r0)
                    r1s.append(r1)
                    op_ids.append(op_id)
        prefixes = np.array(rows, dtype=np.int32)
        pad = np.zeros((prefixes.shape[0], window - prefixes.shape[1]), dtype=np.int32)
        self._prefixes = np.concatenate([pad, prefixes], axis=1)
        self._true_r0 = np.array(r0s, dtype=np.int32)
        self._true_r1 = np.array(r1s, dtype=np.int32)
        self._op_of_row = np.array(op_ids, dtype=np.int32)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            return
        window = self.model.context_length
        if self._prefixes is None:
            self._build_grid(window)

        pred_r0 = np.argmax(np.asarray(self.model.predict(self._prefixes, verbose=0)), axis=-1)
        prefix8 = np.concatenate([self._prefixes[:, 1:], pred_r0[:, None].astype(np.int32)], axis=1)
        pred_r1 = np.argmax(np.asarray(self.model.predict(prefix8, verbose=0)), axis=-1)

        r0_correct = pred_r0 == self._true_r0
        exact_match = r0_correct & (pred_r1 == self._true_r1)

        logs[f"{self.log_prefix}_r0_accuracy"] = float(r0_correct.mean())
        logs[f"{self.log_prefix}_exact_match"] = float(exact_match.mean())
        for op_id, name in self.ops.items():
            mask = self._op_of_row == op_id
            if mask.any():
                logs[f"{self.log_prefix}_exact_match_{name}"] = float(exact_match[mask].mean())
