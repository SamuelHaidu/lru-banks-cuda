"""Ad hoc: does `modarith-baseline` actually learn the operation, or just the
easy structural next-token predictions (digit-of-a, digit-of-b, `<eos>`, ...)?

Builds exact prefixes up to and including `=` for every validation example,
predicts `r0` greedily, appends it, predicts `r1` greedily, and reports:
  - accuracy at the `r0` position alone
  - accuracy at the `r1` position alone (conditioned on the true r0, not the
    model's own r0 -- isolates "can it read off the second digit" from
    compounding first-digit errors)
  - exact-match accuracy on the full 2-digit result (using the model's own
    r0 to predict r1, i.e. actual autoregressive behavior)
  - the same three, broken down by operator (+, -, *)

Usage:
    .venv/bin/python "ad hoc scripts/modarith_operation_accuracy.py"
"""

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.data import load_tokens  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402

RUN_NAME = "modarith-baseline"
CHECKPOINT = "latest"

BOS, EOS, PLUS, MINUS, STAR, EQUALS = 10, 11, 12, 13, 14, 15
OP_NAMES = {PLUS: "+", MINUS: "-", STAR: "*"}
DOC_LEN = 10  # <bos> a0 a1 op b0 b1 = r0 r1 <eos>


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))

    model = load_checkpoint(RUN_NAME, CHECKPOINT)
    window = model.context_length

    val_path = cfg_mod.dataset_split_path("modular-arithmetic-97", "validation")
    tokens = np.asarray(load_tokens(str(val_path)))
    assert tokens.size % DOC_LEN == 0
    docs = tokens.reshape(-1, DOC_LEN)

    n = len(docs)
    # Positions within a doc: 0=<bos> 1=a0 2=a1 3=op 4=b0 5=b1 6=`=` 7=r0 8=r1 9=<eos>
    ops = docs[:, 3]
    true_r0 = docs[:, 7]
    true_r1 = docs[:, 8]

    def left_pad(seqs: np.ndarray) -> np.ndarray:
        """Left-pads a batch of variable-irrelevant-length prefixes to `window`."""
        L = seqs.shape[1]
        if L >= window:
            return seqs[:, -window:]
        pad = np.zeros((seqs.shape[0], window - L), dtype=np.int32)
        return np.concatenate([pad, seqs], axis=1)

    # Predict r0 from the true prefix <bos> a0 a1 op b0 b1 = (7 tokens).
    prefix7 = left_pad(docs[:, :7].astype(np.int32))
    logits_r0 = model.predict(prefix7, verbose=0)
    pred_r0 = np.argmax(logits_r0, axis=-1)

    # Predict r1 given the TRUE r0 appended (isolates second-digit readout).
    prefix8_true = left_pad(np.concatenate([docs[:, :7], true_r0[:, None]], axis=1).astype(np.int32))
    logits_r1_true = model.predict(prefix8_true, verbose=0)
    pred_r1_given_true_r0 = np.argmax(logits_r1_true, axis=-1)

    # Predict r1 given the MODEL'S OWN r0 (actual autoregressive generation).
    prefix8_pred = left_pad(np.concatenate([docs[:, :7], pred_r0[:, None]], axis=1).astype(np.int32))
    logits_r1_pred = model.predict(prefix8_pred, verbose=0)
    pred_r1_given_pred_r0 = np.argmax(logits_r1_pred, axis=-1)

    acc_r0 = pred_r0 == true_r0
    acc_r1_given_true = pred_r1_given_true_r0 == true_r1
    exact_match = acc_r0 & (pred_r1_given_pred_r0 == true_r1)

    def report(mask: np.ndarray, label: str) -> None:
        count = mask.sum()
        if count == 0:
            return
        print(f"{label:>8}  n={count:6d}  r0_acc={acc_r0[mask].mean():.4f}  "
              f"r1_acc|true_r0={acc_r1_given_true[mask].mean():.4f}  "
              f"exact_match={exact_match[mask].mean():.4f}")

    print(f"{RUN_NAME} @ {CHECKPOINT}  (n={n} validation examples)\n")
    report(np.ones(n, dtype=bool), "ALL")
    print()
    for op_id, name in OP_NAMES.items():
        report(ops == op_id, name)

    print("\nBaseline: a random guess over 10 digits gets r0_acc=0.10, exact_match=0.01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
