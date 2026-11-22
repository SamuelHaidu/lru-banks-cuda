"""`rope-state-memory`: can the model recall a rare multi-token name from earlier in the story?

    uv run python "ad hoc scripts/rsm_name_recall.py" --run-name rsm-1L-prenorm [--checkpoint latest]

A story introduces a name (never or almost never seen in training) and uses it, then ends at a slot where the name
must be recalled. At that slot, teacher-forced, each name piece's probability and rank is measured:
  recall    the name was introduced earlier in the story;
  control   the same story with " Bob" earlier: the model's prior for the name at that slot;
  no-memory the recall story with the memory read removed (gate 0).
Then the greedy continuation from the slot, and the read gate at each name position. CPU only.
"""

import argparse

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
setup_device("cpu")

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402
from model.scripts._model_io import load_tokenizer  # noqa: E402

STORY = ("Once upon a time, there was a little dog named{A}.{A} liked to run and play in the park every day. "
         "One day,{A} found a big red ball under a tree. The ball was shiny and new.{A} was very happy. "
         "He took the ball home and showed it to his mom. His mom smiled and said, \"What a nice ball,")
NAMES = (" Rilo", " Roxo", " Roxy")


def encode(tok, bos, a, b):
    """ids of <bos> + story(A = a) + b, and the positions of b's tokens."""
    parts = STORY.split("{A}")
    ids = [bos]
    for i, p in enumerate(parts):
        ids += tok.encode(p).ids
        if i < len(parts) - 1:
            ids += tok.encode(a).ids
    name = tok.encode(b).ids
    start = len(ids)
    return ids + name, list(range(start, start + len(name))), name


def piece_probs(m, ids, positions):
    """Probability and rank of ids[p] predicted at p - 1, for each p in positions; plus the read gate there."""
    with torch.no_grad():
        x = torch.as_tensor([ids], dtype=torch.int32)
        logits = m.readout(m.readout_input(m.dense_hidden(x)))[0]
        lp = torch.log_softmax(torch.as_tensor(logits), -1)
    out = []
    for p in positions:
        row = lp[p - 1]
        tgt = ids[p]
        rank = int((row > row[tgt]).sum()) + 1
        out.append((float(row[tgt].exp()), rank))
    return out


def gates(m, ids):
    with torch.no_grad():
        x = torch.as_tensor([ids], dtype=torch.int32)
        E = m.token_embedding.embeddings.value
        xs = torch.nn.functional.embedding(x.long(), E)
        u = m._norm(m._w("in_norm", 0), xs) if m.pre_norm else xs
        zero = (E.new_zeros(1, m.dim), E.new_zeros(1, m.memory_rows, m.memory_cols))
        _, _, (s, _) = m._layer(0, u, x == m.reset_token, zero, collect=True, fresh=True)
        return m._gate(0, m.read_gate_in(s, s @ m._w("address_q", 0).value))[0, :, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    m = load_checkpoint(a.run_name, a.checkpoint)
    setup_device("cpu")
    m.use_kernels = False
    tok = load_tokenizer(config)
    bos = m.reset_token
    print(f"run {a.run_name}  checkpoint {a.checkpoint}")
    for name in NAMES:
        ids, pos, pieces = encode(tok, bos, name, name)
        ctl_ids, ctl_pos, _ = encode(tok, bos, " Bob", name)
        print(f"\n=== {name!r}: pieces {[tok.id_to_token(i) for i in pieces]}")
        rec = piece_probs(m, ids, pos)
        ctl = piece_probs(m, ctl_ids, ctl_pos)
        orig = m._gate
        m._gate = lambda n, q: torch.zeros(q.shape[:-1] + (1,))
        off = piece_probs(m, ids, pos)
        m._gate = orig
        g = gates(m, ids)
        print(f"   {'piece':>8} | {'recall p (rank)':>18} | {'control p (rank)':>18} | {'memory off p (rank)':>20} | gate")
        for k, i in enumerate(pieces):
            fmt = lambda r: f"{r[0]:.4f} ({r[1]})"
            print(f"   {tok.id_to_token(i)!r:>8} | {fmt(rec[k]):>18} | {fmt(ctl[k]):>18} | {fmt(off[k]):>20} | {g[pos[k] - 1]:.2f}")
        joint = lambda rs: float(np.prod([r[0] for r in rs]))
        print(f"   whole name: recall {joint(rec):.2e}   control {joint(ctl):.2e}   memory off {joint(off):.2e}")
        # earlier mentions too: the 2nd-4th uses of the name (positions of its first piece)
        ctx = ids[:pos[0]]
        with torch.no_grad():
            gen = list(ctx)
            for _ in range(6):
                logits = m(torch.as_tensor([gen[-m.window_size:]], dtype=torch.int32))
                gen.append(int(np.argmax(np.asarray(logits)[0])))
        print(f"   greedy continuation: ...{tok.decode(ctx[-8:])!r} -> {tok.decode(gen[len(ctx):])!r}")


if __name__ == "__main__":
    main()
