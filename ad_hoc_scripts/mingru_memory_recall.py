"""Does a static-memory `min-gru` run recall from its system / tool memories? On validation documents with system or
tool spans:

  1. ablation: the loss of the scored tokens (the assistant's) with the memory reads, with uniform reads (`mem_k` = 0:
     every key is the same, so each read is the plain average of its memory's values: a summary of what the system /
     tool spans hold, no selection) and with the reads switched off (`mem_out` = 0: nothing from the memories), on all
     scored tokens and on the "copyable" ones (tokens that also occur in the document's system / tool spans: tool
     names, parameter names, ...). on vs uniform is what the selective recall is worth;
  2. recall hits, per layer: at the positions predicting a copyable token, is the stored token with the largest read
     weight (q . k_j, normalized per memory, mean over the heads; in the memory holding the target) the target itself?
     Against chance: the share of that memory's entries equal to the target;
  3. the memory gates per layer (mean over the reading positions and heads).

    uv run python ad_hoc_scripts/mingru_memory_recall.py --run-name mingru-4L-v2-mem-cuda [--checkpoint latest]
        [--docs 150] [--device xpu]
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from model.device import setup_device

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "xpu"
setup_device(DEVICE)

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.memory_rows import span_masks  # noqa: E402


def documents(dataset: str, split: str):
    """Whole validation documents (token ids), in corpus order, as `(ids, category)`."""
    meta = cfg_mod.load_dataset_metadata(dataset)
    tokens = np.load(cfg_mod.dataset_split_path(dataset, split), mmap_mode="r")
    starts = np.flatnonzero(np.asarray(tokens) == meta["doc_token_id"])
    cats = None
    path = cfg_mod.DATASETS_DIR / dataset / f"{split}.categories.npy"
    if path.exists():
        names = {v: k for k, v in meta.get("category_ids", {}).items()}
        rows = np.load(path)
        cats = {int(s): names.get(int(c), str(c)) for s, _, c in rows}
    for a, b in zip(starts, np.append(starts[1:], len(tokens))):
        yield np.asarray(tokens[a:b]).astype(np.int32), (cats or {}).get(int(a), "?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--docs", type=int, default=150, help="documents with system / tool spans and scored tokens")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--device", default="xpu")
    a = p.parse_args()
    cfg = cfg_mod.load(a.run_name)
    model = load_checkpoint(a.run_name, int(a.checkpoint) if a.checkpoint.isdigit() else a.checkpoint)
    if not getattr(model, "static_memory", False):
        raise SystemExit(f"{a.run_name} has no static memory")
    spans = cfg_mod.load_dataset_metadata(cfg["data"]["dataset"]).get("loss_spans") or [model.memory_read_span]
    h, d, nm, L = model.memory_heads, model.memory_dim, len(model.memory_spans), model.num_layers
    def ablated(name, mb):
        """The losses with every layer's `name` weight zeroed (restored after)."""
        ws = [lay[name].value for lay in model._params]
        saved = [w.detach().clone() for w in ws]
        with torch.no_grad():
            for w in ws:
                w.zero_()
            out = losses(mb)[0]
            for w, s in zip(ws, saved):
                w.copy_(s)
        return out

    def losses(mb):
        """-log p of every next token over the kept row, and the scored mask (targets inside the loss spans)."""
        with torch.no_grad():
            hid = model._run_memory(mb)[0]
            logp = torch.log_softmax(model.readout(model.readout_input(hid)).float(), -1).cpu().numpy()
        ids = mb["ids"][0].cpu().numpy()
        target = np.zeros(len(ids), bool)
        for o, c in spans:
            inside = False
            for j, t in enumerate(ids):
                inside = False if t == model.reset_token else inside
                target[j] |= inside
                inside = True if t == o else (False if t == c else inside)
        return -logp[np.arange(len(ids) - 1), ids[1:]], target[1:], ids

    stats = defaultdict(list)
    hits = np.zeros((L, 2))  # per layer: hits, chance (summed over the copyable predictions)
    n_copy = 0
    gates = np.zeros((L, nm))
    n_read = 0
    n_docs = 0
    for ids, cat in documents(cfg["data"]["dataset"], "validation"):
        if n_docs >= a.docs:
            break
        ids = ids[:a.max_tokens]
        kind = span_masks(ids.astype(np.int64), model.memory_spans, model.memory_read_span, model.reset_token)[0]
        if not (kind > 0).any():
            continue
        mb = model._from_raw(ids[None])
        on, scored, kept_ids = losses(mb)
        if not scored.any():
            continue
        off, uniform = ablated("mem_out", mb), ablated("mem_k", mb)
        n_docs += 1
        stored = set(ids[kind > 0].tolist())
        copy = scored & np.isin(kept_ids[1:], list(stored))
        for name, v in (("on", on), ("off", off), ("uni", uniform)):
            stats[name] += v[scored].tolist()
            stats["copy_" + name] += v[copy].tolist()
        stats[f"cat_on:{cat}"] += on[scored].tolist()
        stats[f"cat_off:{cat}"] += off[scored].tolist()
        # recall: the read weights at the positions predicting a copyable token
        infos, meta = model.inspect_memory(ids[None])
        tokens = ids[meta["entry_pos"]]
        ekind, edoc = meta["entry_kind"], meta["entry_doc"]
        row = {t: r for r, t in enumerate(meta["read_pos"])}
        for n, lay in enumerate(infos):
            if len(meta["read_pos"]):
                gates[n] += lay["memory_gate"][0][meta["read_pos"]].reshape(-1, nm, h).mean(-1).sum(0)
        n_read += len(meta["read_pos"])
        for t in np.flatnonzero(copy):  # position t predicts kept_ids[t + 1]
            if t not in row:
                continue
            r, tgt = row[t], kept_ids[t + 1]
            mine_doc = edoc == meta["read_doc"][r]
            for i in range(nm):
                mine = mine_doc & (ekind == i)
                if not (mine & (tokens == tgt)).any():
                    continue  # the target is not in this memory
                for n, lay in enumerate(infos):
                    q = lay["memory_query"][0, t].reshape(h, d)
                    s = np.einsum("hd,jhd->jh", q, meta["entry_keys"][mine][:, n]).mean(1)
                    hits[n, 0] += tokens[mine][np.argmax(s)] == tgt
                    hits[n, 1] += (tokens[mine] == tgt).mean()
                n_copy += 1
                break
    mean = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")
    print(f"{a.run_name} ({a.checkpoint}): {n_docs} validation documents with system / tool spans")
    print(f"\n1. loss of the scored tokens (nats): memory reads on / uniform (no selection) / off")
    for label, k in (("all scored", ""), ("copyable", "copy_")):
        on_, uni_, off_ = mean(k + "on"), mean(k + "uni"), mean(k + "off")
        print(f"   {label:16s} {len(stats[k + 'on']):6d} tokens   on {on_:.4f}   uniform {uni_:.4f}   off {off_:.4f}   "
              f"memory worth {off_ - on_:+.4f}, of which selection {uni_ - on_:+.4f}")
    for k in sorted(k for k in stats if k.startswith("cat_on:")):
        c = k.split(":", 1)[1]
        print(f"   {c:16s} {len(stats[k]):6d} tokens   on {mean(k):.4f}   off {mean('cat_off:' + c):.4f}   "
              f"memory worth {mean('cat_off:' + c) - mean(k):+.4f}")
    print(f"\n2. recall: the most-read stored token is the target ({n_copy} copyable predictions)")
    for n in range(L):
        print(f"   layer {n + 1}: hit {hits[n, 0] / max(n_copy, 1):6.1%}   chance {hits[n, 1] / max(n_copy, 1):6.1%}")
    print(f"\n3. memory gates (mean over {n_read} reading positions and the heads)")
    for n in range(L):
        print(f"   layer {n + 1}: " + "   ".join(f"{name} {g / max(n_read, 1):.3f}" for name, g in
                                           zip(["system", "tools"] + [f"m{i}" for i in range(2, nm)], gates[n])))


if __name__ == "__main__":
    main()
