"""GRPO (group relative policy optimization) on the Calc-X calculator task, for stateful archs (`advance`; e.g.
`lru-banks`).

    python -m model.grpo --run-name X-grpo --init-run X [--init-checkpoint latest] [--steps 200] [--prompts 1024]
        [--chunk-prompts 128] [--group 8] [--lr 5e-5] [--sources gsm8k mawps ape210k] [--eval-every 5]

Each step:
1. rollout: `--prompts` training questions (Calc-X train split, numeric answers only: multiple choice gives a 20%
   reward by guessing), each answered `--group` times by sampling at `--temperature` (1.0: the policy itself), through
   the real agent loop (`model/calcx.py`: calls executed, results fed back). Generation runs the model's stateful path.
2. reward: 1 if the final "The answer is X" is correct, else 0 (no answer, a truncated answer, a wrong one); with
   `--partial-credit beta`, a wrong answer gets beta x the share of the reference solution's calculator steps it also
   computed (`calcx.step_credit`: one-to-one matches of tool results, numbers written in the question excluded,
   divided by max(its calls, the reference's steps): extra calls cost credit). beta < 1: finishing right always wins.
3. advantage per answer: (r - group mean) / max(group std, `--std-floor`) (GRPO; the floor keeps tiny partial-credit
   differences from being blown up to full-size advantages); `--no-std`: r - mean only (Dr. GRPO). Groups where
   every answer got the same reward carry no signal and are dropped.
4. one update per step, on-policy, by gradient accumulation: the questions are rolled out `--chunk-prompts` at a
   time and each chunk's gradient is added (every chunk sampled from the same weights), then divided by the step's
   token count, clipped and applied: exactly the gradient of the whole batch. Loss = sum over the model's own tokens (tool results and prompts excluded) of
   -A * log p(token) / the number of those tokens in the step (token-level mean, as DAPO). With a single update per
   rollout the importance ratio is exactly 1, so no clipping and no old-policy pass are needed. No KL penalty: the
   forgetting is measured instead (pretraining validation loss at every evaluation). The update runs the training
   forward pass (fused kernels on CUDA) with the readout's cross-entropy weighted per token by A
   (`fused_linear_cross_entropy(weights=...)`: constant logits memory), micro-batches of `--micro-rows` rows
   sorted by length, AdamW (no weight decay), gradient clipping.

Every `--eval-every` steps (and at step 0): greedy accuracy on a fixed random sample of the Calc-X test splits
(`--eval-limit` per source, all 6 sources: svamp and asdiv_a have no train split, so they measure transfer) and the
pretraining validation loss on fixed rows (the forgetting measure). Writes in runs/<run>/: config.json (the init run's,
so the run loads anywhere), grpo.json (the arguments), grpo_metrics.jsonl (one line per step and per evaluation),
model.keras after each evaluation, checkpoints/best.keras at the best evaluation accuracy.
"""

import argparse
import json
import math
import shutil
import sys
import time
from typing import Any, Dict, List

import numpy as np

from model import calcx
from model import config as cfg_mod


def eval_set(path: Any, limit: int, seed: int = 0) -> List[Dict[str, str]]:
    items = calcx.load_questions(path)
    if not limit:
        return items
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted({it["source"] for it in items}):
        mine = [it for it in items if it["source"] == s]
        out += [mine[i] for i in sorted(rng.choice(len(mine), min(limit, len(mine)), replace=False))]
    return out


def evaluate(m: Any, tok: Any, items: List[Dict[str, str]], a: Any) -> Dict[str, Any]:
    outs = calcx.rollout(m, tok, [it["prompt"] for it in items], a.max_new, 0.0, a.gen_batch)
    ok = [calcx.correct(o["pred"], it["gold"]) for it, o in zip(items, outs)]
    by = {}
    for it, k in zip(items, ok):
        by.setdefault(it["source"], []).append(k)
    return {"accuracy": float(np.mean(ok)), "sources": {s: float(np.mean(v)) for s, v in sorted(by.items())},
            "answered": float(np.mean([o["pred"] is not None for o in outs]))}


def pretrain_rows(dataset: str, rows: int, window: int, seed: int = 0) -> np.ndarray:
    """Fixed windows of the run's validation split (`window + 1` tokens each, random offsets)."""
    v = np.load(calcx.ROOT / "datasets" / dataset / "validation.npy", mmap_mode="r")
    starts = np.random.default_rng(seed).integers(0, len(v) - window - 1, rows)
    return np.stack([np.asarray(v[s:s + window + 1], np.int64) for s in starts])


def pretrain_loss(m: Any, rows: np.ndarray, micro: int) -> float:
    import torch

    from model.fused_linear_cross_entropy import fused_linear_cross_entropy

    total, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(rows), micro):
            r = torch.as_tensor(rows[b:b + micro])
            valid = torch.ones(r.shape[0], r.shape[1] - 1, dtype=torch.bool)
            hv = m.dense_hidden_valid(r[:, :-1], valid)
            y = r[:, 1:].reshape(-1).to(hv.device)
            total += float(fused_linear_cross_entropy(m.readout_input(hv), m.readout.kernel, y)) * y.numel()
            n += y.numel()
    return total / n


def accumulate(m: Any, batch: List[Dict[str, Any]], a: Any, pad: int) -> Dict[str, float]:
    """Adds the gradient of sum_tokens A * nll over the rollouts in `batch` (each with `ids`, `n_prompt`, `sampled`,
    `adv`) to the weights' .grad (not normalized: `apply_update` divides by the step's token count)."""
    import torch

    from model.fused_linear_cross_entropy import fused_linear_cross_entropy

    batch = sorted(batch, key=lambda r: len(r["ids"]))
    loss_sum, n_sum = 0.0, 0
    for b in range(0, len(batch), a.micro_rows):
        mb = batch[b:b + a.micro_rows]
        T = int(math.ceil((max(len(r["ids"]) for r in mb) - 1) / 128) * 128)  # few distinct shapes: few compiles
        x = np.full((a.micro_rows, T), pad, np.int64)  # padded rows: no target
        y = np.full((a.micro_rows, T), -1, np.int64)
        w = np.zeros((a.micro_rows, T), np.float32)
        for i, r in enumerate(mb):
            ids, n0 = r["ids"], r["n_prompt"]
            x[i, :len(ids) - 1] = ids[:-1]
            pos = n0 - 1 + np.flatnonzero(r["sampled"])  # position i predicts ids[i + 1]
            y[i, pos] = np.asarray(ids)[pos + 1]
            w[i, pos] = r["adv"]  # minimizing A * nll = maximizing A * log p
        valid = torch.as_tensor(y >= 0)
        hv = m.dense_hidden_valid(torch.as_tensor(x), valid)
        dev = hv.device
        yv = torch.as_tensor(y, device=dev)[valid.to(dev)]
        wv = torch.as_tensor(w, device=dev)[valid.to(dev)]
        # fused CE = sum(w nll) / n_mb: times n_mb, the sum
        loss = fused_linear_cross_entropy(m.readout_input(hv), m.readout.kernel, yv, weights=wv) * len(yv)
        loss.backward()
        loss_sum += float(loss.detach())
        n_sum += len(yv)
    return {"loss_sum": loss_sum, "tokens": n_sum}


def apply_update(opt: Any, params: List[Any], n_tokens: int, clip: float) -> float:
    """The accumulated sum gradient / n_tokens (the step's token-level mean), clipped, applied. Returns its norm."""
    import torch

    with torch.no_grad():
        for p_ in params:
            if p_.grad is not None:
                p_.grad.div_(max(n_tokens, 1))
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return gnorm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--init-run", required=True)
    p.add_argument("--init-checkpoint", default="latest")
    p.add_argument("--train-data", default=str(calcx.DATA / "train.docs.jsonl"))
    p.add_argument("--eval-data", default=str(calcx.DATA / "test.docs.jsonl"))
    p.add_argument("--sources", nargs="*", default=["gsm8k", "mawps", "ape210k"])
    p.add_argument("--source-weights", default=None,
                   help='JSON {source: share of each step\'s questions}, e.g. {"gsm8k": 0.5, "ape210k": 0.4, '
                        '"mawps": 0.1} (default: uniform over all questions, i.e. by source size)')
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--prompts", type=int, default=1024, help="questions per step (one update)")
    p.add_argument("--chunk-prompts", type=int, default=128, help="questions rolled out per gradient accumulation")
    p.add_argument("--group", type=int, default=8, help="sampled answers per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--no-std", action="store_true", help="advantage r - mean (Dr. GRPO), not divided by the std")
    p.add_argument("--std-floor", type=float, default=0.1, help="the group std divides at least by this")
    p.add_argument("--partial-credit", type=float, default=0.0,
                   help="beta: wrong answers get beta x their share of the reference's calculator steps (0: binary)")
    p.add_argument("--micro-rows", type=int, default=64)
    p.add_argument("--gen-batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-limit", type=int, default=300, help="test questions per source (0: all)")
    p.add_argument("--pretrain-rows", type=int, default=64, help="validation rows for the forgetting measure")
    p.add_argument("--device", default=None, help="default: the init run's")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    init_cfg = cfg_mod.load(a.init_run)
    from model.device import setup_device

    setup_device(a.device or init_cfg.get("device"))
    import torch

    from model.evaluate import load_checkpoint

    out = cfg_mod.run_dir(a.run_name)
    if (out / "grpo_metrics.jsonl").exists():
        print(f"{out} already has a GRPO run (no resume): pick another --run-name", file=sys.stderr)
        return 1
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = dict(init_cfg, run_name=a.run_name)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "grpo.json").write_text(json.dumps(vars(a), indent=2))
    log = open(out / "grpo_metrics.jsonl", "a")

    def write(rec: Dict[str, Any]) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = None
    m = load_checkpoint(a.init_run, int(a.init_checkpoint) if str(a.init_checkpoint).isdigit() else a.init_checkpoint)
    dev = m.token_embedding.embeddings.value.device
    if dev.type == "cuda":
        gen = torch.Generator(device=dev).manual_seed(a.seed)
    tok = calcx.load_tokenizer()
    pad = tok.token_to_id("<eos>")
    params = [v.value for v in m.trainable_weights]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))

    train = calcx.load_questions(a.train_data, a.sources, exclude_choice=True, with_steps=a.partial_credit > 0)
    tests = eval_set(a.eval_data, a.eval_limit, a.seed)
    pre = pretrain_rows(cfg["data"]["dataset"], a.pretrain_rows, int(m.window_size), a.seed)
    print(f"GRPO {a.run_name} from {a.init_run} ({a.init_checkpoint}): {len(train):,} training questions "
          f"({', '.join(a.sources)}), {len(tests)} test questions, {a.prompts} x {a.group} answers per step", flush=True)

    best = -1.0

    def run_eval(step: int) -> None:
        nonlocal best
        t0 = time.time()
        ev = evaluate(m, tok, tests, a)
        ev["pretrain_val_loss"] = pretrain_loss(m, pre, a.micro_rows)
        write({"eval": step, **ev, "seconds": round(time.time() - t0, 1)})
        srcs = " ".join(f"{s} {v:.3f}" for s, v in ev["sources"].items())
        print(f"[eval step {step}] accuracy {ev['accuracy']:.4f} ({srcs}) answered {ev['answered']:.3f} "
              f"pretrain val {ev['pretrain_val_loss']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if step:
            m.save(out / "model.keras")
            if ev["accuracy"] > best:
                shutil.copyfile(out / "model.keras", out / "checkpoints" / "best.keras")
        best = max(best, ev["accuracy"])

    run_eval(0)
    # questions per step by source: each source walks its own shuffled order (a new shuffle at its end)
    shares = json.loads(a.source_weights) if a.source_weights else {"": 1.0}
    pools = {s: [q for q in train if not s or q["source"] == s] for s in shares}
    empty = [s for s, q in pools.items() if not q]
    if empty:
        print(f"--source-weights: no training questions for {empty} (in --sources?)", file=sys.stderr)
        return 1
    counts = {s: int(round(a.prompts * w / sum(shares.values()))) for s, w in shares.items()}
    counts[max(counts, key=counts.get)] += a.prompts - sum(counts.values())
    cursor = {s: [rng.permutation(len(pools[s])), 0] for s in shares}
    print("questions per step: " + ", ".join(f"{s or 'all'} {n} (of {len(pools[s]):,})" for s, n in counts.items()),
          flush=True)

    def draw(s: str, n: int) -> List[Dict[str, str]]:
        out_ = []
        while len(out_) < n:
            order, at = cursor[s]
            if at >= len(order):
                cursor[s] = [rng.permutation(len(pools[s])), 0]
                continue
            take = order[at:at + n - len(out_)]
            cursor[s][1] = at + len(take)
            out_ += [pools[s][i] for i in take]
        return out_

    for step in range(1, a.steps + 1):
        t0 = time.time()
        qs = [q for s, n in counts.items() for q in draw(s, n)]
        qs = [qs[i] for i in rng.permutation(len(qs))]  # sources mixed across the rollout chunks
        opt.zero_grad(set_to_none=True)
        rs, oks, mixed, outs_all, t_gen, loss_sum, n_tok, n_rows = [], [], [], [], 0.0, 0.0, 0, 0
        for c0 in range(0, len(qs), a.chunk_prompts):  # rollout + backward per chunk: gradients accumulate
            cq = qs[c0:c0 + a.chunk_prompts]
            tg = time.time()
            outs = calcx.rollout(m, tok, [q["prompt"] for q in cq for _ in range(a.group)], a.max_new,
                                 a.temperature, a.gen_batch, gen)
            t_gen += time.time() - tg
            per_row = [q for q in cq for _ in range(a.group)]
            ok = np.array([calcx.correct(o["pred"], q["gold"]) for q, o in zip(per_row, outs)]).reshape(len(cq), a.group)
            r = np.array([calcx.reward(o["pred"], o["output"], o["calls"], q, a.partial_credit)
                          for q, o in zip(per_row, outs)], np.float32).reshape(len(cq), a.group)
            mean, std = r.mean(1, keepdims=True), r.std(1, keepdims=True)
            adv = (r - mean) if a.no_std else (r - mean) / np.maximum(std, max(a.std_floor, 1e-4))
            mix = std[:, 0] > 1e-6
            batch = [{**outs[g * a.group + k], "adv": float(adv[g, k])} for g in np.flatnonzero(mix)
                     for k in range(a.group) if any(outs[g * a.group + k]["sampled"])]
            if batch:
                st = accumulate(m, batch, a, pad)
                loss_sum += st["loss_sum"]
                n_tok += st["tokens"]
                n_rows += len(batch)
            rs.append(r)
            oks.append(ok)
            mixed.append(mix)
            outs_all += [{**{k: o[k] for k in ("entropy", "ids", "n_prompt", "calls", "pred")}, "source": q["source"]}
                         for q, o in zip(per_row, outs)]
        gnorm = apply_update(opt, params, n_tok, a.clip) if n_tok else 0.0
        sched.step()
        r, ok, mixed, outs = np.concatenate(rs), np.concatenate(oks), np.concatenate(mixed), outs_all
        flat_ok = ok.reshape(-1)
        by_src = {}
        for o, k in zip(outs, flat_ok):
            by_src.setdefault(o["source"], []).append(k)
        ent = [o["entropy"] for o in outs if o["entropy"] is not None]
        wrong = ~flat_ok
        rec = {"step": step, "reward": float(r.mean()), "accuracy": float(flat_ok.mean()),
               "accuracy_by_source": {s: float(np.mean(v)) for s, v in sorted(by_src.items())},
               "partial_share": float((r.reshape(-1)[wrong] > 0).mean()) if wrong.any() else 0.0,
               "mixed_groups": float(mixed.mean()), "solved_groups": float(ok.all(1).mean()), "rows": n_rows,
               "loss": loss_sum / max(n_tok, 1), "grad_norm": gnorm, "tokens": n_tok,
               "entropy": float(np.mean(ent)) if ent else None,
               "gen_tokens": float(np.mean([len(o["ids"]) - o["n_prompt"] for o in outs])),
               "calls": float(np.mean([o["calls"] for o in outs])),
               "answered": float(np.mean([o["pred"] is not None for o in outs])),
               "lr": sched.get_last_lr()[0], "gen_s": round(t_gen, 1), "step_s": round(time.time() - t0, 1)}
        write(rec)
        srcs = " ".join(f"{s} {v:.3f}" for s, v in rec["accuracy_by_source"].items())
        print(f"step {step}: reward {rec['reward']:.3f} acc {rec['accuracy']:.3f} ({srcs}) partial "
              f"{rec['partial_share']:.2f} mixed {rec['mixed_groups']:.2f} solved "
              f"{rec['solved_groups']:.2f} rows {rec['rows']} loss {rec['loss']:+.4f} |g| {rec['grad_norm']:.3f} "
              f"entropy {rec['entropy'] or 0:.3f} len {rec['gen_tokens']:.0f} ({rec['gen_s']:.0f} + "
              f"{rec['step_s'] - rec['gen_s']:.0f} s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
