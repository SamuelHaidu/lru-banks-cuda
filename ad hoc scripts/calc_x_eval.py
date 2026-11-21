"""Answer accuracy on Calc-X's test splits (`datasets/build_calc_x.py` output, `test.docs.jsonl`): each question is
asked as in training (the calculator tool listed, `<user>question</user><assistant>`), the model generates greedily,
every `<call>calculator(expression="...")</call>` is executed (`build_agent_shell.calculator`) and its result fed back
(`<result>...</result><assistant>`), until the model answers without a call (or `--max-new` tokens). Correct: the
last "The answer is X" equals the reference (numbers within 1e-4 relative; multiple choice: the letter).

The agent loop, the answer check and the question loading are `model/calcx.py` (shared with the RL trainer
`model/grpo.py`): batched, left-padded with `<bos>`, finished rows dropped from the batch.

    uv run python "ad hoc scripts/calc_x_eval.py" --run-name X [--checkpoint latest] [--limit 200] [--batch 128]
        [--sources svamp mawps ...] [--show 3]

Writes runs/<run>/calcx_eval_<checkpoint>.json (per source accuracy, format stats, every answer).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--data", default=str(ROOT / "datasets" / "calc-x-bpe32768" / "test.docs.jsonl"))
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0, help="questions per source (0: all)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--device", default="xpu")
    p.add_argument("--show", type=int, default=2, help="answers printed per source (wrong ones first)")
    p.add_argument("--retry", default=None, help="a previous calcx_eval json: only its wrong questions, sampled")
    p.add_argument("--samples", type=int, default=16, help="with --retry: samples per question")
    p.add_argument("--temperature", type=float, default=0.8, help="with --retry")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from model.device import setup_device

    setup_device(a.device)
    import torch

    from model.calcx import correct, load_questions, load_tokenizer, rollout
    from model.evaluate import load_checkpoint

    m = load_checkpoint(a.run_name, int(a.checkpoint) if a.checkpoint.isdigit() else a.checkpoint)
    tok = load_tokenizer()
    items = load_questions(a.data, a.sources, a.limit)
    if a.retry:
        wrong = {r["id"] for r in json.load(open(a.retry))["answers"] if not r["ok"]}
        items = [it for it in items if it["id"] in wrong]
        items = [it for it in items for _ in range(a.samples)]
        torch.manual_seed(a.seed)
    per = {}
    for it in items:
        per[it["source"]] = per.get(it["source"], 0) + 1
    print(f"{a.run_name} ({a.checkpoint}): {len(items)} questions " + str(per), flush=True)

    results, t0 = [], time.time()
    for b0 in range(0, len(items), a.batch):
        chunk = items[b0:b0 + a.batch]
        outs = rollout(m, tok, [it["prompt"] for it in chunk], a.max_new, a.temperature if a.retry else 0.0, a.batch)
        for it, o in zip(chunk, outs):
            results.append({"source": it["source"], "id": it["id"], "gold": it["gold"], "pred": o["pred"],
                            "ok": correct(o["pred"], it["gold"]), "calls": o["calls"], "bad_call": o["bad_call"],
                            "finished": o["finished"], "question": it["prompt"], "output": o["output"]})
        acc = np.mean([r["ok"] for r in results])
        print(f"  {len(results)}/{len(items)}: accuracy so far {acc:.3f}, {time.time() - t0:.0f} s", flush=True)

    if a.retry:
        by = {}
        for r in results:
            by.setdefault((r["source"], r["id"]), []).append(r["ok"])
        print(f"\nSampled {a.samples}x at temperature {a.temperature} on the questions greedy got wrong")
        print(f"{'source':10s} {'questions':>9s} {'pass@' + str(a.samples):>8s} {'per-sample':>10s} {'any 1+':>7s}  (share of questions with k correct samples: 0 / 1-2 / 3-8 / 9+)")
        for src in sorted({k[0] for k in by}):
            v = [np.array(x) for k, x in by.items() if k[0] == src]
            n = np.array([x.sum() for x in v])
            print(f"{src:10s} {len(v):9d} {np.mean(n > 0):8.3f} {np.mean([x.mean() for x in v]):10.3f} {int((n > 0).sum()):7d}  "
                  f"{np.mean(n == 0):.2f} / {np.mean((n >= 1) & (n <= 2)):.2f} / {np.mean((n >= 3) & (n <= 8)):.2f} / {np.mean(n > 8):.2f}")
        allv = [np.array(x) for x in by.values()]
        print(f"{'all':10s} {len(allv):9d} {np.mean([x.any() for x in allv]):8.3f} {np.mean([x.mean() for x in allv]):10.3f}")
        out = ROOT / "runs" / a.run_name / f"calcx_retry_{a.checkpoint}.json"
        out.write_text(json.dumps({"checkpoint": a.checkpoint, "temperature": a.temperature, "samples": a.samples,
                                   "answers": results}, indent=1))
        print(f"-> {out}")
        return
    report = {}
    print(f"\n{'source':10s} {'n':>5s} {'accuracy':>9s} {'answered':>9s} {'calls/q':>8s} {'bad call':>9s}")
    for s in sorted({r["source"] for r in results}):
        rs = [r for r in results if r["source"] == s]
        report[s] = {"n": len(rs), "accuracy": float(np.mean([r["ok"] for r in rs])),
                     "answered": float(np.mean([r["pred"] is not None for r in rs])),
                     "calls_per_question": float(np.mean([r["calls"] for r in rs])),
                     "bad_call": float(np.mean([r["bad_call"] for r in rs]))}
        q = report[s]
        print(f"{s:10s} {q['n']:5d} {q['accuracy']:9.3f} {q['answered']:9.3f} {q['calls_per_question']:8.2f} "
              f"{q['bad_call']:9.3f}")
    allr = float(np.mean([r["ok"] for r in results]))
    print(f"{'all':10s} {len(results):5d} {allr:9.3f}   ({time.time() - t0:.0f} s)")
    for s in sorted(report):
        rs = sorted([r for r in results if r["source"] == s], key=lambda r: r["ok"])[:a.show]
        for r in rs:
            q = r["question"].split("<user>", 1)[1].split("</user>")[0]
            print(f"\n[{s} {'OK' if r['ok'] else 'WRONG'}] {q[:200]}\n  gold {r['gold']!r}  ->  {r['output'][:300]!r}")
    out = ROOT / "runs" / a.run_name / f"calcx_eval_{a.checkpoint}.json"
    out.write_text(json.dumps({"checkpoint": a.checkpoint, "accuracy": allr, "sources": report,
                               "answers": results}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
