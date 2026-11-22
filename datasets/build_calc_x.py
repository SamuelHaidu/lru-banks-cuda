"""Builds `calc-x-bpe32768`: the Calc-X collection (Kadlcik et al., EMNLP 2023, arXiv 2305.15017; MU-NLPC/Calc-*,
the "default" configs: leak-free splits across the collection) as calculator conversations in the agent format,
tokenized with the 32k BPE. Nothing is mixed or weighted here: one category per source dataset.

    uv run --with pyarrow python datasets/build_calc_x.py [--download] [--threads 3] [--samples 2]

Each row becomes (the format of `build_agent_shell.py`, its `calculator` and `render_call`):

    <bos><tools>
    calculator(expression: str): evaluate an arithmetic expression
    </tools><user>question</user><assistant>text before the call
    <call>calculator(expression="7*8")</call></assistant><result>56</result><assistant>text ... The answer is 26.</assistant><eos>

- Every `<gadget id="calculator">expr</gadget><output>...</output>` becomes a call; the output is recomputed by our
  calculator (what the model will get at inference), and the row is dropped if ours disagrees with Calc-X's value
  (sympy output such as "1/10 = around 0.1") or cannot evaluate the expression.
- Digit separators (Ape210K's 3_000) are removed from expressions and answers; numeric answers are printed as the
  calculator prints them (5/3 -> 1.666667).
- The final `<result>` becomes "The answer is X." (multiple choice, AQuA-RAT and MathQA: the letter and its option, "The answer is E) 23.").
- Rows without calls are kept (`{split}.calls.npy` holds the number of calls per document, to filter later).

Output in `--out` (default datasets/calc-x-bpe32768): `{split}.npy` (uint16 ids), `{split}.categories.npy` (start,
length, category), `{split}.calls.npy` (int16), `{split}.docs.jsonl` (id, source, calls, text), `metadata.json`
(`loss_spans` = the assistant spans; per category documents / tokens / assistant tokens / drops).
Raw parquet in `datasets/raw/calc-x/<source>/<split>.parquet` (`--download`, revisions in `revisions.json`).
"""

import argparse
import ast
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_agent_shell import calculator, render_call  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["gsm8k", "ape210k", "math_qa", "aqua_rat", "mawps", "asdiv_a", "svamp"]
SPLITS = ["train", "validation", "test"]
HEADER = "<tools>\ncalculator(expression: str): evaluate an arithmetic expression\n</tools>"
GADGET = re.compile(r'<gadget id="calculator">(.*?)</gadget>\s*<output>(.*?)</output>', re.S)
RESULT = re.compile(r"<result>(.*?)</result>", re.S)
DIGIT_SEP = re.compile(r"(?<=\d)_(?=\d)")  # Ape210K writes 3_000


def download(raw: Path) -> None:
    revs = {}
    for s in SOURCES:
        repo = f"MU-NLPC/Calc-{s}"
        info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}"))
        files = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet"))["default"]
        revs[s] = {"repo": repo, "sha": info.get("sha"), "license": (info.get("cardData") or {}).get("license")}
        for split, urls in files.items():
            assert len(urls) == 1, (s, split, urls)
            dst = raw / s / f"{split}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(urls[0], dst)
            print(f"  {repo} {split}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    (raw / "revisions.json").write_text(json.dumps(revs, indent=2))


def value(text: str):
    """A number from our calculator's output or Calc-X's ("1/10 = around 0.1" -> 0.1, "56" -> 56), else None."""
    text = text.strip()
    if " = around " in text:
        text = text.split(" = around ")[0]
    try:
        return float(calculator(text))
    except ValueError:
        return None


def close(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def convert(row: dict, source: str):
    """(text, calls) or (None, reason)."""
    chain = row["chain"]
    parts, pos, calls = [HEADER, f"<user>{row['question'].strip()}</user>"], 0, 0
    lead = ""
    for m in GADGET.finditer(chain):
        expr = DIGIT_SEP.sub("", m.group(1).strip())
        ours = calculator(expr)
        if ours.startswith("Error"):
            return None, "calculator error"
        if not close(value(ours), value(m.group(2))):
            return None, "output mismatch"
        lead = chain[pos:m.start()].strip()
        parts.append(f"<assistant>{lead + chr(10) if lead else ''}<call>{render_call('calculator', {'expression': expr})}"
                     f"</call></assistant><result>{ours}</result>")
        pos, calls = m.end(), calls + 1
    r = RESULT.search(chain, pos)
    if r is None:
        return None, "no result"
    tail = chain[pos:r.start()].strip()
    ans = DIGIT_SEP.sub("", row["result"].strip())
    opts = row.get("options") or {}
    if isinstance(opts, str):
        opts = ast.literal_eval(opts)
    if opts:  # AQuA-RAT, MathQA: multiple choice
        if ans not in opts:
            return None, "answer not an option"
        ans = f"{ans}) {opts[ans].strip()}"
    elif value(ans) is not None:
        ans = calculator(ans)  # numbers as the calculator prints them (fractions such as 5/3 -> 1.666667)
    final = f"The answer is {ans}."
    parts.append(f"<assistant>{tail + chr(10) if tail else ''}{final}</assistant>")
    return "".join(parts), calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "datasets" / "raw" / "calc-x"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--tokenizer", default="pretrain-mix-bpe32768")
    p.add_argument("--name", default="calc-x-bpe32768")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=3, help="tokenizer threads")
    p.add_argument("--samples", type=int, default=0, help="print N converted documents per source and exit")
    a = p.parse_args()
    raw = Path(a.raw)
    if a.download:
        download(raw)
    import pyarrow.parquet as pq

    if a.samples:
        for s in SOURCES:
            f = next(raw / s / f"{sp}.parquet" for sp in ("test", "validation", "train") if (raw / s / f"{sp}.parquet").exists())
            rows = pq.read_table(f).to_pylist()[:a.samples]
            for row in rows:
                print(f"===== {s} {row['id']}\n{convert(row, s)[0]}\n")
        return
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizers" / a.tokenizer / "tokenizer.json"))
    ids_of = {t: tok.token_to_id(t) for t in ("<bos>", "<eos>", "<assistant>", "</assistant>")}
    bos, eos, a_open, a_close = ids_of.values()
    out = Path(a.out) if a.out else ROOT / "datasets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    revs = json.loads((raw / "revisions.json").read_text())
    cat_ids = {s: i for i, s in enumerate(SOURCES)}
    cats = {s: {} for s in SOURCES}
    meta = {"name": a.name, "tokenizer": a.tokenizer, "vocabulary_size": tok.get_vocab_size(), "doc_token_id": bos,
            "bos_token_id": bos, "eos_token_id": eos, "dtype": "uint16", "loss_spans": [[a_open, a_close]],
            "category_index": "{split}.categories.npy", "category_ids": cat_ids}
    t0, span_chars, span_tokens = time.time(), 0, 0
    for sp in SPLITS:
        docs = []  # (source, id, calls, text)
        for s in SOURCES:
            f = raw / s / f"{sp}.parquet"
            if not f.exists():
                continue
            rows = pq.read_table(f).to_pylist()
            drops = {}
            for row in rows:
                text, calls = convert(row, s)
                if text is None:
                    drops[calls] = drops.get(calls, 0) + 1
                else:
                    docs.append((s, row["id"], calls, text))
            cats[s][sp] = {"rows": len(rows), "dropped": drops}
        if not docs:
            continue
        encs = tok.encode_batch(["<bos>" + d[3] + "<eos>" for d in docs], add_special_tokens=False)
        lens = np.array([len(e.ids) for e in encs], np.int64)
        tokens = np.concatenate([np.asarray(e.ids, np.uint32) for e in encs])
        assert tokens.max() < 65536
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        np.save(out / f"{sp}.npy", tokens.astype(np.uint16))
        np.save(out / f"{sp}.categories.npy", np.stack([starts, lens, [cat_ids[d[0]] for d in docs]], 1))
        np.save(out / f"{sp}.calls.npy", np.array([d[2] for d in docs], np.int16))
        with open(out / f"{sp}.docs.jsonl", "w") as fh:
            for d in docs:
                fh.write(json.dumps({"id": d[1], "source": d[0], "calls": d[2], "text": d[3]}, ensure_ascii=False) + "\n")
        for s in SOURCES:
            sel = [i for i, d in enumerate(docs) if d[0] == s]
            if not sel:
                continue
            scored = 0
            for i in sel:
                inside = False
                for t in encs[i].ids:
                    if inside:
                        scored += 1
                    inside = (inside or t == a_open) and t != a_close
            cats[s][sp].update(documents=len(sel), tokens=int(lens[sel].sum()), assistant_tokens=scored)
        # bpc: characters per token inside the assistant spans (what the loss scores), validation (or train) split
        if sp == "validation":
            sc, st = 0, 0
            for e in encs:
                inside = False
                for t, (o0, o1) in zip(e.ids, e.offsets):
                    if inside:
                        st += 1
                        sc += 0 if t == a_close else o1 - o0
                    inside = (inside or t == a_open) and t != a_close
            span_chars, span_tokens = sc, st
        # checks: boundaries, tags are single tokens, every document decodes back to its text
        assert all(tokens[s0] == bos and tokens[s0 + n - 1] == eos for s0, n in zip(starts, lens))
        bad = sum(tok.decode(e.ids[1:-1], skip_special_tokens=False) != d[3] for e, d in zip(encs, docs))
        assert bad == 0, f"{sp}: {bad} documents do not decode back"
        meta[sp] = {"file": f"{sp}.npy", "tokens": int(len(tokens)), "documents": len(docs),
                    "doc_tokens": {k: int(np.percentile(lens, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}
                    | {"max": int(lens.max())}}
        print(f"{sp}: {len(docs):,} documents, {len(tokens):,} tokens, {time.time() - t0:.0f} s", flush=True)
    meta["avg_chars_per_token"] = round(span_chars / span_tokens, 4)
    meta["categories"] = cats
    meta["source"] = {s: revs[s] for s in SOURCES}
    meta["notes"] = ("Calc-X default (leak-free) splits as calculator conversations (datasets/build_calc_x.py); "
                     "outputs recomputed by build_agent_shell.calculator; loss on the assistant spans; "
                     "avg_chars_per_token over the assistant spans. Not mixed or weighted.")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{'source':10s} {'split':10s} {'rows':>8s} {'docs':>8s} {'tokens':>11s} {'assistant':>10s}  dropped")
    for s in SOURCES:
        for sp, r in cats[s].items():
            print(f"{s:10s} {sp:10s} {r['rows']:8,d} {r.get('documents', 0):8,d} {r.get('tokens', 0):11,d} "
                  f"{r.get('assistant_tokens', 0):10,d}  {r['dropped']}")
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
