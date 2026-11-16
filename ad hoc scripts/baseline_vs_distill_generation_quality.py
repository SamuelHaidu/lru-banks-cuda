"""Ad hoc: 20 free generations each from `baseline` (CE) and
`distill-gpt-to-baseline` (KL-distilled from `simple-gpt-baseline`), at
temperature 0.5, prompt = `<bos>` only, stopped at the first `<eos>` (capped
at 1024 tokens as a safety net against a non-terminating run).

Analyzes, per model, across all 20 samples:
  - vocabulary diversity: distinct-1/2/3 (unique n-grams / total n-grams)
  - bigram/trigram repetition rate (share of n-grams that are repeats)
  - basic grammar-quality proxies: average sentence length, fraction of
    sentences with a plausible subject-verb pattern (very rough, regex-based
    -- not a real parser, just consistent enough to compare the two models),
    and a repeated-immediate-word rate ("happy happy") as a degeneracy signal

Usage:
    .venv/bin/python "ad hoc scripts/baseline_vs_distill_generation_quality.py"
"""

import re
from collections import Counter
from typing import Dict, List

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.generation import generate  # noqa: E402

RUNS = {
    "baseline (CE)": {"run_name": "baseline", "checkpoint": 9},
    "distill-gpt-to-baseline (KL)": {"run_name": "distill-gpt-to-baseline", "checkpoint": "latest"},
    "retriver-tokenid (CE)": {"run_name": "retriver-tokenid", "checkpoint": 9},
}

N_SAMPLES = 20
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 1024


def _tokenize_words(text: str) -> List[str]:
    # Strip special tokens, lowercase, split on non-alphanumeric.
    text = re.sub(r"<bos>|<eos>|<pad>|<unk>", " ", text)
    return re.findall(r"[a-z']+", text.lower())


def _ngrams(words: List[str], n: int) -> List[tuple]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _distinct_n(all_words: List[List[str]], n: int) -> float:
    total = 0
    seen = set()
    for words in all_words:
        grams = _ngrams(words, n)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def _repetition_rate(all_words: List[List[str]], n: int) -> float:
    """Share of n-grams, within each sample, that are repeats of an earlier
    n-gram in that same sample -- a same-story degeneracy signal, distinct
    from distinct-n (which is corpus-wide diversity across all 20 samples).
    """
    repeats = 0
    total = 0
    for words in all_words:
        grams = _ngrams(words, n)
        seen = set()
        for g in grams:
            total += 1
            if g in seen:
                repeats += 1
            seen.add(g)
    return repeats / total if total else 0.0


def _immediate_repeat_rate(all_words: List[List[str]]) -> float:
    """Share of adjacent word pairs that are the exact same word twice in a
    row ("happy happy") -- a blunt but clear degeneracy signal.
    """
    repeats = 0
    total = 0
    for words in all_words:
        for i in range(len(words) - 1):
            total += 1
            if words[i] == words[i + 1]:
                repeats += 1
    return repeats / total if total else 0.0


def _sentence_stats(text: str) -> Dict[str, float]:
    text = re.sub(r"<bos>|<eos>|<pad>|<unk>", " ", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return {"avg_sentence_len": 0.0, "plausible_svo_rate": 0.0, "n_sentences": 0}

    lengths = [len(s.split()) for s in sentences]

    # Very rough subject-verb-ish heuristic: sentence contains a pronoun/noun
    # near the start followed later by a common verb form. Not a real parser
    # -- just applied identically to both models so the comparison is fair.
    pronouns = r"\b(she|he|they|it|i|we|you|lily|timmy|tom|mom|dad)\b"
    verbs = r"\b(was|were|is|are|said|saw|went|played|loved|felt|wanted|had|ran|found|asked|told|smiled|laughed)\b"
    plausible = sum(
        1 for s in sentences if re.search(pronouns, s.lower()) and re.search(verbs, s.lower())
    )

    return {
        "avg_sentence_len": sum(lengths) / len(lengths),
        "plausible_svo_rate": plausible / len(sentences),
        "n_sentences": len(sentences),
    }


def main() -> int:
    for label, cfg in RUNS.items():
        run_name = cfg["run_name"]
        checkpoint = cfg["checkpoint"]

        config = cfg_mod.load(run_name)
        setup_device(config.get("device"))
        model = load_checkpoint(run_name, checkpoint)

        tokenizer = None
        tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
        if tokenizer_file.exists():
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_file))

        texts = []
        for i in range(N_SAMPLES):
            text = generate(
                model,
                "<bos>",
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                stop_tokens=["<eos>"],
            )
            texts.append(text)

        all_words = [_tokenize_words(t) for t in texts]
        total_words = sum(len(w) for w in all_words)

        distinct_1 = _distinct_n(all_words, 1)
        distinct_2 = _distinct_n(all_words, 2)
        distinct_3 = _distinct_n(all_words, 3)
        rep_2 = _repetition_rate(all_words, 2)
        rep_3 = _repetition_rate(all_words, 3)
        imm_rep = _immediate_repeat_rate(all_words)

        sent_stats = [_sentence_stats(t) for t in texts]
        avg_sentence_len = sum(s["avg_sentence_len"] for s in sent_stats) / len(sent_stats)
        avg_svo_rate = sum(s["plausible_svo_rate"] for s in sent_stats) / len(sent_stats)
        avg_n_sentences = sum(s["n_sentences"] for s in sent_stats) / len(sent_stats)
        avg_words_per_story = total_words / N_SAMPLES

        # Most common words, for a quick eyeball of what's being overused.
        word_counts = Counter(w for words in all_words for w in words)
        top_words = word_counts.most_common(10)

        print(f"\n=== {label} ===")
        print(f"  samples: {N_SAMPLES}, avg words/story: {avg_words_per_story:.1f}, avg sentences/story: {avg_n_sentences:.1f}")
        print(f"  distinct-1 (unigram diversity): {distinct_1:.4f}")
        print(f"  distinct-2 (bigram diversity):  {distinct_2:.4f}")
        print(f"  distinct-3 (trigram diversity): {distinct_3:.4f}")
        print(f"  in-story bigram repetition rate:  {rep_2:.4f}")
        print(f"  in-story trigram repetition rate: {rep_3:.4f}")
        print(f"  immediate word-repeat rate (\"happy happy\"): {imm_rep:.4f}")
        print(f"  avg sentence length (words): {avg_sentence_len:.2f}")
        print(f"  plausible subject-verb rate: {avg_svo_rate:.4f}")
        print(f"  top words: {top_words}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
