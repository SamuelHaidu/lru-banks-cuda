"""Ad hoc: 5 sampled generations from `gpu-retriver-tokenid` at temperature 0.5.

Prompts are explicitly prefixed with `<bos>` (id 0) and generation stops as
soon as `<eos>` (id 1) is sampled, using `model.generation.generate`'s
`stop_tokens` -- the `model generate` CLI doesn't expose either behavior, so
this calls the library function directly.

Usage:
    .venv/bin/python "ad hoc scripts/gpu_retriver_tokenid_generate_sample.py"
"""

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()

from model import config as cfg_mod  # noqa: E402
from model.evaluate import load_checkpoint  # noqa: E402
from model.generation import generate  # noqa: E402

RUN_NAME = "gpu-retriver-tokenid"
CHECKPOINT = "latest"
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 1024  # safety cap only -- generation stops at <eos> naturally

PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog went to the park to play.",
    "One day, a boy found a strange box in the forest.",
    "The sun was shining and the birds were singing in the",
    "Mom said it was time to clean up the toys before",
]


def main() -> int:
    config = cfg_mod.load(RUN_NAME)
    setup_device(config.get("device"))

    model = load_checkpoint(RUN_NAME, CHECKPOINT)

    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    np.random.seed(0)

    for i, prompt in enumerate(PROMPTS, 1):
        bos_prompt = "<bos>" + prompt
        text = generate(
            model,
            bos_prompt,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            stop_tokens=["<eos>"],
        )
        print(f"=== prompt {i} ===")
        print(text)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
