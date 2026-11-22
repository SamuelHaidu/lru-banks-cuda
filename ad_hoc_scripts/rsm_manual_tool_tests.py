"""Manual tool-use prompts for an agent-mix run: greedy, stops at </assistant>, prints each prompt's continuation.

    uv run python ad_hoc_scripts/rsm_manual_tool_tests.py [--run-name rsm-4L-32banks-mix-bpe1024] [--checkpoint latest]
"""

import argparse

from model.device import setup_backend

setup_backend()

from model import config as cfg_mod  # noqa: E402
import model.archs  # noqa: E402,F401
from model.evaluate import load_checkpoint  # noqa: E402

SHELL_TOOLS = ("run(command: str): run a shell command\n"
               "read_file(path: str): return the contents of a file\n"
               "python(code: str): run a Python program\n")


def tools(lines: str) -> str:
    return f"<bos><tools>\n{lines}</tools>"


TESTS = [
    ("arithmetic, new item and numbers (python available)",
     tools(SHELL_TOOLS) + "<user>Apples cost $3.47. How much do 12 cost?</user><assistant>"),
    ("arithmetic, final answer after the result",
     tools(SHELL_TOOLS) + "<user>Apples cost $3.47. How much do 12 cost?</user><assistant>"
     "<call>python(code=\"print(12 * 3.47)\")</call></assistant><result>41.64</result><assistant>"),
    ("arithmetic, only calculator listed",
     tools("calculator(expression: str): evaluate an arithmetic expression\n"
           "send_email(to: str, subject: str, body: str): send an email\n")
     + "<user>A ticket costs $18.25. What do 7 tickets cost?</user><assistant>"),
    ("write code: function + test",
     tools(SHELL_TOOLS) + "<user>Write a Python function that returns the number of vowels in a string and test it "
     "on 'banana'.</user><assistant>"),
    ("recall from a pasted list (new names)",
     tools(SHELL_TOOLS) + "<user>Here are the locker codes:\nBrisko: QT-8814\nVamel: RN-2290\nOltra: KD-5531\n"
     "What is Vamel's code?</user><assistant>"),
    ("unseen tool: invented name and arguments",
     tools("warehouse_locate(item_id: str): return the aisle and shelf of an item in the warehouse\n"
           "read_file(path: str): return the contents of a file\n")
     + "<user>Where is item A-7781 stored?</user><assistant>"),
    ("unseen tool: translation",
     tools("translate_text(text: str, target_lang: str): translate text into another language\n"
           "python(code: str): run a Python program\n")
     + "<user>Translate 'good morning, see you tomorrow' into French.</user><assistant>"),
    ("unseen tool picked from a longer list",
     tools("read_file(path: str): return the contents of a file\n"
           "convert_units(value: float, from_unit: str, to_unit: str): convert a measurement between units\n"
           "send_email(to: str, subject: str, body: str): send an email\n"
           "search_web(query: str): search the internet\n")
     + "<user>How many kilometers is 26.2 miles?</user><assistant>"),
    ("refusal: no listed tool fits",
     tools("read_file(path: str): return the contents of a file\n"
           "calculator(expression: str): evaluate an arithmetic expression\n")
     + "<user>Book me a flight to Lisbon next Friday.</user><assistant>"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="rsm-4L-32banks-mix-bpe1024")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--max-new-tokens", type=int, default=160)
    a = p.parse_args()
    config = cfg_mod.load(a.run_name)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(cfg_mod.tokenizer_path(config["data"]["dataset"])))
    m = load_checkpoint(a.run_name, a.checkpoint)
    for name, prompt in TESTS:
        out = m.generate(prompt, tokenizer=tok, max_new_tokens=a.max_new_tokens, stop_tokens=["</assistant>", "<eos>"])
        user = prompt[prompt.rindex("<user>"):]
        print(f"=== {name}\n{user}" + out[len(prompt):] if out.startswith(prompt) else f"=== {name}\n{out}")
        print()


if __name__ == "__main__":
    main()
