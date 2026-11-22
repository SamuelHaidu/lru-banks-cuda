"""Colored streaming output for agent-format text (`<system>`, `<tools>`, `<user>`, `<assistant>`, `<think>`, `<call>`,
`<result>`), printed with rich. Text is colored by the innermost open tag; the tags themselves are bold."""

import re
from typing import List

from rich.console import Console
from rich.text import Text

STYLES = {
    "system": "cyan",
    "tools": "magenta",
    "user": "green",
    "assistant": "white",
    "think": "italic yellow",
    "call": "bold bright_red",
    "result": "blue",
}
_TAG = re.compile(r"(<(?:/?(?:%s))>|<bos>|<eos>)" % "|".join(STYLES))
_BLOCK = {"system", "tools", "user", "assistant", "result", "think"}  # a line break after these tags


class AgentPrinter:
    """Feed text chunks as they stream; a tag is always inside one chunk (special tokens are atomic)."""

    def __init__(self, console: Console = None):
        self.console = console or Console(highlight=False)
        self.stack: List[str] = []
        self.at_line_start = True
        self.after_tag = False  # the last output was the line break after a tag

    def _out(self, s: str, style: str) -> None:
        if s:
            self.console.print(Text(s, style=style), end="", soft_wrap=True)
            self.at_line_start = s.endswith("\n")

    def feed(self, chunk: str) -> None:
        for part in _TAG.split(chunk):
            if not part:
                continue
            if _TAG.fullmatch(part):
                name = part.strip("<>/")
                if part in ("<bos>", "<eos>"):
                    self._out(part + "\n", "bold dim")
                    continue
                closing = part.startswith("</")
                if not closing:
                    if not self.at_line_start and name in _BLOCK:
                        self._out("\n", "")
                    self.stack.append(name)
                style = "bold reverse " + STYLES[name].replace("italic ", "").replace("bold ", "")
                self._out(part, style)
                if closing:
                    if name in self.stack:
                        while self.stack and self.stack.pop() != name:
                            pass
                if name in _BLOCK:
                    self._out("\n", "")
                self.after_tag = name in _BLOCK
            else:
                if self.after_tag:
                    part = part.removeprefix("\n")  # the tag already broke the line
                self.after_tag = False
                self._out(part, STYLES[self.stack[-1]] if self.stack else "")

    def finish(self) -> None:
        if not self.at_line_start:
            self._out("\n", "")
