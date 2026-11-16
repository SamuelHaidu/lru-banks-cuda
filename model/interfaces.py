"""The single architecture contract.

Everything downstream -- training, evaluation, generation, the dataset -- is
written against `BaseLM` and nothing else. That is the whole point: in the
previous codebase each architecture grew its own `generate()`, its own
parameter accounting and its own quirks, so a change that worked for one
model silently broke another. Here, an architecture supplies four things and
inherits the rest.

To add an architecture:
  1. Write `model/archs/<name>.py` with a `BaseLM` subclass.
  2. Add one line to `ARCHS` in `model/archs/__init__.py`.

`tests/test_interface.py` then covers it automatically -- it iterates `ARCHS`
and asserts the full contract for every registered architecture.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402  (must follow setup_backend -- keras reads KERAS_BACKEND at import)
import numpy as np  # noqa: E402


class BaseLM(keras.Model):
    """Fixed-context, single-next-token language model.

    Every architecture in this repo predicts strictly the *one* token that
    follows a fixed-length causal context window. There is no
    `return_sequences=True` dense-loss regime and no carried recurrent state
    between calls, which is what lets a single `generate()` implementation
    serve every architecture correctly.

    Subclasses must implement:
        `context_length`, `vocabulary_size`, `call`, `get_config`.

    Subclasses inherit:
        `generate`, `parameter_breakdown`.
    """

    # -- required of every architecture --------------------------------------

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Number of context tokens the model reads per sample.

        The dataset builds windows of exactly this width, and `generate()`
        feeds exactly this many trailing ids per step. Being a property (not a
        constructor arg) means an architecture that derives its context from
        several parameters still reports one authoritative number.
        """

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Size of the token vocabulary, i.e. the width of the logits."""

    @abstractmethod
    def call(self, inputs: Any, training: bool = False) -> Any:
        """`(batch, context_length)` int32 token ids -> `(batch, vocabulary_size)` logits."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Serializable constructor kwargs; must round-trip through `from_config`.

        Subclasses merge their own parameters over this base dict:
        ``return {**super().get_config(), **my_params}``. The body here
        delegates to `keras.Model` (name/trainable/dtype) so that chained
        `super()` call has something to merge -- an abstract method whose body
        is only a docstring would return ``None`` and break saving.
        """
        return super().get_config()

    # -- provided once, identical for every architecture ---------------------

    def generate(
        self,
        prompt: Union[str, List[int], np.ndarray],
        tokenizer: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        on_token: Any = None,
    ) -> str:
        """Autoregressive generation. See `model.generation.generate`."""
        from model.generation import generate as _generate

        return _generate(
            self,
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
            on_token=on_token,
        )

    def with_context_length(self, context_length: int) -> "BaseLM":
        """A copy of this model that reads `context_length` tokens, same weights.

        Used for context extrapolation (`--context-length`) and for fine-tuning
        at a longer window (`init_from`). Only architectures whose weights do not
        depend on the window opt in (see `model/archs/_context.py`); the rest
        raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot change its context length "
            f"(its weights are tied to window {self.context_length})."
        )

    def parameter_breakdown(self) -> Dict[str, int]:
        """Splits parameters into embedding tables vs. everything else.

        A variable counts as an embedding table when its first axis is the
        vocabulary size. This makes "mechanism" parameters -- the part that
        actually differs between architectures -- directly comparable without
        each architecture having to maintain its own bookkeeping.
        """
        embeddings = 0
        mechanism = 0
        for var in self.weights:
            count = int(np.prod(var.shape)) if len(var.shape) else 1
            if len(var.shape) >= 1 and var.shape[0] == self.vocabulary_size:
                embeddings += count
            else:
                mechanism += count
        return {
            "total": embeddings + mechanism,
            "embeddings": embeddings,
            "mechanism": mechanism,
        }
