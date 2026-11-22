"""Autoregressive generation, written against `BaseLM` only.

One implementation serves every architecture. Each step re-runs the model's
own `call()` over the last `model.context_length` ids (left zero-padded near
the start), so the training forward pass is the single source of truth for
the model's behaviour -- there is no separate, drift-prone inference path.
The one exception is recurrent archs (`stateful_generation`, e.g. `wave-arrow`):
they carry their state with `model.advance`, so the context is the whole text;
their scan is the training forward pass too.

The default is greedy (`temperature=0.0`, 256 tokens), which is the
project-standard evaluation setting: sampling knobs are for exploration and
must be requested explicitly, never applied silently.
"""

from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import numpy as np

from model.device import setup_backend

setup_backend()

from keras import ops  # noqa: E402


def _encode(prompt: Union[str, Sequence[int], np.ndarray], tokenizer: Any) -> List[int]:
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("A tokenizer is required to generate from a string prompt.")
        ids = list(tokenizer.encode(prompt).ids)
    else:
        ids = [int(i) for i in prompt]
    if not ids:
        raise ValueError("`prompt` must contain at least one token.")
    return ids


def _apply_repetition_penalty(logits: np.ndarray, seen: Sequence[int], penalty: float) -> None:
    if penalty == 1.0:
        return
    for token_id in set(seen):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty


def _filter_top_k(logits: np.ndarray, top_k: int) -> None:
    if top_k <= 0 or top_k >= logits.size:
        return
    keep = np.argpartition(logits, -top_k)[-top_k:]
    mask = np.ones_like(logits, dtype=bool)
    mask[keep] = False
    logits[mask] = -np.inf


def _filter_top_p(logits: np.ndarray, top_p: float) -> None:
    if not 0.0 < top_p < 1.0:
        return
    order = np.argsort(logits)[::-1]
    probs = np.exp(logits[order] - logits[order].max())
    probs /= probs.sum()
    cutoff = np.cumsum(probs) > top_p
    # Shift by one so the token that crosses the threshold is itself kept --
    # otherwise a single token with probability > top_p would leave nothing.
    cutoff[1:] = cutoff[:-1]
    cutoff[0] = False
    logits[order[cutoff]] = -np.inf


class Session:
    """A recurrent model's state over the ids fed into it so far (`stateful_generation` archs only), so that a later
    call whose ids extend `ids` feeds only the new ones (a chat turn costs its new text, not the conversation)."""

    def __init__(self) -> None:
        self.ids: List[int] = []
        self.state: Any = None

    def clear(self) -> None:
        self.ids, self.state = [], None


def stream(
    model: Any,
    ids: Sequence[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    stop_ids: Sequence[int] = (),
    session: Optional[Session] = None,
    cancel: Any = None,
) -> Iterator[int]:
    """Yields each new token id as soon as it is chosen (the stop token included), for prompt `ids`.

    `session` (stateful archs): continue from its state when `ids` extend the ids it covers, and leave it covering the
    prompt and every yielded token. `cancel`: anything with `is_set()` (a `threading.Event`), checked before each
    token."""
    import torch

    ids = list(ids)
    context = int(model.context_length)
    stop = set(stop_ids)
    # Recurrent archs (`stateful_generation`) carry their state from token to token: the whole text is the
    # context, O(1) per token. The prompt starts a fresh document, as every training row does.
    stateful = bool(getattr(model, "stateful_generation", False))
    state = None
    feed = list(ids)
    reset = getattr(model, "reset_token", None)
    if stateful and reset is not None and feed[0] != reset:
        feed = [reset] + feed
    fed: List[int] = []
    if stateful and session is not None:
        n = len(session.ids)
        if session.state is not None and 0 < n < len(feed) and feed[:n] == session.ids:
            fed, state, feed = list(session.ids), session.state, feed[n:]
        else:
            session.clear()

    next_id = None
    for _ in range(max_new_tokens):
        if cancel is not None and cancel.is_set():
            break
        if stateful:
            with torch.no_grad():
                out, state = model.advance(ops.convert_to_tensor(np.asarray([feed], dtype=np.int32)), state)
            fed += feed
            if session is not None:
                session.ids, session.state = list(fed), state
            logits = np.asarray(ops.convert_to_numpy(out)[0], dtype=np.float64)
        else:
            window = ids[-context:]
            x = np.zeros((1, context), dtype=np.int32)
            x[0, -len(window) :] = window
            logits = np.asarray(
                ops.convert_to_numpy(model(ops.convert_to_tensor(x, dtype="int32"), training=False))[0],
                dtype=np.float64,
            )
        _apply_repetition_penalty(logits, ids, repetition_penalty)

        if temperature <= 0.0:
            next_id = int(np.argmax(logits))
        else:
            logits /= temperature
            if top_k:
                _filter_top_k(logits, int(top_k))
            if top_p:
                _filter_top_p(logits, float(top_p))
            probs = np.exp(logits - np.nanmax(logits[np.isfinite(logits)]))
            probs[~np.isfinite(probs)] = 0.0
            probs /= probs.sum()
            next_id = int(np.random.choice(probs.size, p=probs))

        ids.append(next_id)
        feed = [next_id]
        yield next_id
        if next_id in stop:
            break
    if stateful and session is not None and next_id is not None and session.ids[-1:] != [next_id]:
        with torch.no_grad():  # the last token too, so the session covers everything that was yielded
            _, state = model.advance(ops.convert_to_tensor(np.asarray([[next_id]], dtype=np.int32)), state)
        session.ids, session.state = fed + [next_id], state


def generate(
    model: Any,
    prompt: Union[str, Sequence[int], np.ndarray],
    tokenizer: Any = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    stop_tokens: Optional[List[str]] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> str:
    """Generates a continuation.

    Args:
        model: Any `BaseLM`; only `context_length` and `call()` are used (`advance` for stateful archs).
        prompt: A string (requires `tokenizer`) or a sequence of token ids.
        max_new_tokens: Number of tokens to append.
        temperature: `0.0` means greedy argmax. Above zero, logits are scaled
            and sampled after optional top-k / top-p filtering.
        stop_tokens: Strings whose final token id ends generation early.
        on_token: Called with each new token id as soon as it is chosen (streaming), the stop token included.

    Returns:
        The decoded prompt + continuation, or a space-joined id list when no
        tokenizer was supplied.
    """
    ids = _encode(prompt, tokenizer)
    stop_ids = set()
    for text in stop_tokens or []:
        encoded = tokenizer.encode(text).ids if tokenizer is not None else []
        if encoded:
            stop_ids.add(encoded[-1])
    for next_id in stream(model, ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                          top_p=top_p, repetition_penalty=repetition_penalty, stop_ids=stop_ids):
        ids.append(next_id)
        if on_token is not None:
            on_token(next_id)

    if tokenizer is None:
        return " ".join(str(i) for i in ids)
    return tokenizer.decode(ids, skip_special_tokens=False)
