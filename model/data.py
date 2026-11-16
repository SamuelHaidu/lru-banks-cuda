"""The dataset. One sampling scheme, two modes (train / full eval).

Document-based, variable-prefix, single-next-token sampling:

    context = tokens[max(doc_start, t - window) : t]   left zero-padded to `window`
    target  = tokens[t]                                one token

The *start* of the window varies (it is clipped at the document start, so
short prefixes are left-padded); nothing after the start varies -- the window
always runs contiguously up to the target. Windows never cross a document
boundary.

Memory is O(1): the corpus is memory-mapped, never loaded. The 891 MB
TinyStories `train.npy` costs a few MB of resident memory.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402

from model import config as cfg_mod  # noqa: E402


def load_tokens(path: str) -> np.ndarray:
    """Memory-maps a flat `.npy` token array."""
    tokens = np.load(path, mmap_mode="r")
    return tokens.reshape(-1) if tokens.ndim > 1 else tokens


def build_doc_index(path: str, tokens: np.ndarray, doc_token_id: int, use_cache: bool = True) -> np.ndarray:
    """Positions of every document-start marker, cached at `<path>.doc_idx.npy`.

    Scans in 50M-token chunks so a multi-hundred-million-token corpus never
    materializes. The cache is validated (not just trusted) by checking that
    its first entry really points at `doc_token_id`, so a cache built for a
    different marker is detected and rebuilt.
    """
    cache_path = f"{path}.doc_idx.npy"
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.size and int(tokens[cached[0]]) == doc_token_id:
            return cached

    chunk = 50_000_000
    parts = []
    for begin in range(0, len(tokens), chunk):
        block = np.asarray(tokens[begin : begin + chunk])
        hits = np.flatnonzero(block == doc_token_id)
        if hits.size:
            parts.append(hits.astype(np.int64) + begin)
    starts = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if use_cache:
        try:
            np.save(cache_path, starts)
        except OSError:
            pass
    return starts


def _document_bounds(path: str, tokens: np.ndarray, doc_token_id: int, use_cache: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Returns `(doc_starts, doc_ends)` with `doc_ends` EXCLUSIVE.

    `doc_ends[i]` is one past the last valid token index of document `i`
    (`len(tokens)` for the final document). Every sampler below therefore uses
    `doc_ends` as an exclusive upper bound -- writing `doc_ends + 1` lets the
    final document sample `t == len(tokens)`, which is out of bounds and
    crashed live training in the previous codebase.
    """
    starts = build_doc_index(path, tokens, doc_token_id, use_cache)
    if starts.size == 0:
        raise ValueError(f"No document markers (doc_token_id={doc_token_id}) found in {path}.")
    ends = np.append(starts[1:], len(tokens))
    keep = (ends - starts) >= 2  # need at least one context token plus a target
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        raise ValueError(f"No documents of length >= 2 in {path}.")
    return starts, ends


def loss_target_mask(path: str, tokens: np.ndarray, spans: Optional[Any], doc_token_id: int,
                     use_cache: bool = True) -> Optional[np.ndarray]:
    """Which positions are prediction targets when the dataset restricts the loss to spans (`loss_spans` in its
    metadata, e.g. `[[<assistant>, </assistant>]]`): position p counts if an open tag comes after the last close
    tag before p, so every token inside a span and its close tag count, the open tag and everything outside do
    not (prompts, tool results). A document marker closes any span. None if the dataset has no `loss_spans`.
    Cached at `<path>.loss_<open>_<close>.npy`."""
    if not spans:
        return None
    key = "_".join(f"{int(a)}_{int(b)}" for a, b in spans)
    cache_path = f"{path}.loss_{key}.npy"
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(tokens):
            return cached
    out = loss_mask_of(np.asarray(tokens), spans, doc_token_id)
    if use_cache:
        try:
            np.save(cache_path, out)
        except OSError:
            pass
    return out


def loss_mask_of(tok: np.ndarray, spans: Any, doc_token_id: int) -> np.ndarray:
    """`loss_target_mask` of a token array (no cache); exact on any slice that starts at a document marker."""
    idx = np.arange(len(tok), dtype=np.int32 if len(tok) < 2 ** 31 - 1 else np.int64)
    out = np.zeros(len(tok), dtype=bool)
    for open_id, close_id in spans:
        last_open = np.maximum.accumulate(np.where(tok == int(open_id), idx, -1))
        last_close = np.maximum.accumulate(np.where((tok == int(close_id)) | (tok == doc_token_id), idx, -1))
        # the state strictly before p: inside if the latest open is after the latest close
        out[1:] |= last_open[:-1] > last_close[:-1]
    return out


KEPT_CHUNK = 50_000_000
LAZY_LOSS_MASK_TOKENS = 1_000_000_000  # above this, PackedDocumentDataset computes the loss mask per batch  # tokens per chunk when indexing the kept stream (cut at document starts)


def kept_runs(path: str, tokens: np.ndarray, doc_starts: np.ndarray, spans: Any, read_span: Any, doc_token_id: int,
              use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """The static-memory rows' stream (the corpus without span contents) as runs of consecutive corpus positions:
    `(corpus_start, stream_start)` per run, plus a last entry `(len(tokens), stream length)`. Stream index k is corpus
    position `corpus_start[r] + k - stream_start[r]` for the run r holding it. A few runs per document instead of one
    int64 per token (5.9B tokens: 47 GB). Built in chunks cut at document starts (spans close at a marker), cached at
    `<path>.kept_<spans>.npz`."""
    from model.memory_rows import span_masks

    key = "_".join(str(int(t)) for sp in spans for t in sp)
    cache_path = f"{path}.kept_{key}.npz"
    if use_cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        if int(z["corpus"][-1]) == len(tokens):
            return z["corpus"], z["stream"]
    n = len(tokens)
    cuts = doc_starts[np.searchsorted(doc_starts, np.arange(KEPT_CHUNK, n, KEPT_CHUNK))] if n > KEPT_CHUNK else []
    bounds = np.unique(np.concatenate([[0], np.asarray(cuts, np.int64), [n]]))
    corpus, length = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        kept = span_masks(np.asarray(tokens[a:b]), spans, read_span, doc_token_id)[0] == 0
        edge = np.flatnonzero(np.diff(np.concatenate([[False], kept, [False]]).astype(np.int8)))
        starts, ends = edge[0::2], edge[1::2]  # runs [start, end) of kept tokens in the chunk
        corpus.append(starts.astype(np.int64) + a)
        length.append((ends - starts).astype(np.int64))
    corpus, length = np.concatenate(corpus + [np.array([n], np.int64)]), np.concatenate(length)
    stream = np.concatenate([[0], np.cumsum(length)]).astype(np.int64)
    if use_cache:
        try:
            np.savez(cache_path, corpus=corpus, stream=stream)
        except OSError:
            pass
    return corpus, stream


def _fill_windows(tokens: np.ndarray, target_positions: np.ndarray, doc_starts: np.ndarray, window_size: int):
    """Builds left-zero-padded context windows and their validity mask."""
    n = len(target_positions)
    x = np.zeros((n, window_size), dtype=np.int32)
    valid = np.zeros((n, window_size), dtype=bool)
    n_real = np.minimum(target_positions - doc_starts, window_size)
    for i in range(n):
        r = int(n_real[i])
        if r:
            t = int(target_positions[i])
            x[i, -r:] = tokens[t - r : t]
            valid[i, -r:] = True
    return x, valid


class VariablePrefixDataset(keras.utils.PyDataset):
    """Training sampler: random document, random cut point, one next token.

    Random position corruption (`corrupt_prob`): a non-recurrent model reading
    a flat concatenated window will otherwise lean almost entirely on the last
    few positions, since recency is the strongest single predictor. Each
    *valid* (non-padding) context position is independently replaced by a
    uniform random vocabulary id with probability `corrupt_prob`. The target
    is never corrupted, and corruption is train-only -- applying it at
    validation would change what is being measured.
    """

    def __init__(
        self,
        path: str,
        window_size: int,
        batch_size: int,
        vocabulary_size: int,
        doc_token_id: int,
        steps_per_epoch: int = 1000,
        corrupt_prob: float = 0.0,
        seed: int = 42,
        use_cache: bool = True,
        loss_spans: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        if not 0.0 <= corrupt_prob < 1.0:
            raise ValueError(f"`corrupt_prob` must be in [0.0, 1.0), got {corrupt_prob}")

        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.vocabulary_size = int(vocabulary_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.corrupt_prob = float(corrupt_prob)
        self.rng = np.random.default_rng(seed)

        self.tokens = load_tokens(path)
        self.doc_starts, self.doc_ends = _document_bounds(path, self.tokens, int(doc_token_id), use_cache)
        self.num_docs = len(self.doc_starts)
        mask = loss_target_mask(path, self.tokens, loss_spans, int(doc_token_id), use_cache)
        self.allowed = None if mask is None else np.flatnonzero(mask)  # targets restricted to the loss spans

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.allowed is not None:  # a random allowed target, in its document
            targets = self.allowed[self.rng.integers(0, len(self.allowed), size=self.batch_size)]
            d_starts = self.doc_starts[np.searchsorted(self.doc_starts, targets, side="right") - 1]
        else:
            docs = self.rng.integers(0, self.num_docs, size=self.batch_size)
            d_starts = self.doc_starts[docs]
            # `doc_ends` is exclusive -- see `_document_bounds`.
            targets = self.rng.integers(d_starts + 1, self.doc_ends[docs])

        x, valid = _fill_windows(self.tokens, targets, d_starts, self.window_size)
        y = np.asarray(self.tokens[targets]).astype(np.int32)

        if self.corrupt_prob > 0.0:
            mask = (self.rng.random(x.shape) < self.corrupt_prob) & valid
            if mask.any():
                noise = self.rng.integers(0, self.vocabulary_size, size=x.shape, dtype=np.int32)
                x[mask] = noise[mask]
        return x, y


class DenseWindowDataset(keras.utils.PyDataset):
    """Dense-supervision sampler: `T` consecutive targets per sample, each with a full window.

    Each sample is a random document and a random first target `p0`. The
    input covers absolute tokens `[p0 - T, p0 + T - 2]` (`2T - 1` ids) and the
    targets are `p0 .. p0 + T - 1`, so query `m` (input index `T - 1 + m`)
    sees exactly the window `inputs[m : m + T]` that `VariablePrefixDataset`
    would build for target `p0 + m`: real tokens, left-padded with 0 before
    the document start. Targets past the document end are `-1` (ignored).

    `corrupt_prob` corrupts real input tokens like the single-target sampler;
    targets are never corrupted.
    """

    def __init__(
        self,
        path: str,
        window_size: int,
        batch_size: int,
        vocabulary_size: int,
        doc_token_id: int,
        steps_per_epoch: int = 1000,
        corrupt_prob: float = 0.0,
        seed: int = 42,
        use_cache: bool = True,
        loss_spans: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        if not 0.0 <= corrupt_prob < 1.0:
            raise ValueError(f"`corrupt_prob` must be in [0.0, 1.0), got {corrupt_prob}")
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.vocabulary_size = int(vocabulary_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.corrupt_prob = float(corrupt_prob)
        self.rng = np.random.default_rng(seed)
        self.tokens = load_tokens(path)
        self.doc_starts, self.doc_ends = _document_bounds(path, self.tokens, int(doc_token_id), use_cache)
        self.num_docs = len(self.doc_starts)
        self.loss_mask = loss_target_mask(path, self.tokens, loss_spans, int(doc_token_id), use_cache)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        T = self.window_size
        docs = self.rng.integers(0, self.num_docs, size=self.batch_size)
        starts, ends = self.doc_starts[docs], self.doc_ends[docs]
        # First target in [start + 1, end - T] when the document is long enough
        # to fill all T targets, else start + 1 (fewer valid targets).
        p0 = self.rng.integers(starts + 1, np.maximum(starts + 2, ends - T + 1))

        positions = (p0 - T)[:, None] + np.arange(2 * T - 1)[None, :]
        real = (positions >= starts[:, None]) & (positions < ends[:, None])
        # Cast before masking: token files may be stored unsigned (uint16), where -1 / 0 fills overflow.
        x = np.where(real, np.asarray(self.tokens[np.clip(positions, 0, len(self.tokens) - 1)]).astype(np.int32), 0)

        target_pos = p0[:, None] + np.arange(T)[None, :]
        y = np.where(target_pos < ends[:, None],
                     np.asarray(self.tokens[np.clip(target_pos, 0, len(self.tokens) - 1)]).astype(np.int32), -1)
        if self.loss_mask is not None:  # targets outside the loss spans are ignored
            y[~self.loss_mask[np.clip(target_pos, 0, len(self.tokens) - 1)]] = -1

        if self.corrupt_prob > 0.0:
            mask = (self.rng.random(x.shape) < self.corrupt_prob) & real
            if mask.any():
                noise = self.rng.integers(0, self.vocabulary_size, size=x.shape, dtype=np.int32)
                x[mask] = noise[mask]
        return x, y


class PackedDocumentDataset(keras.utils.PyDataset):
    """Infinite-context sampler: rows of `T` consecutive corpus tokens, a target at every position.

    Each row starts at a random document start and runs on through the following documents (packed, no
    padding); the model restarts its state at every document marker. `x` = tokens `[s, s + T)`, `y` = tokens
    `[s + 1, s + T]`, with `-1` (ignored) where the next token is a document marker (never a target, as in the
    other samplers) or past the corpus end.

    `corrupt_prob` corrupts input tokens like the other samplers, but never a marker and never into one
    (a marker is a state reset). `fixed`: batch i is drawn from its own seed (seed, i), so every pass over the
    dataset sees the same rows (dense validation: epochs scored on the same targets).

    `random_windows`: one epoch is one pass over the corpus instead. The token stream is cut into consecutive
    windows of `T` tokens from a random offset (0..T-1, new every epoch) and the windows are served in a shuffled
    order, so every token is trained once per epoch (all but the < 2 T at the two ends) and rows start at document
    starts, middles and ends alike, each from a zero state. `epoch_passes` (default 1): the fraction of a pass one
    epoch covers (0.25: four epochs per pass; the passes continue across epochs, each pass with its own offset and
    order). `steps_per_epoch` is then the batches of a pass (windows / `batch_size`) times `epoch_passes`, rounded
    down to a multiple of `steps_multiple` (the micro-batches per update).

    `static_memory` `{"spans": [[open, close], ...], "read": [open, close]}` (archs with a static memory, `min-gru`):
    the contents of the system / tool spans leave the rows (their tags stay). Rows are `T` consecutive tokens of the
    rest (random windows, or from document starts), and `x` is a batch dict (`model.memory_rows`) carrying each read
    document's span tokens from the corpus (also for a document cut by the row start) as the memory encoder's input.
    Rows and encoder tokens are corrupted alike.
    """

    def __init__(
        self,
        path: str,
        window_size: int,
        batch_size: int,
        vocabulary_size: int,
        doc_token_id: int,
        steps_per_epoch: int = 1000,
        corrupt_prob: float = 0.0,
        seed: int = 42,
        use_cache: bool = True,
        loss_spans: Optional[Any] = None,
        fixed: bool = False,
        stream_chunks: int = 0,
        random_windows: bool = False,
        steps_multiple: int = 1,
        epoch_passes: float = 1.0,
        static_memory: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        if not 0.0 <= corrupt_prob < 1.0:
            raise ValueError(f"`corrupt_prob` must be in [0.0, 1.0), got {corrupt_prob}")
        self.stream_chunks, self._epoch = int(stream_chunks), 0
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.vocabulary_size = int(vocabulary_size)
        self.doc_token_id = int(doc_token_id)
        self.steps_per_epoch = int(steps_per_epoch)
        self.corrupt_prob = float(corrupt_prob)
        self.seed, self.fixed = int(seed), bool(fixed)
        self.rng = np.random.default_rng(seed)
        self.tokens = load_tokens(path)
        self.doc_starts, _ = _document_bounds(path, self.tokens, self.doc_token_id, use_cache)
        self.num_docs = len(self.doc_starts)
        self.loss_spans = loss_spans
        self.static_memory = None
        self.loss_mask = None
        if static_memory:
            # nothing per token: the rows' stream (the corpus without span contents) is a list of runs, and the read
            # / loss masks and the documents' span tokens are computed per batch from the documents a row touches
            # (a corpus of billions of tokens fits; `loss_mask` is not built)
            if self.stream_chunks:
                raise ValueError("static_memory rows cannot be carried (training.stateful)")
            self._run_corpus, self._run_stream = kept_runs(path, self.tokens, self.doc_starts, static_memory["spans"],
                                                           static_memory["read"], self.doc_token_id, use_cache)
            self._doc_bounds = np.append(self.doc_starts, len(self.tokens))
            self.static_memory = static_memory
        elif len(self.tokens) > LAZY_LOSS_MASK_TOKENS:
            # a per-token mask would need int64 arrays over the whole corpus (5.9B tokens: 47 GB): computed per batch
            # from each row's documents instead (`_row_loss_mask`)
            self._doc_bounds = np.append(self.doc_starts, len(self.tokens))
        else:
            self.loss_mask = loss_target_mask(path, self.tokens, loss_spans, self.doc_token_id, use_cache)
        self.random_windows = bool(random_windows)
        if self.random_windows:
            if self.stream_chunks or self.fixed:
                raise ValueError("random_windows cannot be combined with streams (training.stateful) or fixed rows")
            if not 0.0 < epoch_passes:
                raise ValueError(f"epoch_passes must be positive, got {epoch_passes}")
            windows = (self._stream_length() - self.window_size) // self.window_size  # fit at any offset (T + 1)
            self.steps_per_pass = windows // self.batch_size
            steps = int(self.steps_per_pass * float(epoch_passes))
            self.steps_per_epoch = max(1, steps - steps % max(1, int(steps_multiple)))
            self._windows: Optional[Tuple[int, int, np.ndarray]] = None

    def _stream_length(self) -> int:
        return int(self._run_stream[-1]) if self.static_memory else len(self.tokens)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def on_epoch_end(self) -> None:
        self._epoch += 1

    def _window_order(self, p: int) -> Tuple[int, np.ndarray]:
        """(offset, shuffled window numbers) of pass p: window w covers offset + w T .. + T."""
        if self._windows is None or self._windows[0] != p:
            rng = np.random.default_rng((self.seed, p, 7))
            offset = int(rng.integers(self.window_size))
            count = (self._stream_length() - offset - 1) // self.window_size
            self._windows = (p, offset, rng.permutation(count))
        return self._windows[1], self._windows[2]

    def __getitem__(self, idx: int) -> Tuple[Any, np.ndarray]:
        T, n = self.window_size, len(self.tokens)
        rng = np.random.default_rng((self.seed, idx)) if self.fixed else self.rng
        if self.stream_chunks:
            # streams (truncated BPTT): row b of batch g continues row b of batch g - 1 (the next `T` tokens); every
            # `stream_chunks` batches a row jumps to a new random document start, where the state restarts anyway.
            # The rows' jumps are staggered (row b is `b stream_chunks / batch_size` batches ahead), so an update
            # never restarts every stream at once. A pure function of the global batch number, so repeated reads of
            # a batch agree. (At the very first batch most rows start mid-document from a zero state.)
            g = self._epoch * self.steps_per_epoch + idx
            starts = np.empty(self.batch_size, dtype=np.int64)
            for b in range(self.batch_size):
                run, j = divmod(g + b * self.stream_chunks // self.batch_size, self.stream_chunks)
                starts[b] = self.doc_starts[np.random.default_rng((self.seed, b, run)).integers(self.num_docs)] + j * T
        elif self.random_windows:
            # global batch number -> (pass, batch within the pass): passes run on across epochs
            p, j = divmod(self._epoch * self.steps_per_epoch + idx, self.steps_per_pass)
            offset, order = self._window_order(p)
            starts = offset + order[j * self.batch_size:(j + 1) * self.batch_size].astype(np.int64) * T
        else:
            starts = self.doc_starts[rng.integers(0, self.num_docs, size=self.batch_size)]
            if self.static_memory:  # a document start in the rows' stream (markers are kept)
                starts = self._to_stream(starts)
        if self.static_memory:
            return self._memory_batch(starts, rng)
        positions = starts[:, None] + np.arange(T + 1)[None, :]
        real = positions < n
        # Cast before masking: token files may be stored unsigned (uint16).
        seq = np.where(real, np.asarray(self.tokens[np.clip(positions, 0, n - 1)]).astype(np.int32), self.doc_token_id)
        x, y = seq[:, :-1].copy(), seq[:, 1:].copy()
        y[(y == self.doc_token_id) | ~real[:, 1:]] = -1
        if self.loss_mask is not None:  # targets outside the loss spans (prompts, tool results) are ignored
            y[~self.loss_mask[np.clip(positions[:, 1:], 0, n - 1)]] = -1
        elif self.loss_spans:  # large corpus: the mask from each row's documents
            for b in range(len(starts)):
                at = np.clip(positions[b, 1:], 0, n - 1)
                y[b, ~self._row_loss_mask(at)] = -1
        self._corrupt(x, rng)
        return x, y

    def _row_loss_mask(self, at: np.ndarray) -> np.ndarray:
        """`loss_target_mask` at the increasing corpus positions `at` of one row, from the start of the document
        holding its first position (exact: the spans close at a document marker)."""
        lo = int(self._doc_bounds[np.searchsorted(self._doc_bounds, at[0], side="right") - 1])
        piece = np.asarray(self.tokens[lo:int(at[-1]) + 1])
        return loss_mask_of(piece, self.loss_spans, self.doc_token_id)[at - lo]

    def _corrupt(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """Replaces tokens with random ones (in place) with probability `corrupt_prob`, never a marker or into one."""
        if self.corrupt_prob > 0.0:
            mask = (rng.random(x.shape) < self.corrupt_prob) & (x != self.doc_token_id)
            if mask.any():
                noise = rng.integers(0, self.vocabulary_size - 1, size=x.shape, dtype=np.int32)
                noise += noise >= self.doc_token_id  # skip the marker
                x[mask] = noise[mask]

    def _to_corpus(self, k: np.ndarray) -> np.ndarray:
        """Stream indices (the rows' stream, without span contents) -> corpus positions."""
        r = np.searchsorted(self._run_stream, k, side="right") - 1
        return self._run_corpus[r] + (k - self._run_stream[r])

    def _to_stream(self, p: np.ndarray) -> np.ndarray:
        """Corpus positions of kept tokens -> stream indices."""
        r = np.searchsorted(self._run_corpus, p, side="right") - 1
        return self._run_stream[r] + (p - self._run_corpus[r])

    def _memory_batch(self, starts: np.ndarray, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """`static_memory` rows: `T` tokens of the stream without span contents from each start, and the span tokens
        of every document they read (see the class doc)."""
        from model.memory_rows import pack, span_masks

        T, K, bos = self.window_size, self._stream_length(), self.doc_token_id
        spans, read_span = self.static_memory["spans"], self.static_memory["read"]
        positions = starts[:, None] + np.arange(T + 1)[None, :]
        real = positions < K
        at = self._to_corpus(np.clip(positions, 0, K - 1))  # corpus positions
        seq = np.where(real, np.asarray(self.tokens[at.reshape(-1)]).reshape(at.shape).astype(np.int32), bos)
        y = seq[:, 1:].copy()
        y[(y == bos) | ~real[:, 1:]] = -1
        # the read and loss masks of each row: from its first document's start to its last position
        read = np.zeros(at.shape, bool)
        for b in range(len(starts)):
            lo = int(self._doc_bounds[np.searchsorted(self._doc_bounds, at[b, 0], side="right") - 1])
            piece = np.asarray(self.tokens[lo:int(at[b].max()) + 1])
            read[b] = span_masks(piece, spans, read_span, bos)[2][at[b] - lo]
            if self.loss_spans:
                y[b, ~loss_mask_of(piece, self.loss_spans, bos)[at[b, 1:] - lo]] = -1
        at = at[:, :-1]
        read = read[:, :-1] & real[:, :-1]
        doc = np.searchsorted(self.doc_starts, at, side="right") - 1
        docs = {}
        for g in np.unique(doc[read]):
            toks = np.asarray(self.tokens[self._doc_bounds[g]:self._doc_bounds[g + 1]])
            kind, tag = span_masks(toks, spans, read_span, bos)[:2]
            e = (kind > 0) | tag  # the document's span tags and contents, in order
            if (kind > 0).any():  # a marker, then the document's span tags and contents
                docs[int(g)] = (np.concatenate([[bos], toks[e]]).astype(np.int32),
                                np.concatenate([[0], kind[e]]).astype(np.int8), -1)
        batch = pack([(seq[b, :-1], np.ones(T, bool), read[b], doc[b]) for b in range(len(starts))], docs, bos,
                     len(self.static_memory["spans"]), T=T)
        for key in ("_g_of", "_end", "valid"):
            del batch[key]
        self._corrupt(batch["ids"], rng)
        self._corrupt(batch["enc"], rng)
        return batch, y


class FullEvalDataset(keras.utils.PyDataset):
    """Evaluation sampler: one seeded position per document, every document once.

    `VariablePrefixDataset` draws fresh random positions on every call, which
    is right for training but not reproducible across checkpoints. This class
    materializes a fixed set (one target per document, chosen by `seed`) so
    two checkpoints are always scored on the identical samples.
    """

    def __init__(
        self,
        path: str,
        window_size: int,
        batch_size: int,
        doc_token_id: int,
        seed: int = 0,
        use_cache: bool = True,
        loss_spans: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.window_size = int(window_size)
        self.batch_size = int(batch_size)

        tokens = load_tokens(path)
        doc_starts, doc_ends = _document_bounds(path, tokens, int(doc_token_id), use_cache)
        self.num_docs = len(doc_starts)

        rng = np.random.default_rng(seed)
        mask = loss_target_mask(path, tokens, loss_spans, int(doc_token_id), use_cache)
        if mask is None:
            targets = rng.integers(doc_starts + 1, doc_ends)  # exclusive upper bound
        else:  # one target per document among its loss-span positions (documents without any are left out)
            allowed = np.flatnonzero(mask)
            lo, hi = np.searchsorted(allowed, doc_starts + 1), np.searchsorted(allowed, doc_ends)
            keep = hi > lo
            doc_starts, lo, hi = doc_starts[keep], lo[keep], hi[keep]
            targets = allowed[lo + (rng.random(len(lo)) * (hi - lo)).astype(np.int64)]
            self.num_docs = len(doc_starts)
        self.x, _ = _fill_windows(tokens, targets, doc_starts, self.window_size)
        self.y = np.asarray(tokens[targets]).astype(np.int32)

    def __len__(self) -> int:
        return int(np.ceil(self.num_docs / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s = idx * self.batch_size
        return self.x[s : s + self.batch_size], self.y[s : s + self.batch_size]


def gradient_accumulation(config: Dict[str, Any]) -> int:
    """Micro-batches per optimizer update: data.batch_size / data.micro_batch_size (1 without micro-batches)."""
    data = config["data"]
    micro = data.get("micro_batch_size")
    if not micro:
        return 1
    if int(data["batch_size"]) % int(micro):
        raise ValueError(f"data.batch_size ({data['batch_size']}) must be a multiple of data.micro_batch_size ({micro})")
    return int(data["batch_size"]) // int(micro)


def build_datasets(config: Dict[str, Any], model_context_length: int, packed: bool = False,
                   static_memory: Optional[Dict[str, Any]] = None) -> Tuple[keras.utils.PyDataset, VariablePrefixDataset]:
    """Builds the (train, validation) pair for a run.

    Dataset-level facts (`doc_token_id`, `vocabulary_size`) come from the
    dataset's `metadata.json`, never from the run config, so two runs on the
    same corpus cannot disagree about them. Validation uses the single-target
    sampler with corruption off, bounded by `training.validation_steps`.
    `packed` (archs with `packed_documents`, e.g. infinite context) trains on
    `PackedDocumentDataset` rows. `static_memory` (the model's `static_memory_spec`): training and validation rows
    without the span contents, which go to the memory encoder (see `PackedDocumentDataset`).
    """
    data = config["data"]
    name = data["dataset"]
    meta = cfg_mod.load_dataset_metadata(name)

    common = dict(
        window_size=model_context_length,
        batch_size=data["batch_size"],
        vocabulary_size=meta["vocabulary_size"],
        doc_token_id=meta["doc_token_id"],
        seed=data["seed"],
        loss_spans=meta.get("loss_spans"),
    )
    if packed:
        train_cls = PackedDocumentDataset
    else:
        train_cls = DenseWindowDataset if config["training"].get("dense_supervision") else VariablePrefixDataset
    # micro-batches: `micro_batch_size` rows per forward pass, batch_size / micro_batch_size of them per optimizer
    # update (accumulated by the dense train step); steps_per_epoch still counts updates
    micro = int(data.get("micro_batch_size") or data["batch_size"])
    accum = gradient_accumulation(config)
    common["batch_size"] = micro
    train = train_cls(
        path=str(cfg_mod.dataset_split_path(name, "train")),
        steps_per_epoch=data["steps_per_epoch"] * accum,
        corrupt_prob=data["corrupt_prob"],
        **({"stream_chunks": int(data.get("stream_chunks") or 0) if config["training"].get("stateful") else 0,
            "random_windows": bool(data.get("random_windows")), "steps_multiple": accum,
            "epoch_passes": float(data.get("epoch_passes") or 1.0),
            "static_memory": static_memory}
           if packed else {}),
        **common,
    )
    if packed and config["training"].get("dense_validation"):
        # packed rows scored at every target (as training), the same rows every epoch: one pass over a row gives
        # all its targets, instead of a whole window per sampled target
        # data.validation_window / validation_batch_size: longer rows than training's (truncated BPTT trains on
        # short carried chunks; validation rows start at a document start with a zero state)
        val = {**common, "window_size": int(data.get("validation_window") or common["window_size"]),
               "batch_size": int(data.get("validation_batch_size") or common["batch_size"])}
        validation = PackedDocumentDataset(
            path=str(cfg_mod.dataset_split_path(name, "validation")),
            steps_per_epoch=config["training"]["validation_steps"],
            corrupt_prob=0.0,
            fixed=True,
            static_memory=static_memory,
            **val,
        )
        return train, validation
    validation = VariablePrefixDataset(
        path=str(cfg_mod.dataset_split_path(name, "validation")),
        steps_per_epoch=config["training"]["validation_steps"],
        corrupt_prob=0.0,
        **common,
    )
    return train, validation


def build_full_eval_dataset(config: Dict[str, Any], model_context_length: int, batch_size: Optional[int] = None) -> FullEvalDataset:
    """Deterministic full-coverage validation set for `model evaluate`."""
    data = config["data"]
    name = data["dataset"]
    meta = cfg_mod.load_dataset_metadata(name)
    return FullEvalDataset(
        path=str(cfg_mod.dataset_split_path(name, "validation")),
        window_size=model_context_length,
        batch_size=batch_size or data["batch_size"],
        doc_token_id=meta["doc_token_id"],
        seed=0,
        loss_spans=meta.get("loss_spans"),
    )
