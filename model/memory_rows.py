"""Batches for `min-gru`'s static memory (numpy, shared by the data loader and the model).

The contents of the system / tool spans never enter the layers: they leave the token rows (their open / close tags
stay) and become one memory per document, built by a small encoder over the document's span tokens. A batch is a
dict of arrays:

    ids        (B, T)      the rows without span contents (filled with markers past `lengths`)
    lengths    (B,)        real tokens per row
    enc        (E, Le)     encoder rows: each document's sequence (a marker, then its span tags and contents in order),
                           packed several per row, the rest markers
    enc_tok    (Ne,)       flat positions in enc of the documents' tokens (the rest is filling)
    enc_init   (E,)        row b whose carried encoder state starts this encoder row (generation), -1: none
    mem_count  (G, n)      entries of each document's memories (1..n: system, tools)
    mem_pos    (Nm,)       the entries' flat positions in enc
    mem_slot   (Nm,)       their slots in chunks of READ_CHUNK entries of one (document, memory) each (flat)
    mem_seg    (Nck,)      each chunk's (document, memory): g n + i
    read_pos   (R,)        reading positions (flat into ids)
    read_slot  (R,)        their slots in chunks of READ_CHUNK positions reading one document each (flat, Nc x 32)
    chunk_doc  (Nc,)       the document (0..G-1) each chunk reads
    valid      (B, T)      targets (training / validation: positions with a loss target)

Every index list holds distinct positions (no padding entries): on the Iris Xe a scatter-add with thousands of
entries on one padding row costs ~7x one with distinct rows, and that is the backward of every padded gather.

Generation adds `carry_doc` (B,) (the document continuing from the previous call, -1: none), `last_doc` (B,) (the
row's last document, whose memory is carried) and `enc_end` (B,) (flat index into enc of that document's last
encoder token; -1: none in this call).

Every document's span tokens come before its first `<assistant>` in the agent datasets (checked on
agent-mix-v2-bpe1024), so a reading position reads its document's whole memory.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

READ_CHUNK = 32  # reading positions per chunk (one document each)


def span_masks(tokens: np.ndarray, spans: Sequence[Sequence[int]], read_span: Sequence[int], reset: int,
               open0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For a 1-D token array: (kind (N,) int8: 1..n inside memory span i (tags excluded), 0 elsewhere; tag (N,) bool:
    a memory span's open / close tag; read (N,) bool: inside the read span or its opening tag; open_after
    (n + 1,) bool; doc (N,) int: document number, 0 = the one open before the array). `open0` (n + 1,) bool: spans
    open before the array (generation)."""
    N = len(tokens)
    pos = np.arange(N, dtype=np.int32 if N < 2 ** 31 - 1 else np.int64)  # (int32: the whole corpus in less memory)
    is_reset = tokens == reset
    last_reset = np.maximum.accumulate(np.where(is_reset, pos, -2)) if N else pos

    def inside(o: int, c: int, before: bool) -> np.ndarray:  # before: a virtual opening at -1
        last_open = np.maximum.accumulate(np.where(tokens == o, pos, -1 if before else -2)) if N else pos
        last_close = np.maximum.accumulate(np.where(tokens == c, pos, -2)) if N else pos
        return (last_open > np.maximum(last_close, last_reset)) & (tokens != o)

    open0 = np.zeros(len(spans) + 1, bool) if open0 is None else np.asarray(open0, bool)
    kind = np.zeros(N, np.int8)
    tag = np.zeros(N, bool)
    open_after = np.zeros(len(spans) + 1, bool)
    for i, (o, c) in enumerate(list(spans) + [list(read_span)]):
        m = inside(o, c, bool(open0[i]))
        if i < len(spans):
            kind[m] = i + 1
            tag |= (tokens == o) | (tokens == c)
        else:
            read = m | (tokens == o)
        open_after[i] = (m[-1] or tokens[-1] == o) if N else open0[i]
    return kind, tag, read, open_after, np.cumsum(is_reset)


def pack(rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], docs: Dict[Any, Tuple[np.ndarray, np.ndarray, int]],
         reset: int, memories: int, T: Optional[int] = None, chunk: int = READ_CHUNK, enc_length: int = 256,
         keep_docs: Sequence[Any] = ()) -> Dict[str, np.ndarray]:
    """rows: per row (ids, valid, read, doc key) of its kept tokens; docs: key -> (encoder tokens, their kinds, row
    whose carried encoder state the sequence continues or -1). Builds the batch dict (see the module doc). Only the
    documents that are read (and `keep_docs`) get a memory; reads of documents without one are dropped (nothing to
    read). T: the row length (default: the longest row)."""
    B = len(rows)
    T = int(T or max([len(r[0]) for r in rows] + [1]))
    ids = np.full((B, T), reset, np.int32)
    valid = np.zeros((B, T), bool)
    lengths = np.zeros(B, np.int32)
    flat_read, read_keys = [], []
    for b, (x, v, r, d) in enumerate(rows):
        n = len(x)
        ids[b, :n], valid[b, :n], lengths[b] = x, v, n
        at = np.flatnonzero(r)
        flat_read.append(b * T + at)
        read_keys.extend(d[at].tolist())
    flat_read = np.concatenate(flat_read) if flat_read else np.zeros(0, np.int64)
    # documents with a memory: read or kept, and with entries (or continuing a carried memory)
    used = [k for k in dict.fromkeys(list(dict.fromkeys(read_keys)) + list(keep_docs))
            if k in docs and (len(docs[k][0]) or docs[k][2] >= 0 or k in keep_docs)]
    g_of = {k: g for g, k in enumerate(used)}
    # encoder rows: continuations first in their own rows (their carried state starts the row), then first fit
    lens = [len(docs[k][0]) for k in used]
    Le = max([enc_length] + lens)
    enc_rows: List[np.ndarray] = []
    fill: List[int] = []
    enc_init: List[int] = []
    start = np.zeros(len(used), np.int64)
    order = sorted(range(len(used)), key=lambda g: (docs[used[g]][2] < 0, -lens[g]))
    for g in order:
        tok, kind, cont = docs[used[g]]
        if not len(tok):
            continue
        r = next((r for r in range(len(enc_rows)) if cont < 0 and enc_init[r] < 0 and fill[r] + len(tok) <= Le),
                 None) if cont < 0 else None
        if r is None:
            enc_rows.append(np.full(Le, reset, np.int32))
            fill.append(0)
            enc_init.append(int(cont))
            r = len(enc_rows) - 1
        enc_rows[r][fill[r]:fill[r] + len(tok)] = tok
        start[g] = r * Le + fill[r]
        fill[r] += len(tok)
    E = len(enc_rows)
    G = len(used)
    # memory entries in chunks of `chunk`, each holding one (document, memory)
    mem_count = np.zeros((G, memories), np.int32)
    mem_pos, mem_slot, mem_seg = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], []
    chunks_so_far = 0
    for g, k in enumerate(used):
        kind = docs[k][1]
        for i in range(memories):
            at = np.flatnonzero(kind == i + 1)
            if not len(at):
                continue
            mem_count[g, i] = len(at)
            mem_pos.append(start[g] + at)
            mem_slot.append(chunks_so_far * chunk + np.arange(len(at)))
            c = -(-len(at) // chunk)
            mem_seg += [g * memories + i] * c
            chunks_so_far += c
    enc_tok = np.concatenate([start[g] + np.arange(n) for g, n in enumerate(lens) if n] + [np.zeros(0, np.int64)])
    # reads: grouped by document, in chunks of `chunk` positions
    g_read = np.array([g_of.get(k, -1) for k in read_keys], np.int64)
    keep = g_read >= 0
    flat_read, g_read = flat_read[keep], g_read[keep]
    o = np.argsort(g_read, kind="stable")
    flat_read, g_read = flat_read[o], g_read[o]
    counts = np.bincount(g_read, minlength=G)
    chunks = (counts + chunk - 1) // chunk
    first_chunk = np.concatenate([[0], np.cumsum(chunks)[:-1]]).astype(np.int64)
    first_read = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    rank = np.arange(len(g_read)) - first_read[g_read]
    out = {"ids": ids, "lengths": lengths, "valid": valid,
           "enc": np.stack(enc_rows) if E else np.zeros((0, Le), np.int32), "enc_tok": enc_tok.astype(np.int64),
           "enc_init": np.asarray(enc_init, np.int64), "mem_count": mem_count,
           "mem_pos": np.concatenate(mem_pos).astype(np.int64), "mem_slot": np.concatenate(mem_slot).astype(np.int64),
           "mem_seg": np.asarray(mem_seg, np.int64),
           "read_pos": flat_read.astype(np.int64), "read_slot": (first_chunk[g_read] + rank // chunk) * chunk + rank % chunk,
           "chunk_doc": np.repeat(np.arange(G), chunks).astype(np.int64)}
    out["_g_of"] = g_of  # (python) document key -> memory number
    out["_end"] = {k: start[g] + len(docs[k][0]) - 1 for g, k in enumerate(used) if len(docs[k][0])}
    return out


def from_raw(ids: np.ndarray, spans: Sequence[Sequence[int]], read_span: Sequence[int], reset: int,
             valid: Optional[np.ndarray] = None, open0: Optional[np.ndarray] = None, carried: Optional[np.ndarray] = None,
             chunk: int = READ_CHUNK) -> Tuple[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]:
    """Rows of raw tokens `(B, T)` (span contents included) as a batch dict. `open0` `(B, n + 1)`: spans open before
    each row; `carried` `(B,)` bool: the row's first document continues one from a previous call (its memory and
    encoder state are carried: generation). A row's first document otherwise starts fresh (its spans before the row
    are not seen). Also returns each row's kept positions and the spans open after each row."""
    ids = np.asarray(ids)
    B = ids.shape[0]
    rows, docs, kept_at, opens = [], {}, [], []
    keep_docs, last_doc, carry_doc = [], np.full(B, -1, np.int64), np.full(B, -1, np.int64)
    for b in range(B):
        tok = ids[b].astype(np.int64)
        kind, tag, read, open_after, doc = span_masks(tok, spans, read_span, reset, None if open0 is None else open0[b])
        kept = kind == 0
        at = np.flatnonzero(kept)
        kept_at.append(at)
        opens.append(open_after)
        v = np.ones(len(tok), bool) if valid is None else np.asarray(valid[b], bool)
        rows.append((tok[at], v[at], read[at], doc[at] * B + b))  # document key: number in the row * B + row
        cont = carried is not None and bool(carried[b])
        for d in np.unique(np.concatenate([[0], doc])):
            m = (doc == d) & ((kind > 0) | tag)
            seq, kd = tok[m], kind[m]
            if not (d == 0 and cont):  # a fresh document: its marker starts the encoder (state restarts)
                seq, kd = np.concatenate([[reset], seq]), np.concatenate([[0], kd]).astype(np.int8)
            docs[int(d) * B + b] = (seq.astype(np.int32), kd, b if (d == 0 and cont) else -1)
        last = int(doc[-1]) * B + b if len(doc) else b
        keep_docs.append(last)
        last_doc[b] = last
        if cont:
            keep_docs.append(b)
            carry_doc[b] = b
    batch = pack(rows, docs, reset, len(spans), T=None, chunk=chunk, keep_docs=keep_docs if carried is not None else ())
    if carried is not None:
        g_of = batch["_g_of"]
        batch["last_doc"] = np.array([g_of.get(int(k), -1) for k in last_doc], np.int64)
        batch["carry_doc"] = np.array([g_of.get(int(k), -1) if k >= 0 else -1 for k in carry_doc], np.int64)
        batch["enc_end"] = np.array([batch["_end"].get(int(k), -1) for k in last_doc], np.int64)
    return batch, kept_at, np.stack(opens)
