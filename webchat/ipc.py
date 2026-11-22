"""Server <-> worker messages over a Unix socket: frames of a 4-byte big-endian length then a msgpack map.

Request (server -> worker): {"id": int, "op": str, "args": {...}}; op "cancel" with args {"target": id} stops a
running request. Replies (worker -> server) carry the request's id:
    {"id", "event": "chunk", "data": ...}     a streamed piece (generation: {"text": delta})
    {"id", "event": "result", "data": ...}    the end: a one-shot op's value, or a stream's final summary
    {"id", "event": "error", "message": str}  the end, failed
One connection carries any number of requests at once: replies are routed by id.
"""

from __future__ import annotations

import asyncio
import itertools
import struct
import sys
from typing import Any, AsyncIterator, Dict, Optional

import msgpack

HEADER = struct.Struct(">I")
MAX_FRAME = 1 << 30


async def read_frame(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    """The next message, or None at the end of the connection."""
    try:
        (n,) = HEADER.unpack(await reader.readexactly(HEADER.size))
        if n > MAX_FRAME:
            raise ValueError(f"frame of {n} bytes")
        # integer map keys are allowed: results carry dicts keyed by position (JSON turns them into strings later)
        return msgpack.unpackb(await reader.readexactly(n), raw=False, strict_map_key=False)
    except asyncio.IncompleteReadError:
        return None


def _plain(obj: Any) -> Any:
    """numpy values (in results built from arrays) as Python ones."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"cannot send {type(obj).__name__}")


def pack(msg: Dict[str, Any]) -> bytes:
    body = msgpack.packb(msg, use_bin_type=True, default=_plain)
    return HEADER.pack(len(body)) + body


class WorkerError(RuntimeError):
    """The worker reported an error, or the connection to it was lost."""


class WorkerClient:
    """The server's side: `call` for one-shot ops, `stream` for streamed ones (cancelled when the consumer stops)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self.pending: Dict[int, asyncio.Queue] = {}
        self.ids = itertools.count(1)
        self.lock = asyncio.Lock()
        self.task = asyncio.create_task(self._route())

    @classmethod
    async def connect(cls, path: str) -> "WorkerClient":
        reader, writer = await asyncio.open_unix_connection(path, limit=MAX_FRAME)
        return cls(reader, writer)

    async def _route(self) -> None:
        reason = "the inference worker stopped"
        try:
            while (msg := await read_frame(self.reader)) is not None:
                q = self.pending.get(msg.get("id"))
                if q is not None:
                    q.put_nowait(msg)
        except Exception as e:  # noqa: BLE001 (an unreadable frame: the connection cannot be trusted any more)
            reason = f"lost the inference worker connection: {type(e).__name__}: {e}"
            print(f"webchat: {reason}", file=sys.stderr, flush=True)
        finally:  # every waiting request ends with an error; the next request connects again
            for q in self.pending.values():
                q.put_nowait({"event": "error", "message": reason})

    async def _send(self, msg: Dict[str, Any]) -> None:
        async with self.lock:
            self.writer.write(pack(msg))
            await self.writer.drain()

    async def _start(self, op: str, args: Dict[str, Any]) -> tuple[int, asyncio.Queue]:
        if self.task.done():
            raise WorkerError("the inference worker stopped")
        rid = next(self.ids)
        q: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = q
        await self._send({"id": rid, "op": op, "args": args})
        return rid, q

    async def call(self, op: str, **args: Any) -> Any:
        rid, q = await self._start(op, args)
        try:
            while True:
                msg = await q.get()
                if msg["event"] == "result":
                    return msg.get("data")
                if msg["event"] == "error":
                    raise WorkerError(msg.get("message", "error"))
        finally:
            self.pending.pop(rid, None)

    async def stream(self, op: str, **args: Any) -> AsyncIterator[Dict[str, Any]]:
        """Yields {"chunk": data} for each piece, then {"result": data}; raises WorkerError on an error. If the
        consumer stops early (a closed page), the worker is told to cancel."""
        rid, q = await self._start(op, args)
        finished = False
        try:
            while True:
                msg = await q.get()
                if msg["event"] == "chunk":
                    yield {"chunk": msg.get("data")}
                elif msg["event"] == "result":
                    finished = True
                    yield {"result": msg.get("data")}
                    return
                else:
                    finished = True
                    raise WorkerError(msg.get("message", "error"))
        finally:
            self.pending.pop(rid, None)
            if not finished and not self.task.done():
                try:
                    await self._send({"id": next(self.ids), "op": "cancel", "args": {"target": rid}})
                except (ConnectionError, RuntimeError):
                    pass

    async def close(self) -> None:
        self.task.cancel()
        self.writer.close()
