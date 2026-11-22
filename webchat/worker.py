"""The inference worker: loads one model at a time and runs generation and the Inspect computations, for the web
server, over a Unix socket (`webchat/ipc.py`).

    python -m webchat.worker --socket PATH [--device xpu|cpu|cuda]      (started by `python -m webchat.server`)

Ops: ping; generate (streams {"text": delta}, ends with {"done", "tokens", "tps"}); inspect, attend, banks (the
Inspect tab, `webchat/inspection.py`); cancel (stops a running generate). Model work runs on one thread, in order
(torch is not shared across threads); the event loop keeps reading messages, so cancel and ping are answered while a
generation runs.

Generation is `model.generation.stream`: for recurrent architectures the state is carried token by token, and the
state after a reply is kept, so when the next document extends the previous one (the next chat turn) only the new
text is fed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

from webchat import inspection, ipc
from webchat.models import Models


class Worker:
    def __init__(self, device: Optional[str]):
        self.models = Models(device)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="model")
        self.cancels: Dict[int, threading.Event] = {}
        self.session: Any = None  # model.generation.Session of the last generation (recurrent models)
        self.session_key: Optional[Tuple[str, str]] = None
        self.session_text = ""  # the document text the session's state covers

    # -- model work (the executor thread) ---------------------------------------------------------------------------

    def generate(self, args: Dict[str, Any], emit: Callable[[Dict[str, Any]], None], cancel: threading.Event):
        from model.generation import Session, stream

        key = (args["run"], str(args.get("checkpoint", "latest")))
        model, tok = self.models.load(*key)
        doc = args["document"]
        stop = ["<eos>"] if args.get("mode") == "completion" else ["</assistant>", "<eos>"]
        stop_ids = [ids[-1] for ids in (tok.encode(s).ids for s in stop) if ids]
        stateful = bool(getattr(model, "stateful_generation", False))
        session = None
        reused = 0
        if stateful:
            if self.session is None or self.session_key != key:
                self.session, self.session_key, self.session_text = Session(), key, ""
            session = self.session
            reuse = self.session_text and doc.startswith(self.session_text) and len(doc) > len(self.session_text)
            if reuse:  # the state covers the start of this document: feed only the rest
                reused = len(session.ids)
                ids = session.ids + tok.encode(doc[len(self.session_text):]).ids
            else:
                session.clear()
                ids = tok.encode(doc).ids
        else:
            ids = tok.encode(doc).ids
        new: list = []
        shown = 0
        stamps = []
        for i in stream(model, ids, max_new_tokens=int(args.get("max_new_tokens", 256)),
                        temperature=float(args.get("temperature", 0.0)),
                        top_k=int(args["top_k"]) if args.get("top_k") else None,
                        top_p=float(args["top_p"]) if args.get("top_p") else None,
                        repetition_penalty=float(args.get("repetition_penalty", 1.0)),
                        stop_ids=stop_ids, session=session, cancel=cancel):
            stamps.append(time.perf_counter())
            new.append(i)
            text = tok.decode(new, skip_special_tokens=False)
            if not text.endswith("�"):  # hold a partial UTF-8 character
                emit({"text": text[shown:]})
                shown = len(text)
        text = tok.decode(new, skip_special_tokens=False)
        if text[shown:]:
            emit({"text": text[shown:]})
        if session is not None:
            self.session_text = doc + text
        tps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        return {"done": True, "tokens": len(new), "tps": round(tps, 1), "cancelled": cancel.is_set(),
                "prompt_tokens": len(ids), "reused_tokens": reused}

    # -- messages (the event loop) ----------------------------------------------------------------------------------

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lock = asyncio.Lock()

        async def send(msg: Dict[str, Any]) -> None:
            async with lock:
                writer.write(ipc.pack(msg))
                await writer.drain()

        tasks = set()
        while (msg := await ipc.read_frame(reader)) is not None:
            op, rid, args = msg.get("op"), msg.get("id"), msg.get("args") or {}
            if op == "cancel":
                ev = self.cancels.get(args.get("target"))
                if ev is not None:
                    ev.set()
                continue
            if op == "ping":
                await send({"id": rid, "event": "result", "data": "pong"})
                continue
            t = asyncio.create_task(self.run(op, rid, args, send))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        for ev in self.cancels.values():  # the server went away: stop what runs
            ev.set()

    async def run(self, op: str, rid: int, args: Dict[str, Any], send) -> None:
        loop = asyncio.get_running_loop()
        try:
            if op == "generate":
                cancel = threading.Event()
                self.cancels[rid] = cancel
                chunks: asyncio.Queue = asyncio.Queue()
                emit = lambda d: loop.call_soon_threadsafe(chunks.put_nowait, d)  # noqa: E731
                fut = loop.run_in_executor(self.pool, self.generate, args, emit, cancel)
                getter = asyncio.ensure_future(chunks.get())
                while True:
                    done, _ = await asyncio.wait({fut, getter}, return_when=asyncio.FIRST_COMPLETED)
                    if getter in done:
                        await send({"id": rid, "event": "chunk", "data": getter.result()})
                        getter = asyncio.ensure_future(chunks.get())
                    elif fut in done:
                        getter.cancel()
                        while not chunks.empty():  # chunks emitted just before the end
                            await send({"id": rid, "event": "chunk", "data": chunks.get_nowait()})
                        await send({"id": rid, "event": "result", "data": fut.result()})
                        return
            elif op in ("inspect", "attend", "banks"):
                fn = {"inspect": inspection.inspect_text, "attend": inspection.attend, "banks": inspection.banks_at}[op]
                data = await loop.run_in_executor(self.pool, fn, self.models, args)
                await send({"id": rid, "event": "result", "data": data})
            else:
                await send({"id": rid, "event": "error", "message": f"unknown op {op!r}"})
        except Exception as e:  # noqa: BLE001 (reported to the page)
            try:
                await send({"id": rid, "event": "error", "message": f"{type(e).__name__}: {e}"})
            except ConnectionError:
                pass
        finally:
            self.cancels.pop(rid, None)


async def main_async(path: str, device: Optional[str]) -> None:
    worker = Worker(device)
    if os.path.exists(path):
        os.unlink(path)
    old = os.umask(0o177)  # the socket is the user's only (0600)
    try:
        server = await asyncio.start_unix_server(worker.serve, path=path, limit=ipc.MAX_FRAME)
    finally:
        os.umask(old)
    print(f"worker ready on {path}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--socket", required=True)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.device:
        from model.device import setup_device

        setup_device(a.device)
    try:
        asyncio.run(main_async(a.socket, a.device))
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(a.socket):
            os.unlink(a.socket)


if __name__ == "__main__":
    main()
