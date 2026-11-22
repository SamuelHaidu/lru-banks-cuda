"""Local web chat with a trained run: pick a run and checkpoint, a system prompt and tools from the training data, and
talk to the model; tokens stream to the page, and the model's tool calls run for real.

    cd webchat/ui && npm install && npm run build       # once (and after UI changes): builds ui/dist/index.html
    uv run python -m webchat.server [--port 8000] [--device xpu|cpu|cuda]     # then open http://localhost:8000

Two processes: this async HTTP server (FastAPI) serves the page, the tools and the catalogs; the model runs in a
worker process (`webchat/worker.py`) it starts and talks to over a Unix socket (`webchat/ipc.py`). Closing or
stopping a reply cancels it in the worker. If the worker dies, the next request starts a new one.

Tools the page can list and the server executes (`webchat/tools.py`): the simulated bash and files of the synthetic
dataset (`run`, `read_file`, `write_file`), `calculator`, `python` (real Python in a separate `python -I`, 10 s
timeout), the memory tools of every naming in the training data (saved to .webchat/memory.json), and the Atum09
tools mapped onto those; anything else answers {"error": "tool not available"}.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from webchat import catalog, ipc
from webchat.models import ROOT, Models
from webchat.tools import Workspace

UI = Path(__file__).resolve().parent / "ui" / "dist" / "index.html"
NO_UI = """<!doctype html><meta charset="utf-8"><title>webchat</title>
<body style="font:15px system-ui;max-width:640px;margin:4em auto">
<h3>The page is not built yet</h3><p>Build it once (needs Node), then reload:</p>
<pre>cd webchat/ui &amp;&amp; npm install &amp;&amp; npm run build</pre></body>"""


class WorkerProcess:
    """The inference worker: started on demand, connected over a Unix socket, started again if it exited."""

    def __init__(self, device: Optional[str] = None, start_timeout: float = 180.0):
        self.device, self.start_timeout = device, start_timeout
        base = os.environ.get("XDG_RUNTIME_DIR") or str(ROOT / ".webchat")
        Path(base).mkdir(parents=True, exist_ok=True)
        self.path = str(Path(base) / f"rnn-webchat-{os.getpid()}.sock")
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._client: Optional[ipc.WorkerClient] = None
        self._lock = asyncio.Lock()

    async def client(self) -> ipc.WorkerClient:
        async with self._lock:
            if self._client is None or self._client.task.done():
                await self._start()
            assert self._client is not None
            return self._client

    async def _start(self) -> None:
        await self.stop()
        args = [sys.executable, "-m", "webchat.worker", "--socket", self.path]
        if self.device:
            args += ["--device", self.device]
        self.proc = await asyncio.create_subprocess_exec(*args, cwd=str(ROOT))
        deadline = asyncio.get_running_loop().time() + self.start_timeout
        while True:
            if self.proc.returncode is not None:
                raise ipc.WorkerError(f"the inference worker exited at start (code {self.proc.returncode})")
            if os.path.exists(self.path):
                try:
                    self._client = await ipc.WorkerClient.connect(self.path)
                    await self._client.call("ping")
                    return
                except (ConnectionError, OSError, ipc.WorkerError):
                    pass
            if asyncio.get_running_loop().time() > deadline:
                raise ipc.WorkerError("the inference worker did not start")
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.proc.wait(), 10)
            if self.proc.returncode is None:
                self.proc.kill()
        if os.path.exists(self.path):
            os.unlink(self.path)


def create_app(worker: Any, workspace: Optional[Workspace] = None) -> FastAPI:
    """The app; `worker` is anything with `async client() -> ipc.WorkerClient` and `async stop()`."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await worker.stop()

    app = FastAPI(title="webchat", lifespan=lifespan)
    ws = workspace or Workspace()
    runs = Models(None)

    def safe(fn, *args):  # catalog reads: report errors to the page as {"error": ...}
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def page():
        return HTMLResponse(UI.read_text() if UI.exists() else NO_UI)

    @app.get("/api/runs")
    async def api_runs():
        return await asyncio.to_thread(runs.runs)

    @app.get("/api/presets")
    async def api_presets(dataset: str = ""):
        examples = await asyncio.to_thread(catalog.data_examples, dataset) if dataset else {}
        return {"system": catalog.system_presets(), "tool_lines": catalog.TOOL_LINES, "tool_sets": catalog.TOOL_SETS,
                "atum_tool_lines": catalog.ATUM_TOOL_LINES, "atum_implemented": catalog.ATUM_IMPLEMENTED,
                "decoys": catalog.DECOY_TOOLS, "examples": catalog.EXAMPLES, "data_examples": examples}

    @app.get("/api/starters")
    async def api_starters(dataset: str = ""):
        return await asyncio.to_thread(safe, catalog.completion_starters, dataset)

    @app.get("/api/documents")
    async def api_documents(dataset: str, split: str = "validation"):
        return await asyncio.to_thread(safe, catalog.dataset_documents, dataset, split)

    @app.get("/api/workspace")
    async def api_workspace():
        return {"files": ws.files(), "memory": ws.memory}

    @app.post("/api/tool")
    async def api_tool(body: Dict[str, Any]):
        return {"result": await asyncio.to_thread(ws.call, body.get("call", ""))}

    @app.post("/api/workspace/reset")
    async def api_workspace_reset(body: Dict[str, Any]):
        seed = int(body.get("seed", random.randrange(1 << 30)))
        with ws.lock:
            ws.reset(seed)
        return {"files": ws.files()}

    @app.post("/api/memory/clear")
    async def api_memory_clear():
        with ws.lock:
            ws.memory.clear()
            ws._save_memory()
        return {"memory": {}}

    @app.post("/api/generate")
    async def api_generate(body: Dict[str, Any]):
        """Server-sent events: {"text": delta} as tokens come, then {"done": true, "tokens": n, "tps": ...}."""

        async def events():
            try:
                client = await worker.client()
                async with contextlib.aclosing(client.stream("generate", **body)) as it:
                    async for ev in it:
                        yield f"data: {json.dumps(ev.get('chunk', ev.get('result')), ensure_ascii=False)}\n\n"
            except ipc.WorkerError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    async def model_op(op: str, body: Dict[str, Any]):
        try:
            return await (await worker.client()).call(op, **body)
        except ipc.WorkerError as e:
            return {"error": str(e)}

    @app.post("/api/inspect")
    async def api_inspect(body: Dict[str, Any]):
        return await model_op("inspect", body)

    @app.post("/api/attend")
    async def api_attend(body: Dict[str, Any]):
        return await model_op("attend", body)

    @app.post("/api/banks")
    async def api_banks(body: Dict[str, Any]):
        return await model_op("banks", body)

    @app.exception_handler(json.JSONDecodeError)
    async def bad_json(request: Request, exc: json.JSONDecodeError):
        return JSONResponse({"error": "bad json"}, 400)

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Local web chat with a trained run.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default=None, help="xpu / cpu / cuda (default: the first loaded run's device)")
    a = p.parse_args()
    if not UI.exists():
        print("note: the page is not built: cd webchat/ui && npm install && npm run build", file=sys.stderr)
    print(f"webchat on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    uvicorn.run(create_app(WorkerProcess(a.device)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
