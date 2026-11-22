"""webchat (basic): the streaming generation core, the server <-> worker messages, and the HTTP routes (a stub
worker stands in for the model process)."""

import asyncio
import os
import tempfile
import threading
import unittest

import tests.helpers  # noqa: F401  (pins the device before keras/torch use)
from tests.helpers import tiny_arch_params

from model import generation as gen
from model.archs import build_model


def tiny(arch: str):
    return build_model({"arch": arch, "arch_params": tiny_arch_params(arch)})


class TestStream(unittest.TestCase):
    def test_stream_is_what_generate_returns(self):
        for arch in ("dual-sphere", "min-gru"):  # the windowed path and the recurrent (advance) path
            with self.subTest(arch=arch):
                m = tiny(arch)
                full = gen.generate(m, [1, 2, 3], max_new_tokens=6).split()
                self.assertEqual(full[3:], [str(i) for i in gen.stream(m, [1, 2, 3], max_new_tokens=6)])

    def test_session_continues_like_a_full_feed(self):
        m = tiny("min-gru")
        s = gen.Session()
        list(gen.stream(m, [1, 2, 3], max_new_tokens=4, session=s))
        more = s.ids + [5, 6]  # the next turn extends what the session covers
        self.assertEqual(list(gen.stream(m, more, max_new_tokens=4, session=s)),
                         list(gen.stream(m, more, max_new_tokens=4)))

    def test_cancel_stops_before_the_next_token(self):
        ev = threading.Event()
        ev.set()
        self.assertEqual(list(gen.stream(tiny("min-gru"), [1, 2], max_new_tokens=5, cancel=ev)), [])


class StubWorker:
    """Speaks the worker protocol: generate streams "a", "b" and ends; ping; other ops echo their arguments."""

    def __init__(self):
        self.cancelled = []

    async def serve(self, reader, writer):
        from webchat import ipc

        while (msg := await ipc.read_frame(reader)) is not None:
            rid, op = msg["id"], msg["op"]
            if op == "cancel":
                self.cancelled.append(msg["args"]["target"])
                continue
            if op == "generate":
                for t in ("a", "b"):
                    writer.write(ipc.pack({"id": rid, "event": "chunk", "data": {"text": t}}))
                    await writer.drain()
                    await asyncio.sleep(0.01)
                data = {"done": True, "tokens": 2, "tps": 0.0}
            else:
                data = "pong" if op == "ping" else {"echo": msg["args"]}
            writer.write(ipc.pack({"id": rid, "event": "result", "data": data}))
            await writer.drain()
        writer.close()


class FakeWorkerProcess:
    """The server's worker handle, backed by a StubWorker on a socket in a temp dir (in the server's event loop)."""

    def __init__(self):
        self.stub, self.client_ = StubWorker(), None
        self.dir = tempfile.mkdtemp()

    async def client(self):
        from webchat import ipc

        if self.client_ is None:
            path = os.path.join(self.dir, "w.sock")
            self.server = await asyncio.start_unix_server(self.stub.serve, path=path)
            self.client_ = await ipc.WorkerClient.connect(path)
        return self.client_

    async def stop(self):
        if self.client_ is not None:
            await self.client_.close()
            self.server.close()


class TestIPC(unittest.TestCase):
    def test_frames_keep_integer_keys(self):  # Inspect results hold dicts keyed by position
        from webchat import ipc

        async def main():
            r = asyncio.StreamReader()
            r.feed_data(ipc.pack({"id": 1, "data": {3: "x"}}))
            r.feed_eof()
            return await ipc.read_frame(r)

        self.assertEqual(asyncio.run(main())["data"], {3: "x"})

    def test_call_stream_and_cancel(self):
        async def main():
            w = FakeWorkerProcess()
            c = await w.client()
            self.assertEqual(await c.call("inspect", x=1), {"echo": {"x": 1}})
            events = [ev async for ev in c.stream("generate")]
            self.assertEqual([e.get("chunk") for e in events[:2]], [{"text": "a"}, {"text": "b"}])
            self.assertEqual(events[-1]["result"]["tokens"], 2)
            it = c.stream("generate")  # stopped after the first piece: the worker is told to cancel
            await it.__anext__()
            await it.aclose()
            await asyncio.sleep(0.05)
            self.assertEqual(len(w.stub.cancelled), 1)
            await w.stop()

        asyncio.run(main())


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from webchat.server import create_app

        cls.client = TestClient(create_app(FakeWorkerProcess()))
        cls.client.__enter__()  # one event loop for every request, as under uvicorn

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_page_and_runs(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        r = self.client.get("/api/runs")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_tool_runs_on_the_server(self):
        r = self.client.post("/api/tool", json={"call": 'calculator(expression="12*9")'})
        self.assertEqual(r.json(), {"result": "108"})

    def test_generate_streams_server_sent_events(self):
        r = self.client.post("/api/generate", json={"run": "x", "document": "<bos>"})
        self.assertEqual(r.headers["content-type"].split(";")[0], "text/event-stream")
        lines = [l for l in r.text.split("\n\n") if l.startswith("data: ")]
        self.assertEqual(lines[:2], ['data: {"text": "a"}', 'data: {"text": "b"}'])
        self.assertIn('"done": true', lines[-1])

    def test_model_ops_go_to_the_worker(self):
        self.assertEqual(self.client.post("/api/inspect", json={"text": "hi"}).json(), {"echo": {"text": "hi"}})


if __name__ == "__main__":
    unittest.main()
