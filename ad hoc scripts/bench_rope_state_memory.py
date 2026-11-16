"""Times a dense training step (forward + fused readout loss + backward) of `rope-state-memory` on the XPU.

    uv run python "ad hoc scripts/bench_rope_state_memory.py" --layers 1 2 4 [--profile]
"""

import argparse
import time

import model  # noqa: F401  (backend setup)
from model.device import setup_device

import numpy as np
import torch

from model.archs import build_model
from model.fused_linear_cross_entropy import fused_linear_cross_entropy


def step(m, x, y):
    for v in m.trainable_weights:
        v.value.grad = None
    h = m.dense_hidden(x)
    loss = fused_linear_cross_entropy(h.reshape(-1, h.shape[-1]), m.readout.kernel, y.reshape(-1))
    loss.backward()
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="rope-state-memory")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--memory", default=None, help="arch_params.memory (default: the arch default)")
    a = p.parse_args()
    setup_device("xpu")
    rng = np.random.default_rng(0)
    for L in a.layers:
        params = {"vocabulary_size": 10000, "window_size": a.length, "chunk_size": a.chunk, "num_layers": L}
        if a.memory:
            params["memory"] = a.memory
        m = build_model({"arch": a.arch, "arch_params": params})
        dev = m.readout.kernel.value.device
        ids = rng.integers(1, 10000, (a.batch, a.length + 1))
        ids[:, 0] = 0
        ids[rng.random(ids.shape) < 1 / 200] = 0  # document markers, ~200-token stories
        x = torch.as_tensor(ids[:, :-1], dtype=torch.int32, device=dev)
        y = torch.as_tensor(ids[:, 1:], dtype=torch.int64, device=dev)
        for _ in range(2):
            step(m, x, y)
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            step(m, x, y)
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        mem = torch.xpu.max_memory_allocated() / 2 ** 30
        n = sum(int(np.prod(v.shape)) for v in m.trainable_weights)
        print(f"{a.arch} memory={a.memory or 'default'} layers={L} chunk={a.chunk} batch={a.batch}x{a.length}: {dt * 1000:.0f} ms/step, "
              f"{a.batch * a.length / dt / 1000:.0f}k tok/s, peak {mem:.2f} GiB, {n / 1e6:.2f}M params", flush=True)
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.XPU]) as prof:
                for _ in range(3):
                    step(m, x, y)
                torch.xpu.synchronize()
            print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=25))


if __name__ == "__main__":
    main()
