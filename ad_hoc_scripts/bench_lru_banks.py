"""Times `lru-banks` training steps on CUDA (forward, fused readout loss, backward, AdamW) and shows where the GPU
time goes: tensor-core matmuls, fla / our Triton recurrence kernels, the rest (elementwise, norms, copies).

    python ad_hoc_scripts/bench_lru_banks.py [--batch 64] [--window 512] [--set dim=512 ...] [--steps 10]
        [--peak 111.1] [--profile]
"""

import argparse
import time

from model.device import setup_device

setup_device("cuda")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.archs.lru_banks import LRUBanksLM  # noqa: E402
from model.fused_linear_cross_entropy import fused_linear_cross_entropy  # noqa: E402


def forward_macs(m: LRUBanksLM, chunk: int = 64) -> float:
    """Multiply-adds per token of one forward pass (the matmuls; the bank reads as two chunked passes per hop)."""
    D, N, H, Dh, M, F, V, E0 = (m.dim, m.lru_dim, m.num_heads, m.head_dim, m.slots, m.ffn_mult * m.dim,
                                m.vocabulary_size, m.embed_dim)
    HD = H * Dh
    proj = D * 2 * N + 2 * (2 * N * HD) + D * HD + HD * D  # LRU in, W_k, W_q, W_v, W_o
    read = H * (chunk * Dh + Dh * M + chunk * M + M * Dh) * 2  # scores pass + values pass (chunk + state terms)
    per_hop = 2 * D * F + read  # every hop: a read and an FFN pass; hop 2 also its query and output projections
    return sum(proj + h * per_hop + (h - 1) * 2 * D * HD for h in m.layer_hops) + E0 * D + D * E0 + E0 * V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--peak", type=float, default=111.1, help="bf16 dense tensor-core TFLOPS of the GPU")
    p.add_argument("--set", nargs="*", default=[], help="arch params key=value")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    kw = {"window_size": a.window}
    for s in a.set:
        k, v = s.split("=")
        kw[k] = float(v) if "." in v else int(v)
    m = LRUBanksLM(**kw)
    V = m.vocabulary_size
    params = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    rng = np.random.default_rng(0)
    x = rng.integers(1, V, (a.batch, a.window)).astype(np.int32)
    x[:, ::300] = 0  # some document starts
    y = torch.as_tensor(rng.integers(1, V, (a.batch, a.window)), device="cuda")
    valid = torch.ones(a.batch, a.window, dtype=torch.bool, device="cuda")
    opt = torch.optim.AdamW([w.value for w in m.trainable_weights], lr=1e-4, fused=True)

    def step():
        h = m.dense_hidden_valid(x, valid)
        loss = fused_linear_cross_entropy(m.readout_input(h), m.readout.kernel, y[valid])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t0 = time.time()
    for _ in range(3):  # compile / autotune
        step()
    torch.cuda.synchronize()
    print(f"warm-up (compile, autotune) {time.time() - t0:.0f} s", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(a.steps):
        loss = step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / a.steps
    tokens = a.batch * a.window
    flops = 6 * forward_macs(m) * tokens
    print(f"lru-banks {kw}: {params / 1e6:.1f}M params, loss {float(loss):.3f}")
    print(f"step {dt * 1e3:.0f} ms, {tokens / dt / 1e3:.1f}k tokens/s, ~{flops / dt / 1e12:.1f} TFLOPS "
          f"({100 * flops / dt / 1e12 / a.peak:.0f}% of the {a.peak} bf16 peak), peak memory "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    if not a.profile:
        return
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    groups = {"tensor-core matmul": 0.0, "fla bank kernels": 0.0, "our Triton scans (LRU, mass)": 0.0,
              "compiled elementwise (inductor)": 0.0, "other (elementwise, norms, copies, reductions)": 0.0}
    rows = []
    for e in prof.key_averages():
        if e.device_type.name != "CUDA" or not e.self_device_time_total:
            continue
        n = e.key.lower()
        us = e.self_device_time_total
        if "_lru_" in n or "_mass_" in n:
            g = "our Triton scans (LRU, mass)"
        elif any(s in n for s in ("chunk", "gsa", "gla", "fwd_h", "bwd_dh", "cumsum")):
            g = "fla bank kernels"
        elif n.startswith("triton_"):
            g = "compiled elementwise (inductor)"
        elif any(s in n for s in ("gemm", "xmma", "cutlass", "ampere", "sm80", "sm86", "matmul", "mm_")):
            g = "tensor-core matmul"
        else:
            g = "other (elementwise, norms, copies, reductions)"
        groups[g] += us
        rows.append((us, g, e.key))
    total = sum(groups.values())
    print(f"\nGPU busy {total / 3 / 1e3:.0f} ms per step of {dt * 1e3:.0f} ms ({100 * total / 3 / 1e3 / (dt * 1e3):.0f}%)")
    for g, us in groups.items():
        print(f"  {100 * us / total:5.1f}%  {g}")
    print("\ntop kernels:")
    for us, g, k in sorted(rows, reverse=True)[:25]:
        print(f"  {100 * us / total:5.1f}%  [{g.split(' ')[0]}] {k[:110]}")


if __name__ == "__main__":
    main()
