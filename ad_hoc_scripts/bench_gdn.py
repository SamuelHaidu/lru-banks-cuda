"""Times `gated-deltanet` training steps on CUDA and shows where the time goes: achieved TFLOPS against the GPU's
bf16 tensor-core peak, the GPU kernels by time, and how much of it is matmuls on the tensor cores (bf16 / tf32 MMA
kernels) vs on the plain CUDA cores (fp32 SIMT gemms, elementwise).

    python ad_hoc_scripts/bench_gdn.py [--batch 96] [--window 512] [--set dim=256 num_layers=16 ...]
        [--peak 111.1] [--steps 10] [--profile]
"""

import argparse
import time

from model.device import setup_device

setup_device("cuda")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.archs.gated_deltanet import GatedDeltaNetLM  # noqa: E402
from model.fused_linear_cross_entropy import fused_linear_cross_entropy  # noqa: E402


def forward_macs(m: GatedDeltaNetLM, chunk: int = 64) -> float:
    """Multiply-adds per token of one forward pass (top-1 MoE: one expert per token)."""
    D, H, Dh, F, E, V = m.dim, m.num_heads, m.head_dim, m.expert_hidden, m.num_experts, m.vocabulary_size
    HD = H * Dh
    proj = D * 3 * HD + D * 2 * H + D * HD + HD * D + D * E + 2 * D * F
    delta = H * (4 * Dh * Dh + 2 * chunk * 2 * Dh)  # state reads / writes + the chunk's K K^T, Q K^T, (.)U
    return m.num_layers * (proj + delta) + D * V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--peak", type=float, default=111.1, help="bf16 dense tensor-core TFLOPS of the GPU")
    p.add_argument("--set", nargs="*", default=[], help="arch params key=value")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    kw = {"vocabulary_size": 1024, "window_size": a.window}
    for s in a.set:
        k, v = s.split("=")
        kw[k] = float(v) if "." in v else int(v)
    m = GatedDeltaNetLM(**kw)
    params = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    rng = np.random.default_rng(0)
    x = rng.integers(1, 1024, (a.batch, a.window)).astype(np.int32)
    x[:, ::300] = 0  # some document starts
    y = torch.as_tensor(rng.integers(1, 1024, (a.batch, a.window)), device="cuda")
    valid = torch.ones(a.batch, a.window, dtype=torch.bool, device="cuda")
    opt = torch.optim.AdamW([w.value for w in m.trainable_weights], lr=1e-4)

    def step():
        h = m.dense_hidden_valid(x, valid)
        loss = fused_linear_cross_entropy(m.readout_input(h), m.readout.kernel, y[valid])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(3):  # compile / autotune
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / a.steps
    tokens = a.batch * a.window
    flops = 6 * forward_macs(m) * tokens  # forward 2 x MACs, backward 2x the forward
    print(f"gated-deltanet {kw}: {params / 1e6:.1f}M params ({m.num_experts} experts, top-1)")
    print(f"step {dt * 1e3:.0f} ms, {tokens / dt / 1e3:.0f}k tokens/s, ~{flops / dt / 1e12:.1f} TFLOPS achieved "
          f"({100 * flops / dt / 1e12 / a.peak:.0f}% of the {a.peak} bf16 tensor-core peak), "
          f"peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    # what a plain bf16 matmul reaches here (the realistic ceiling)
    A = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        A @ A
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        A @ A
    torch.cuda.synchronize()
    print(f"plain bf16 8192^3 matmul: {10 * 2 * 8192 ** 3 / (time.time() - t) / 1e12:.1f} TFLOPS")
    if not a.profile:
        return
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    groups = {"tensor-core matmul (bf16/tf32 MMA)": 0.0, "fp32 matmul on CUDA cores (SIMT)": 0.0,
              "fla delta-rule kernels": 0.0, "other (elementwise, norms, copies, scatter)": 0.0}
    rows = []
    for e in prof.key_averages():
        if e.device_type.name != "CUDA" or not e.self_device_time_total:
            continue
        n = e.key.lower()
        us = e.self_device_time_total
        if any(s in n for s in ("chunk", "fla", "delta", "wy_", "kkt", "solve_tril", "fwd_h", "bwd_dh", "fwd_o", "bwd_dqkwg")):
            g = "fla delta-rule kernels"
        elif "_moe_up_gelu" in n or "_moe_dz_gelu" in n:  # our grouped Triton GEMMs (tl.dot: tensor cores)
            g = "tensor-core matmul (bf16/tf32 MMA)"
        elif any(s in n for s in ("gemm", "xmma", "cutlass", "ampere", "sm80", "sm86", "matmul", "mm_")):
            g = ("tensor-core matmul (bf16/tf32 MMA)" if any(s in n for s in ("bf16", "s16816", "tensorop", "16816", "h1688", "tf32", "xmma", "cutlass"))
                 else "fp32 matmul on CUDA cores (SIMT)")
        else:
            g = "other (elementwise, norms, copies, scatter)"
        groups[g] += us
        rows.append((us, g, e.key))
    total = sum(groups.values())
    print(f"\nGPU busy {total / 3 / 1e3:.0f} ms per step of {dt * 1e3:.0f} ms ({100 * total / 3 / 1e3 / (dt * 1e3):.0f}%: the rest"
          f" is the GPU waiting for Python / launches / host syncs)")
    print("GPU time by kind (3 steps):")
    for g, us in groups.items():
        print(f"  {100 * us / total:5.1f}%  {g}")
    print("\ntop kernels:")
    for us, g, k in sorted(rows, reverse=True)[:18]:
        print(f"  {100 * us / total:5.1f}%  [{g.split(' ')[0]}] {k[:110]}")


if __name__ == "__main__":
    main()
