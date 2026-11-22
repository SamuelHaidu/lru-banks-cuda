"""Ad hoc: does `keras.ops.associative_scan`'s per-step cost on XPU actually
improve (relative to real FLOPs) as the per-position dimension scales up?

Motivation: `dual-sphere-linear-recurrence` (a context-conditioned linear
recurrence over the window's dim_in=8 mini tokens, computed via
`keras.ops.associative_scan`) measured ~155-260ms/step full training --
close to a 2-block ALiBi Transformer's ~170ms/step despite far fewer real
FLOPs. The suspected cause: XPU kernel-launch overhead is roughly fixed per
op, so scans over small (dim=8) matrices are dominated by dispatch overhead
rather than compute. This script tests that hypothesis directly: same
window_size, same batch_size, same combine function, dim_in swept from 8 up
to 128 -- if the hypothesis is right, ms/step should NOT scale linearly (or
even close to linearly) with dim_in^3 (the FLOP cost of dim x dim matmuls),
because at small dim the wall-clock time is dominated by a roughly-constant
per-stage overhead, not by dim.

Usage:
    .venv/bin/python ad_hoc_scripts/associative_scan_dim_scaling.py
"""

import time

from model.device import setup_backend

setup_backend()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from keras import ops  # noqa: E402

BATCH_SIZE = 512
WINDOW_SIZE = 128
DIMS = [8, 16, 32, 64, 128]
N_TRIALS = 20
N_WARMUP = 3


def _combine(left, right):
    a_i, b_i = left
    a_j, b_j = right
    a_combined = ops.einsum("...ij,...jk->...ik", a_j, a_i)
    b_combined = ops.einsum("...ij,...j->...i", a_j, b_i) + b_j
    return a_combined, b_combined


def _time_scan(dim: int) -> float:
    rng = np.random.default_rng(0)
    a = ops.convert_to_tensor(
        (rng.standard_normal((BATCH_SIZE, WINDOW_SIZE, dim, dim)) * 0.05).astype("float32")
    )
    b = ops.convert_to_tensor(
        (rng.standard_normal((BATCH_SIZE, WINDOW_SIZE, dim)) * 0.05).astype("float32")
    )

    for _ in range(N_WARMUP):
        _ = ops.associative_scan(_combine, (a, b), axis=1)
    if torch.xpu.is_available():
        torch.xpu.synchronize()

    t0 = time.time()
    for _ in range(N_TRIALS):
        _ = ops.associative_scan(_combine, (a, b), axis=1)
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    return (time.time() - t0) / N_TRIALS


def main() -> int:
    print(f"batch_size={BATCH_SIZE}  window_size={WINDOW_SIZE}  device={'xpu' if torch.xpu.is_available() else 'cpu'}")
    print(f"{'dim_in':>8} {'ms/scan':>10} {'FLOPs (rel, dim^3)':>20} {'ms per rel-FLOP':>18}")

    results = []
    for dim in DIMS:
        ms = _time_scan(dim) * 1000
        rel_flops = dim ** 3
        results.append((dim, ms, rel_flops))
        print(f"{dim:>8} {ms:>10.2f} {rel_flops:>20,} {ms / rel_flops * 1e6:>18.4f}")

    base_dim, base_ms, base_flops = results[0]
    print(f"\nRelative to dim_in={base_dim} ({base_ms:.2f}ms):")
    for dim, ms, rel_flops in results[1:]:
        flop_ratio = rel_flops / base_flops
        time_ratio = ms / base_ms
        print(
            f"  dim_in={dim}: {flop_ratio:.0f}x the FLOPs, but only {time_ratio:.2f}x the wall-clock time "
            f"({'sub-linear -- overhead-bound at small dim, as hypothesized' if time_ratio < flop_ratio else 'scales with FLOPs'})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
