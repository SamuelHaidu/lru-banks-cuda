"""Linear scan state[t] = keep[t] state[t-1] + write[t] at a training shape: the chunked scan (`min_gru.linear_scan`)
vs parallel associative scans (Hillis-Steele in eager torch; torch's `associative_scan` compiled to one Triton
kernel). Forward and forward + backward, and the max difference to the chunked scan.

    python "ad hoc scripts/bench_scan.py" [--device cuda] [--shape 32 512 256]
"""

import argparse
import sys
import time

from model.device import setup_device

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
setup_device(DEVICE)

import torch  # noqa: E402

from model.archs.min_gru import linear_scan  # noqa: E402


def hillis_steele(keep, write):
    """Inclusive scan of (keep, write) pairs along dim 1: log2(T) steps, each combining position t with t - 2^k:
    (a, b) <- (a_t a_{t-s}, a_t b_{t-s} + b_t)."""
    a, b = keep, write
    T, s = keep.shape[1], 1
    while s < T:
        b = torch.cat([b[:, :s], torch.addcmul(b[:, s:], a[:, s:], b[:, :-s])], 1)
        a = torch.cat([a[:, :s], a[:, s:] * a[:, :-s]], 1)
        s *= 2
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--shape", type=int, nargs=3, default=[32, 512, 256])
    a = p.parse_args()
    B, T, D = a.shape
    sync = getattr(torch, a.device).synchronize
    g = torch.Generator(device=a.device).manual_seed(0)
    keep = torch.rand(B, T, D, device=a.device, generator=g) * 0.5 + 0.5
    write = torch.randn(B, T, D, device=a.device, generator=g) * 0.1

    def timed(f, n=20):
        f()
        sync()
        t = time.perf_counter()
        for _ in range(n):
            f()
        sync()
        return (time.perf_counter() - t) / n * 1e3

    def fb(f):
        k, w = keep.clone().requires_grad_(), write.clone().requires_grad_()
        return lambda: f(k, w).sum().backward()

    ref = linear_scan(keep, write)
    scans = {"chunked (min_gru.linear_scan)": linear_scan, "hillis-steele (eager)": hillis_steele}
    try:
        from torch._higher_order_ops.associative_scan import associative_scan

        def combine(x, y):
            return x[0] * y[0], y[0] * x[1] + y[1]

        @torch.compile(fullgraph=True)
        def compiled(keep, write):
            return associative_scan(combine, (keep, write), dim=1, combine_mode="pointwise")[1]

        scans["associative_scan (compiled)"] = compiled
    except Exception as e:  # noqa: BLE001
        print("associative_scan unavailable:", repr(e)[:200])
    for name, f in scans.items():
        try:
            err = float((f(keep, write) - ref).abs().max())
            fwd = timed(lambda: f(keep, write))
            both = timed(fb(f))
            print(f"{name:32s} fwd {fwd:7.3f} ms   fwd+bwd {both:7.3f} ms   max diff {err:.1e}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:32s} failed: {repr(e)[:300]}")


if __name__ == "__main__":
    main()
