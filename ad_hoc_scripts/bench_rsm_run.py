"""Times full training steps of a run's config on real batches (nothing is written to the run).

    uv run python ad_hoc_scripts/bench_rsm_run.py --run-name rsm-8L-768-16k [--steps 4] [--batch 1] [--profile]

Prints per step: seconds, tokens / s (every position of the rows), targets / s (positions in the loss), peak
device memory; with --profile, the device time per op of one step (torch profiler).
"""

import argparse
import time

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()

    from model import config as cfg_mod
    from model.device import setup_device

    cfg = cfg_mod.load(a.run_name)
    if a.batch:
        cfg["data"]["batch_size"] = a.batch
    if a.window:
        cfg["arch_params"]["window_size"] = a.window
    setup_device(cfg.get("device"))
    import torch

    from model.data import build_datasets
    from model.archs import build_model
    from model.train import _exclude_from_weight_decay, resolve_optimizer
    from model.losses import SparseCategoricalCrossentropyFromLogits

    model = build_model(cfg)
    model.compile(optimizer=resolve_optimizer(cfg), loss=SparseCategoricalCrossentropyFromLogits(),
                  metrics=["sparse_categorical_accuracy"])
    _exclude_from_weight_decay(model, model.optimizer)
    model.dense_supervision_enabled = True
    model.stateful_training = bool(cfg["training"].get("stateful"))
    model.evaluate(np.zeros((1, 8), np.int32), np.zeros((1,), np.int32), verbose=0)  # builds the metrics
    train, _ = build_datasets(cfg, model.context_length, packed=True)
    print("stateful", model.stateful_training)
    print(f"params {model.count_params():,}; rows per pass {train.batch_size} x {model.context_length}")

    def step(i):
        x, y = train[i]
        model.train_on_batch(x, y)
        return x.size, int((y >= 0).sum())

    step(0)  # warm-up (kernel builds, allocator)
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()
    for i in range(1, a.steps + 1):
        t = time.perf_counter()
        n, k = step(i)
        torch.xpu.synchronize()
        dt = time.perf_counter() - t
        print(f"step {i}: {dt:.2f} s  {n / dt:,.0f} tokens/s  {k / dt:,.0f} targets/s ({k / n:.0%} of positions)  "
              f"peak {torch.xpu.max_memory_allocated() / 2**30:.2f} GiB")
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            step(a.steps + 1)
            torch.xpu.synchronize()
        print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=30))


if __name__ == "__main__":
    main()
