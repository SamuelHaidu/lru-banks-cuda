"""Where a `min-gru` training step spends its time, on real agent-mix batches through the real train step (keras
AdamW, fused readout loss on the assistant targets only). Writes nothing.

    uv run python ad_hoc_scripts/profile_min_gru.py [--dataset agent-mix-bpe1024] [--batch 16] [--seq 1024]
        [--set hidden_size=256 num_layers=4 ...] [--device cuda] [--arch min-gru-cuda]

Prints:
  1. the step split into forward (to the loss), backward and optimizer (synchronized wall clock);
  2. each part of the model timed alone at the step's shapes, forward + backward;
  3. the profiler's device time per op (all of the step).
"""

import argparse
import sys
import copy
import json
import time

import numpy as np

from model.device import setup_backend, setup_device

setup_backend()
DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "xpu"  # before torch
setup_device(DEVICE)

import torch  # noqa: E402

from model import config as cfg_mod  # noqa: E402
from model.archs import arch_defaults, build_model  # noqa: E402
from model.archs import min_gru as mg  # noqa: E402
from model.data import build_datasets  # noqa: E402
from model.fused_linear_cross_entropy import fused_linear_cross_entropy  # noqa: E402
from model.losses import SparseCategoricalCrossentropyFromLogits  # noqa: E402
from model.train import _exclude_from_weight_decay, resolve_optimizer  # noqa: E402

sync = getattr(torch, DEVICE).synchronize


def timed(f, n=10):
    f()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        f()
    sync()
    return (time.perf_counter() - t) / n * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="agent-mix-bpe1024")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--set", nargs="*", default=[], help="arch params, key=json value")
    p.add_argument("--device", default="xpu", help="xpu, cuda or cpu")
    p.add_argument("--arch", default="min-gru", help="min-gru or min-gru-cuda")
    p.add_argument("--step-only", action="store_true", help="only the train step time (best of 3 runs of 10 steps)")
    a = p.parse_args()

    meta = cfg_mod.load_dataset_metadata(a.dataset)
    config = copy.deepcopy(cfg_mod.DEFAULT_CONFIG)
    config.update(run_name="profile", arch=a.arch, device=DEVICE)
    for section, values in mg.MinGRULM.config_defaults.items():
        config[section].update(values)
    config["arch_params"] = {**arch_defaults(a.arch), "vocabulary_size": meta["vocabulary_size"],
                             "window_size": a.seq, **{k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}}
    config["data"].update(dataset=a.dataset, batch_size=a.batch)
    model = build_model(config)
    model.compile(optimizer=resolve_optimizer(config), loss=SparseCategoricalCrossentropyFromLogits(),
                  metrics=["sparse_categorical_accuracy"])
    _exclude_from_weight_decay(model, model.optimizer)
    model.dense_supervision_enabled = model.dense_validation_enabled = True
    model.evaluate(np.zeros((1, a.seq), np.int32), np.zeros((1,), np.int32), verbose=0)
    train, _ = build_datasets(config, a.seq, packed=True, static_memory=model.static_memory_spec)
    batches = [train[i] for i in range(8)]
    frac = np.mean([(y >= 0).mean() for _, y in batches])
    ap = config["arch_params"]
    tokens = a.batch * a.seq
    print(f"min-gru dim {ap['dim']} hidden {ap['hidden_size']} layers {ap['num_layers']}  rows {a.batch} x {a.seq}  "
          f"targets {frac:.0%} of positions  params {model.count_params():,}")

    # 1. the real step, then split
    i = [0]

    def step():
        model.train_step(batches[i[0] % len(batches)])
        i[0] += 1

    t_step = timed(step)
    if a.step_only:
        t_step = min([t_step] + [timed(step) for _ in range(2)])
        print(f"train step {t_step:7.1f} ms   {tokens / t_step * 1e3:9,.0f} tokens/s")
        return
    print(f"\n== train step {t_step:7.1f} ms   {tokens / t_step * 1e3:9,.0f} tokens/s")
    x, y = batches[0]
    y = torch.as_tensor(y, device=DEVICE).long()
    valid = y >= 0
    weights = [v.value for v in model.trainable_weights]

    def forward():
        h = model.dense_hidden_valid(x, valid)
        return fused_linear_cross_entropy(model.readout_input(h), model.readout.kernel, y[valid])

    def fwd_bwd():
        for w in weights:
            w.grad = None
        forward().backward()

    t_fwd = timed(lambda: forward().detach())
    t_fb = timed(fwd_bwd)
    grads = [w.grad for w in weights]
    t_opt = timed(lambda: model.optimizer.apply(grads, model.trainable_weights))
    print(f"   forward to the loss {t_fwd:7.1f} ms")
    print(f"   backward            {t_fb - t_fwd:7.1f} ms")
    print(f"   optimizer (keras)   {t_opt:7.1f} ms")
    print(f"   rest (keras step)   {t_step - t_fb - t_opt:7.1f} ms   (metrics, accuracy readout, bookkeeping)")

    # 2. the parts alone at the step's shapes, forward + backward (per layer unless noted)
    D, H, L = ap["dim"], ap["hidden_size"], ap["num_layers"]
    B, T = a.batch, a.seq
    R = int(valid.sum())
    dev = DEVICE
    rnd = lambda *s: torch.randn(*s, device=dev, requires_grad=True)
    res, w_in, w_out = rnd(B, T, D), rnd(D, 2 * H), rnd(H, D)
    norm_w, hn_w, scale = rnd(D), rnd(H), rnd(D)
    gl, raw, st = rnd(B, T, H), rnd(B, T, H), rnd(B, T, H)
    not_reset = torch.ones(B, T, device=dev)
    keep = torch.rand(B, T, H, device=dev, requires_grad=True)
    write = rnd(B, T, H)
    rows = valid.reshape(-1).nonzero().squeeze(1)

    def fb(f):
        def run():
            out = f()
            out.backward(torch.ones_like(out))
        return timed(run)

    parts = [
        ("rms_norm(stream)", lambda: mg.rms_norm(res)),
        ("in_proj matmul (folded norm weight)", lambda: res.detach() @ (norm_w[:, None] * w_in)),
        ("gates: sigmoid x2, tanh, products", lambda: torch.sigmoid(gl) * torch.tanh(raw)
         + torch.sigmoid(-gl) * not_reset[..., None]),
        ("linear_scan", lambda: mg.linear_scan(keep, write)),
        ("rms_norm(state)", lambda: mg.rms_norm(st)),
        ("out_proj matmul", lambda: st.detach() @ (hn_w[:, None] * w_out)),
        ("write: tanh * scale + add", lambda: res + scale * torch.tanh(res.detach())),
    ]
    total = 0.0
    print(f"\n== parts alone, forward + backward ({L} layers)")
    for name, f in parts:
        ms = fb(f)
        total += ms * L
        print(f"   {name:38s} {ms:7.2f} ms  x{L} = {ms * L:7.1f} ms")
    emb = model.token_embedding.embeddings.value
    ids = torch.as_tensor(x["ids"] if isinstance(x, dict) else x, device=dev).long().reshape(-1)
    e_ms = fb(lambda: emb.index_select(0, ids))
    print(f"   {'embedding gather':38s} {e_ms:7.2f} ms")
    hv, k = rnd(R, D), model.readout.kernel.value
    ce_ms = timed(lambda: fused_linear_cross_entropy(mg.rms_norm(hv, model.final_norm.value), k, y[valid]).backward())
    print(f"   {'final norm + readout loss (' + str(R) + ' rows)':38s} {ce_ms:7.2f} ms")
    print(f"   sum of parts {total + e_ms + ce_ms:7.1f} ms vs forward + backward {t_fb:7.1f} ms")

    # 3. profiler: device time per op over 3 steps
    from torch.profiler import ProfilerActivity, profile

    sync()
    with profile(activities=[ProfilerActivity.CPU, getattr(ProfilerActivity, DEVICE.upper())]) as prof:
        for _ in range(3):
            step()
        sync()
    rows_ = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
    tot = sum(e.self_device_time_total for e in rows_ if e.key.startswith("aten::")) or 1
    print("\n== device time per op (aten ops, per step)")
    for e in [e for e in rows_ if e.key.startswith("aten::")][:16]:
        print(f"   {e.key:28s} {e.self_device_time_total / 3e3:7.1f} ms  {e.self_device_time_total / tot:5.1%}  "
              f"calls/step {e.count / 3:6.0f}")
    print(f"   device busy {tot / 3e3:.1f} ms per step of {t_step:.1f} ms wall")


if __name__ == "__main__":
    main()
