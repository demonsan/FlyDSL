#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Run the schedule-emitted gfx950 MHA with register pins (``mha_gfx950_pinned``): check and time it.

    python tools/named_reg/mha_pinned_run.py --pins o [--bench 50]

Correctness: a small shape against an fp32 reference and the timed shape against
``torch.nn.functional.scaled_dot_product_attention``.  Use through
``tools/named_reg/named_reg.py`` (prepare / build / run) to execute the binary the pin
branch's ``llc`` produced; run directly, the pin markers are no-ops.
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from kernels.attention.flash_attn_sched_gfx950 import build_flash_attn_sched_gfx950_module
from kernels.attention.sched.mha_gfx950_pinned import build_pinned_mha_schedule


def _ref_fp32(q, k, v):
    qf, kf, vf = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    s = torch.einsum("bhqd,bhkd->bhqk", qf, kf) / math.sqrt(q.shape[-1])
    return torch.einsum("bhqk,bhkd->bhqd", s.softmax(-1), vf).permute(0, 2, 1, 3)


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default="o", help="preset of mha_gfx950_pinned.PRESETS, 'none', or 'value=range,...'")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--shape", default="4,8192,64,128", help="B,S,H,D of the timed run")
    ap.add_argument("--waves-per-eu", type=int, default=2)
    args = ap.parse_args(argv)
    B, S, H, D = map(int, args.shape.split(","))
    pins = args.pins
    if "=" in pins:  # "v_o=v[224:255],m_row=v190"
        pins = dict(kv.split("=", 1) for kv in pins.split(","))
    sched = None if pins == "none" else build_pinned_mha_schedule(pins, head_dim=D, causal=False)
    kern = build_flash_attn_sched_gfx950_module(
        H, D, causal=False, dtype_str="bf16", schedule=sched, waves_per_eu=args.waves_per_eu
    )
    name = (sched.name if sched is not None else "base") + f"_wpe{args.waves_per_eu}"

    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1024, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = torch.empty_like(q)
    kern(q, k, v, out, 1, 1024)
    torch.cuda.synchronize()
    err_small = (out.float() - _ref_fp32(q, k, v)).abs().max().item()

    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = torch.empty_like(q)
    kern(q, k, v, out, B, S)
    ref = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    ok = err_small < 2e-2 and err < 2e-2 and bool(torch.isfinite(out).all())
    print(f"{name}: max_abs_err small={err_small:.3e} B{B}S{S}H{H}D{D}={err:.3e} {'PASS' if ok else 'FAIL'}")
    if args.bench:
        med, lo, hi = bench(lambda: kern(q, k, v, out, B, S), args.bench)
        flops = 4 * B * H * S * S * D
        print(f"BENCH {name} median {med:.1f} us (min {lo:.1f} max {hi:.1f}) {flops / med / 1e6:.0f} TFLOP/s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
