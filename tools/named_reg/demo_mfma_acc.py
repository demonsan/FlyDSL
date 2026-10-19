#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Small pin demo: an MFMA accumulation loop whose accumulator is pinned to a register range.

Every block (one wave) accumulates ``steps`` bf16 32x32x16 MFMAs into one f32x16
accumulator per lane and stores it.  The kernel body is emitted from a Schedule IR
whose loop carries ``acc``; ``--pin v[128:143]`` pins it (``Section.pins``), which
``flydsl.sched.program.emit_program`` lowers to pin markers.  With the stock
toolchain the markers are no-ops; ``tools/named_reg/named_reg.py`` turns them into
named-register intrinsics.

    python tools/named_reg/demo_mfma_acc.py --pin v[128:143] [--bench]

Checks the result against a torch reference (fp32 matmul of the same fragments).
"""

import argparse
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.sched.ir import Builder, Queue, Schedule, Section
from flydsl.sched.ir import T as Tile
from flydsl.sched.program import emit_program, schedule_fingerprint
from flydsl.sched.verify import verify
from kernels.common import buffer_ops

LANES = 64
# shape -> (M=N, accumulator dwords per lane)
SHAPES = {"32x32x16": (32, 16), "16x16x32": (16, 4)}


def build_schedule(pin=None, shape="32x32x16") -> Schedule:
    acc_dw = SHAPES[shape][1]
    pro = Builder()
    pro.op("valu", "init", outs=("acc",), out_dwords={"acc": acc_dw})
    body = Builder()
    body.op("vmem_load", "load_a", outs=("a",), args={"k": Tile("k")}, queue="gld", out_dwords={"a": 4})
    body.op("vmem_load", "load_b", outs=("b",), args={"k": Tile("k")}, queue="gld", out_dwords={"b": 4})
    body.op(
        "mfma", "mma", ins=("a", "b", "acc"), outs=("acc",), mfma=f"mfma_f32_{shape}_bf16", out_dwords={"acc": acc_dw}
    )
    ep = Builder()
    ep.op("vmem_store", "store", ins=("acc",), count=acc_dw // 4)
    loop = Section("k", body.ops, loop=("k", lambda bases: range(0, 4)), carried=("acc",))
    if pin:
        loop.pins = {"acc": pin}
    return Schedule(
        name=f"demo_mfma_acc_{shape}" + ("_pinned" if pin else ""),
        target="gfx950",
        buffers={},
        queues={"gld": Queue("gld", "vm", tracked=True)},
        sections=[Section("prologue", pro.ops), loop, Section("epilogue", ep.ops)],
        meta={"verify_tiles": (4,), "waves_per_eu": 1},
    )


def build_kernel(sched: Schedule, shape="32x32x16"):
    acc_dw = SHAPES[shape][1]
    mfma = getattr(rocdl, f"mfma_f32_{shape}_bf16")
    report = verify(sched)  # includes the PIN.* rules
    if not report.ok:
        raise ValueError(str(report))
    cache_tag = (sched.name, schedule_fingerprint(sched))

    @flyc.kernel(known_block_size=[LANES, 1, 1])
    def mfma_acc_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, steps: fx.Int32):
        lane = fx.Index(fx.thread_idx.x)
        blk = fx.Index(fx.block_idx.x)
        a_rsrc = buffer_ops.create_buffer_resource(A)
        b_rsrc = buffer_ops.create_buffer_resource(B)
        c_rsrc = buffer_ops.create_buffer_resource(C)
        n_steps = fx.Index(steps)
        acc_ty = Vec.make_type(acc_dw, fx.Float32)

        def frag(rsrc, k):
            elem = ((blk * n_steps + k) * LANES + lane) * 8
            return buffer_ops.buffer_load(rsrc, fx.Int32(elem), vec_width=8, dtype=T.bf16)

        def store(acc):
            v = Vec(acc, (acc_dw,), fx.Float32)
            for q in range_constexpr(acc_dw // 4):
                part = v.shuffle(v, [4 * q + i for i in range(4)]).ir_value()
                buffer_ops.buffer_store(part, c_rsrc, fx.Int32((blk * LANES + lane) * acc_dw + 4 * q))

        bindings = {
            "init": lambda: Vec.filled(acc_dw, 0.0, fx.Float32),
            "load_a": lambda k: frag(a_rsrc, k),
            "load_b": lambda k: frag(b_rsrc, k),
            "mma": lambda a, b, acc: mfma(acc_ty, [a, b, acc, 0, 0, 0]),
            "store": store,
        }

        def tile_of(extra):
            return lambda ref: extra[ref.base] + ref.off if ref.off else extra[ref.base]

        emit_program(
            sched, bindings, tile_of, loop_bounds=lambda name: (fx.Index(0), n_steps, fx.Index(1)), tag=sched.name
        )

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, steps: fx.Int32, blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ):
        _ = cache_tag
        mfma_acc_kernel(A, B, C, steps).launch(grid=(fx.Index(blocks), 1, 1), block=(LANES, 1, 1), stream=stream)

    return launch


def reference(A, B, blocks, steps, shape="32x32x16"):
    """fp32 sum over steps of A_k (MxK) @ B_k (KxM), in the per-lane accumulator layout of the MFMA."""
    m, acc_dw = SHAPES[shape]
    g = LANES // m  # lane groups; K = 8 * g
    a = A.view(blocks, steps, g, m, 8).float()  # lane = grp * m + row, element = k within the group
    b = B.view(blocks, steps, g, m, 8).float()
    am = a.permute(0, 1, 3, 2, 4).reshape(blocks, steps, m, 8 * g)  # [row][k]
    bm = b.permute(0, 1, 2, 4, 3).reshape(blocks, steps, 8 * g, m)  # [k][col]
    c = torch.einsum("bsrk,bskc->brc", am, bm)  # [block][row][col]
    out = torch.empty(blocks, LANES, acc_dw, device=A.device)
    for lane in range(LANES):
        col, grp = lane % m, lane // m
        for i in range(acc_dw):
            out[:, lane, i] = c[:, (i // 4) * 4 * g + grp * 4 + i % 4, col]
    return out


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
    return ts[len(ts) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=os.environ.get("DEMO_PIN") or None)
    ap.add_argument("--blocks", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="32x32x16")
    ap.add_argument("--bench", type=int, default=0, help="timed iterations (0: correctness only)")
    args = ap.parse_args(argv)
    sched = build_schedule(args.pin, args.shape)
    launch = build_kernel(sched, args.shape)
    m, acc_dw = SHAPES[args.shape]
    torch.manual_seed(0)
    n = args.blocks * args.steps * LANES * 8
    A = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(args.blocks * LANES * acc_dw, device="cuda", dtype=torch.float32)
    launch(A, B, C, args.steps, args.blocks)
    torch.cuda.synchronize()
    ref = reference(A, B, args.blocks, args.steps, args.shape).reshape(-1)
    err = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, atol=1e-2, rtol=1e-3)
    print(f"{sched.name}: max_abs_err={err:.3e} {'PASS' if ok else 'FAIL'}")
    if args.bench:
        us = bench(lambda: launch(A, B, C, args.steps, args.blocks), args.bench)
        flops = 2 * m * m * (8 * LANES // m) * args.steps * args.blocks
        print(f"BENCH {sched.name} {us:.2f} us {flops / us / 1e6:.1f} TFLOP/s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
