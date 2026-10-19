# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Schedule IR for the gfx950 MLA decode kernel (128 heads, fp8 Q/KV, BLOCK_N=64).

Source: ``mla_fwd_decode_m16x8_fp8_fp8.py`` (gfx950 V3 path).  One work item
walks its KV tiles; per tile:

    barrier(vmcnt 0, lgkmcnt 0)          # tile t landed in lKV[t%2] for every warp
    GEMM1  QK^T: 9 x (8 ds_read K, 8 MFMA) over NoPE+RoPE, and -- interleaved --
           the 18 buffer_load...lds DMAs of tile t+1 into lKV[(t+1)%2]
    softmax (VALU), pack P to fp8
    GEMM2  P*V: 8 x (8 ds_read_b64_tr_b8 V from lKV[t%2], 4 x {wait, swap, 2 MFMA})

K and V share one LDS double buffer ``lKV`` (V is read transposed straight out
of the KV tile).  The last tile bounces its bf16/f32 output through the
*other* slot.  ``gemm1_ops`` / ``gemm2_core_ops`` are emitted by the kernel;
``build_mla_schedule`` composes the whole work item for the verifier and the
cost model.
"""

from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T

NUM_HEADS = 128
QK_NOPE, QK_ROPE, V_DIM = 512, 64, 512
BLOCK_N = 64
NUM_WARPS = 8
MFMA_N, MFMA_K = 16, 32
P_COMP_SUBS = BLOCK_N // MFMA_N  # 4
NUM_NOPE_ITERS = QK_NOPE // (MFMA_K * 2)  # 8
NUM_ROPE_ITERS = QK_ROPE // (MFMA_K * 2)  # 1
NUM_PV_ITERS = V_DIM // (MFMA_N * 2)  # 16
KV_BLOCKS = (QK_NOPE + QK_ROPE) // 64  # 9 col-blocks of 64 fp8
KV_SLOT_BYTES = 264 * (BLOCK_N // 4) * KV_BLOCKS  # V3 layout, 38016
Q_LDS_BYTES = 8 * 1088  # per-warp Q staging, 8704
MFMA = "mfma_f32_16x16x32_fp8_fp8"


def _kv(slot, tile):
    return SlotUse("lKV", slot, T("t0", tile) if tile is not None else None)


def gemm1_ops(cur=0, nxt=1, tile=None, prefetch_tile=None, prefetch=True):
    """QK^T over NoPE then RoPE, with the next tile's DMA interleaved block by block."""
    b = Builder()

    def pf(block):
        if not prefetch:
            return
        for p in (0, 1):  # two 32-row passes cover BLOCK_N=64
            b.op(
                "lds_dma",
                "prefetch_kv",
                args={"block": block, "pass_idx": p},
                count=1,
                queue="dma",
                writes=(_kv(nxt, prefetch_tile),),
                label=f"dma KV[t+1] blk{block} pass{p} -> lKV[{nxt}]",
            )

    def reads(h_tile, col_tile, tag):
        for h in range(P_COMP_SUBS):
            b.op(
                "ds_read",
                "read_k",
                outs=(f"{tag}_{h}",),
                args={"row_offset": 16 * h, "col_offset": col_tile * MFMA_K},
                count=1,
                queue="ds",
                reads=(_kv(cur, tile),),
                out_dwords={f"{tag}_{h}": 2},
                label=f"ds_read K[{16 * h}:,{col_tile * MFMA_K}] lKV[{cur}]",
            )

    def mfmas(tag, q_kind, q_idx, init, first=False):
        for h in range(P_COMP_SUBS):
            ins = (f"{tag}_{h}",) if init else (f"{tag}_{h}", f"p{h}")
            b.op(
                "mfma",
                "qk_mfma",
                ins=ins,
                outs=(f"p{h}",),
                args={"q_kind": q_kind, "q_idx": q_idx},
                count=1,
                mfma=MFMA,
                out_dwords={f"p{h}": 4},
                consumes=(("K", T("t0", tile)),) if (first and h == 0 and tile is not None) else (),
                label=f"mfma S[{h}] += K{tag[1]}.Q{q_kind}[{q_idx}]",
            )

    pf(0)
    for pair in range(NUM_NOPE_ITERS):
        t0, t1 = 2 * pair, 2 * pair + 1
        reads(pair, t0, "k0")
        reads(pair, t1, "k1")
        pf(pair + 1)
        b.sched_barrier(0)
        b.wait(lgkm=P_COMP_SUBS)
        mfmas("k0", "nope", t0, init=(pair == 0), first=(pair == 0))
        if pair == 0:
            b.setprio(15)
        b.wait(lgkm=0)
        mfmas("k1", "nope", t1, init=False)
    for pair in range(NUM_ROPE_ITERS):
        t0, t1 = 2 * pair, 2 * pair + 1
        reads(pair, t0 + 16, "k0")
        reads(pair, t1 + 16, "k1")
        b.sched_barrier(0)
        b.wait(lgkm=P_COMP_SUBS)
        mfmas("k0", "rope", t0, init=False)
        b.wait(lgkm=0)
        mfmas("k1", "rope", t1, init=False)
    b.setprio(14)
    return b.ops


def gemm2_core_ops(cur=0, tile=None):
    """P*V: per PV pair issue 8 transposed V reads, then drain them 2 at a time into 2 MFMAs."""
    b = Builder()
    b.setprio(15)
    for pair in range(NUM_PV_ITERS // 2):
        ia, ib = 2 * pair, 2 * pair + 1
        names = []
        for it, strip in ((ia, ia * MFMA_N * 2), (ib, ib * MFMA_N * 2)):
            for half in (0, 32):
                for part, off in (("top", 0), ("bot", 16)):
                    n = f"v{it}_{half}_{part}"
                    names.append(n)
                    b.op(
                        "ds_read_tr",
                        "read_v_tr",
                        outs=(n,),
                        args={"row_offset": half + off, "col_offset": strip},
                        count=1,
                        queue="ds_tr",
                        reads=(_kv(cur, tile),),
                        out_dwords={n: 2},
                        label=f"ds_read_b64_tr_b8 V[{half + off}:,{strip}] lKV[{cur}]",
                    )
        steps = [(ia, 0, "p_lo"), (ia, 32, "p_hi"), (ib, 0, "p_lo"), (ib, 32, "p_hi")]
        for step, (it, half, p) in enumerate(steps):
            b.sched_barrier(0)
            b.wait(lgkm=(6, 4, 2, 0)[step])
            b.op(
                "valu",
                "v_swap",
                ins=(f"v{it}_{half}_top", f"v{it}_{half}_bot"),
                outs=("lhs0", "lhs1"),
                count=0,
                out_dwords={"lhs0": 2, "lhs1": 2},
            )
            for j, lhs in enumerate(("lhs0", "lhs1")):
                acc = f"o{2 * it + j}"
                b.op(
                    "mfma",
                    "pv_mfma",
                    ins=(lhs, p, acc),
                    outs=(acc,),
                    count=1,
                    mfma=MFMA,
                    out_dwords={acc: 4},
                    consumes=(
                        (("V", T("t0", tile)),) if (pair == 0 and step == 0 and j == 0 and tile is not None) else ()
                    ),
                    label=f"mfma O[{2 * it + j}] += V.{p}",
                )
        b.sched_barrier(0)
        if pair < NUM_PV_ITERS // 2 - 1:
            b.nop(1, asm=False)
    b.setprio(0)
    return b.ops


def _softmax(b, first):
    b.op(
        "valu",
        "softmax",
        ins=tuple(f"p{h}" for h in range(P_COMP_SUBS)) + (() if first else ("row_max", "row_sum")),
        outs=("row_max", "row_sum", "p_lo", "p_hi", "rescale"),
        count=0,
        out_dwords={"row_max": 1, "row_sum": 1, "p_lo": 2, "p_hi": 2, "rescale": 1},
    )


def _acc_names():
    return tuple(f"o{i}" for i in range(2 * NUM_PV_ITERS))


def _tile_ops(t, n, cur, first, last):
    b = Builder()
    b.barrier(fence="after", vm=0, lgkm=0, asm=True, label=f"tile {t}: s_barrier")
    nxt = 1 - cur
    b.extend(gemm1_ops(cur, nxt, tile=t, prefetch_tile=t + 1, prefetch=not last))
    _softmax(b, first)
    if not last:
        if first:
            b.op("valu", "zero_acc", outs=_acc_names(), count=0, out_dwords={o: 4 for o in _acc_names()})
        else:
            b.setprio(3)  # _rescale_oaccu: 4 groups of 8 muls at priority 3,2,1,0
            b.op(
                "valu",
                "rescale_acc",
                ins=_acc_names() + ("rescale",),
                outs=_acc_names(),
                count=0,
                out_dwords={o: 4 for o in _acc_names()},
            )
            b.setprio(0)
        b.barrier(fence="after", lgkm=0, asm=True, label="gemm2: s_barrier")
        b.extend(gemm2_core_ops(cur, tile=t))
        return b.ops
    # last tile: PV with per-pair rescale / normalise and the output bounced through the other KV slot
    b.barrier(fence="after", lgkm=0, asm=True, label="gemm2 last: s_barrier")
    if first:
        b.op("valu", "zero_acc", outs=_acc_names(), count=0, out_dwords={o: 4 for o in _acc_names()})
    core = gemm2_core_ops(cur, tile=t)
    b.extend(core[:-1])
    b.op(
        "ds_write",
        "store_o",
        ins=_acc_names() + ("row_sum",),
        count=2 * NUM_PV_ITERS,
        writes=(SlotUse("lKV", 1 - cur),),
        label=f"O -> lKV[{1 - cur}] -> buffer_store",
    )
    b.setprio(0)
    return b.ops


def build_mla_schedule() -> Schedule:
    def program(n):
        pro = Builder()
        pro.op(
            "vmem_load",
            "load_q",
            outs=("q",),
            count=0,
            out_dwords={"q": 2 * (NUM_NOPE_ITERS + NUM_ROPE_ITERS) * 2},
            note="VRAM->LDS->VGPR shuffle with its own internal waits; drained on return",
        )
        pro.op(
            "lds_dma",
            "load_kv_tile0",
            count=2 * KV_BLOCKS,
            queue="dma",
            writes=(_kv(0, 0),),
            label="dma KV[0] (18 blocks) -> lKV[0]",
        )
        secs = [Section("prologue", pro.ops)]
        for t in range(n):
            secs.append(Section(f"tile{t}", _tile_ops(t, n, t % 2, first=(t == 0), last=(t == n - 1))))
        return secs

    steady = _tile_ops(1, 3, 1, first=False, last=False)
    kv_tile_bytes = BLOCK_N * (QK_NOPE + QK_ROPE)

    def workload(num_seqs, kv_len, num_cus=256):
        tiles_total = num_seqs * -(-kv_len // BLOCK_N)
        per = -(-tiles_total // num_cus)
        return {
            "tiles_per_wg": [per] * min(num_cus, tiles_total),
            "lds_bytes": Q_LDS_BYTES + 2 * KV_SLOT_BYTES,
            "flops": tiles_total * 2 * NUM_HEADS * BLOCK_N * (QK_NOPE + QK_ROPE + V_DIM),
            "hbm_bytes_unique": num_seqs * kv_len * (QK_NOPE + QK_ROPE)
            + num_seqs * NUM_HEADS * ((QK_NOPE + QK_ROPE) + 2 * V_DIM),
            "hbm_bytes_noreuse": tiles_total * kv_tile_bytes + num_seqs * NUM_HEADS * ((QK_NOPE + QK_ROPE) + 2 * V_DIM),
        }

    return Schedule(
        name="mla_decode_m16x8_fp8_gfx950",
        target="gfx950",
        buffers={"lKV": Buffer("lKV", "lds", "KV", 2, KV_SLOT_BYTES, note="K and V share the tile; V read transposed")},
        queues={
            "dma": Queue("dma", "vm", tracked=False),
            "ds": Queue("ds", "lgkm", tracked=True),
            "ds_tr": Queue("ds_tr", "lgkm", tracked=False),
        },
        sections=[Section("tile", steady, loop=("t", lambda b: range(1, 2)))],
        program=program,
        groups=1,
        waves_per_group=NUM_WARPS,
        bases=lambda n: {"t0": 0},
        valid_tiles=lambda n: n >= 1,
        tensors={"K": "GEMM1 reads every KV tile once", "V": "GEMM2 reads every KV tile once"},
        meta={
            "verify_tiles": (1, 2, 3, 4, 5),
            "waves": NUM_WARPS,
            "waves_per_eu": 2,
            "max_wg_per_cu": 1,
            "loop_tiles_per_iter": 1,
            "lds_extra_bytes": Q_LDS_BYTES,
            "workload": workload,
            "value_dwords": {"q": 36},
            "expert_intent": [
                "next KV tile streamed into the other LDS slot one 64-col block per GEMM1 step, so DMA hides under MFMA",
                "V read with ds_read_b64_tr_b8 straight from the KV tile: no Vt staging buffer on gfx950",
                "lgkmcnt drained 2 reads at a time so each MFMA pair starts as soon as its operands land",
                "s_setprio 15 around the MFMA streams, descending 3..0 across the O rescale",
            ],
        },
    )
