# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Schedule IR for the gfx950 dual-wave software-pipelined flash-attention forward.

This is the hand schedule of ``flash_attn_gfx950.py`` (dense path: no paging,
bias, ALiBi, sink, split-K or varlen) written as data:

* Resources: K and V each double-buffered in LDS (``lK[0..1]``, ``lV[0..1]``),
  filled by ``buffer_load ... lds`` DMA on the vm counter (not tracked by the
  compiler); K read to VGPRs with ``ds_read_b128`` (lgkm, compiler-tracked), V
  read transposed with inline-asm ``ds_read_b64_tr_b16`` (lgkm, untracked).
* Execution: two wave groups run the same program one cluster apart
  (``stagger``), so each SIMD time-multiplexes a "memory" cluster of one group
  with a "compute" cluster of the other.  Every cluster ends with an
  ``s_barrier``.
* Instruction placement inside compute clusters is pinned with
  ``sched_group_barrier`` interleaves {1 MFMA, n VALU/TRANS}.

``build_mha_schedule`` returns the schedule; ``kernels/attention/
flash_attn_sched_gfx950.py`` emits a kernel from it and ``flydsl.sched``
verifies / costs it without a GPU.
"""

from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T

BLOCK_M = 256
BLOCK_N = 64
NUM_WAVES = 8
WARP_SIZE = 64
MFMA = {"bf16": "mfma_f32_32x32x16_bf16", "f16": "mfma_f32_32x32x16_f16"}


def mha_geometry(head_dim: int) -> dict:
    """Compile-time geometry; mirrors ``_make_dualwave_swp_traits`` (checked on GPU builds)."""
    if head_dim not in (64, 128):
        raise ValueError(f"dual-wave MHA supports head_dim 64/128, got {head_dim}")
    bf16 = 2
    smem_linear_wave = WARP_SIZE * 16 // bf16
    smem_n_rpt = BLOCK_N // (smem_linear_wave // 64)
    smem_d_rpt = head_dim // 64
    k_tile = smem_n_rpt * smem_d_rpt * (smem_linear_wave + 16 // bf16)
    v_tile = smem_n_rpt * smem_d_rpt * (smem_linear_wave + 64 // bf16)
    return {
        "K_STEPS_QK": head_dim // 16,
        "D_CHUNKS": head_dim // 32,
        "NUM_DMA": smem_d_rpt,  # buffer_load ... lds per wave per K or V tile
        "K_TILE_BYTES": k_tile * bf16,
        "V_TILE_BYTES": v_tile * bf16,
    }


# Bindings the schedule-driven kernel (flash_attn_sched_gfx950.py) implements.  A schedule
# that needs anything else is rejected before tracing, with the missing names listed.
KERNEL_BINDINGS = frozenset(
    {
        "load_k", "load_v", "read_k", "read_v", "load_q", "scale_q", "qk", "pv_step", "pv",
        "causal_mask_prologue", "causal_mask", "seq_pad_mask", "to_lists", "reduce_max", "floor_masked_max",
        "sub_m", "exp2", "reduce_sum", "cast_p", "anchor_p", "anchor_o", "lazy_rescale_o", "rescale_o",
        "rescale_from_tile_max", "apply_l_rescale", "scale_o", "normalize", "init_acc", "stagger_open",
        "stagger_close", "store_o",
    }
)  # fmt: skip


def _pairs(head_dim: int, pairs: int) -> int:
    """``_scale_sched_pairs``: D=64 halves the interleave length."""
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs


def build_mha_schedule(
    head_dim: int = 128,
    causal: bool = True,
    dtype: str = "bf16",
    lazy_rescale: bool = True,
    setprio: bool = True,
    stagger: bool = True,
) -> Schedule:
    g = mha_geometry(head_dim)
    KS, DC, NDMA = g["K_STEPS_QK"], g["D_CHUNKS"], g["NUM_DMA"]
    mfma = MFMA[dtype]

    def P(n):
        return _pairs(head_dim, n)

    # ---- op vocabulary ---------------------------------------------------------
    def load_k(b, tile, slot):
        b.op(
            "lds_dma",
            "load_k",
            args={"tile": tile, "buf": slot},
            count=NDMA,
            queue="dma",
            writes=(SlotUse("lK", slot, tile),),
            label=f"dma K[{tile}]->lK[{slot}]",
        )

    def load_v(b, tile, slot):
        b.op(
            "lds_dma",
            "load_v",
            args={"tile": tile, "buf": slot},
            count=NDMA,
            queue="dma",
            writes=(SlotUse("lV", slot, tile),),
            label=f"dma V[{tile}]->lV[{slot}]",
        )

    def read_k(b, slot, tile):
        b.op(
            "ds_read",
            "read_k",
            outs=("v_k",),
            args={"buf": slot},
            count=2 * KS,
            queue="ds",
            reads=(SlotUse("lK", slot, tile),),
            out_dwords={"v_k": 8 * KS},
            label=f"ds_read lK[{slot}]",
        )

    def read_v(b, slot, tile, out="v_v"):
        b.op(
            "ds_read_tr",
            "read_v",
            outs=(out,),
            args={"buf": slot},
            count=8 * DC,
            queue="ds_tr",
            reads=(SlotUse("lV", slot, tile),),
            out_dwords={out: 16 * DC},
            label=f"ds_read_tr lV[{slot}]",
        )

    def qk(b, out, tile):
        b.op(
            "mfma",
            "qk",
            ins=("v_k", "q"),
            outs=(out,),
            count=2 * KS,
            mfma=mfma,
            consumes=(("K", tile),),
            out_dwords={out: 32},
            label=f"QK^T -> {out}",
        )

    def pv(b, step, p, v="v_v", tile=None):
        b.op(
            "mfma",
            "pv_step",
            ins=(p, v, "v_o"),
            outs=("v_o",),
            args={"step": step},
            count=DC,
            mfma=mfma,
            consumes=(("V", tile),) if tile is not None else (),
            out_dwords={"v_o": 16 * DC},
            label=f"PV step{step} ({p})",
        )

    def pv_all(b, p, v, tile):
        b.op(
            "mfma",
            "pv",
            ins=(p, v, "v_o"),
            outs=("v_o",),
            count=4 * DC,
            mfma=mfma,
            consumes=(("V", tile),),
            out_dwords={"v_o": 16 * DC},
            label=f"PV ({p})",
        )

    # per-lane dwords of VALU results whose size depends on the producer:
    # P is f32 (2 x 16 dwords) out of exp2 and bf16 (2 x 2 packs x 4 dwords) after cast_p.
    SIZES = {"cast_p": 16, "exp2": 32, "scale_q": 4 * KS}

    def valu(b, binding, ins, outs, kind="valu", **args):
        d = SIZES.get(binding)
        b.op(
            kind,
            binding,
            ins=ins,
            outs=outs,
            args=args,
            count=0,
            out_dwords={o: d for o in outs} if d is not None else {},
        )

    def mask(b, s, tile, end_tile):
        if causal:
            valu(b, "causal_mask", (s,), (s,), tile=tile, end_tile=end_tile)
        else:
            valu(b, "to_lists", (s,), (s,))

    def finish_p(b, p, group):
        """C1/C5 tail: second half of exp2, row sum, bf16 cast, anchor, interleave."""
        valu(b, "exp2", (p,), (p,), kind="trans", start=16, length=16)
        valu(b, "reduce_sum", ("l_row", p), ("l_row",))
        valu(b, "cast_p", (p,), (p,))
        b.anchor("anchor_p", (p,), (p,))
        b.interleave(P(6), "trans", 3, group)
        b.interleave(P(10), "valu", 5, group)

    def mem_cluster(b, issue, read, wait_vm):
        b.nop(7)
        b.sched_barrier(0)
        issue(b)
        read(b)
        b.wait(lgkm=0)
        b.wait(vm=wait_vm)
        b.barrier()

    def pv_cluster(b, p_in, p_out, s_in, tmax, v_tile, group_a, group_b, s_to_lists=False):
        """C3/C7: P*V with the online-softmax rescale threaded between the 4 PV steps."""
        if setprio:
            b.setprio(1)
        pv(b, 0, p_in, tile=v_tile)
        if s_to_lists:
            valu(b, "to_lists", (s_in,), (s_in,))
        valu(b, "reduce_max", (s_in,), (tmax,))
        b.interleave(P(4), "valu", 6, group_a)
        valu(
            b,
            "lazy_rescale_o" if lazy_rescale else "rescale_o",
            ("v_o", "m_row", "l_row", tmax, p_in),
            ("v_o", "m_row", "l_row", p_in),
        )
        for step in (1, 2, 3):
            pv(b, step, p_in)
        valu(b, "sub_m", (s_in, "m_row"), (s_in,))
        valu(b, "exp2", (s_in,), (p_out,), kind="trans", start=0, length=16)
        b.interleave(P(6), "valu", group_b[0], group_b[1])
        b.interleave(P(6), "trans", 3, group_b[1])
        if setprio:
            b.setprio(0)
        b.barrier()

    # ---- prologue ----------------------------------------------------------------
    pro = Builder()
    load_k(pro, T("first"), 0)
    pro.wait(vm=0, lgkm=0, exp=0)
    pro.barrier(fence="before")
    pro.op("vmem_load", "load_q", outs=("q",), count=KS, queue="gld", out_dwords={"q": 4 * KS})  # 8 bf16/pack
    valu(pro, "scale_q", ("q",), ("q",))
    load_k(pro, T("first", 1), 1)
    load_v(pro, T("first"), 0)
    read_k(pro, 0, T("first"))
    pro.sched_barrier(0)
    pro.wait(lgkm=0)
    pro.wait(vm=NDMA)
    pro.stagger("open")
    qk(pro, "v_s0", T("first"))
    pro.sched_barrier(0)
    if causal:
        valu(pro, "causal_mask_prologue", ("v_s0",), ("v_s0",))
    else:
        valu(pro, "seq_pad_mask", ("v_s0",), ("v_s0",), tile=T("first"))
    valu(pro, "reduce_max", ("v_s0",), ("m_row",))
    if causal:
        valu(pro, "floor_masked_max", ("m_row",), ("m_row",))
    valu(pro, "sub_m", ("v_s0", "m_row"), ("v_s0",))
    valu(pro, "exp2", ("v_s0",), ("v_p0",), kind="trans", start=0, length=16)
    pro.barrier()
    load_k(pro, T("first", 2), 0)
    valu(pro, "init_acc", (), ("l_row", "v_o"))

    # ---- steady state: two KV tiles per iteration, 8 clusters ----------------------
    lp = Builder()
    mem_cluster(lp, lambda b: load_v(b, T("j", -2), 1), lambda b: read_k(b, 1, T("j", -2)), 2 * NDMA)  # C0
    qk(lp, "v_s1", T("j", -2))  # C1
    finish_p(lp, "v_p0", 1)
    lp.barrier()
    mem_cluster(lp, lambda b: load_k(b, T("j"), 1), lambda b: read_v(b, 0, T("j", -3)), 2 * NDMA)  # C2
    pv_cluster(lp, "v_p0", "v_p1", "v_s1", "m_tile", T("j", -3), 2, (6, 2), s_to_lists=True)  # C3
    mem_cluster(lp, lambda b: load_v(b, T("j", -1), 0), lambda b: read_k(b, 0, T("j", -1)), 2 * NDMA)  # C4
    qk(lp, "v_s0", T("j", -1))  # C5
    finish_p(lp, "v_p1", 3)
    lp.barrier()

    def c6_read(b):
        read_v(b, 1, T("j", -2))
        mask(b, "v_s0", T("j", -1), T("j"))

    mem_cluster(lp, lambda b: load_k(b, T("j", 1), 0), c6_read, 2 * NDMA)  # C6
    pv_cluster(lp, "v_p1", "v_p0", "v_s0", "m_tile", T("j", -2), 4, (5, 4))  # C7

    # ---- epilogue: drain the last three tiles --------------------------------------
    ep = Builder()
    mem_cluster(ep, lambda b: load_v(b, T("end", -3), 1), lambda b: read_k(b, 1, T("end", -3)), 2 * NDMA)  # E0
    qk(ep, "v_s1", T("end", -3))  # E1
    finish_p(ep, "v_p0", 5)
    ep.barrier()

    def e2_read(b):
        read_v(b, 0, T("end", -4), out="v_v")
        if causal:
            mask(b, "v_s1", T("end", -3), T("end", -2))
        else:
            valu(b, "seq_pad_mask", ("v_s1",), ("v_s1",), tile=T("end", -3))

    mem_cluster(ep, lambda b: load_k(b, T("end", -1), 1), e2_read, 2 * NDMA)  # E2

    def pv_full_cluster(b, p_in, p_out, s_in, v_tile, group):
        """E3/E7/E11 head: full P*V, then the unconditional rescale of the new tile."""
        if setprio:
            b.setprio(1)
        pv_all(b, p_in, "v_v", v_tile)
        valu(b, "reduce_max", (s_in,), ("m_tile",))
        valu(b, "rescale_from_tile_max", ("m_row", "m_tile"), ("m_row", "rescale"))
        valu(b, "sub_m", (s_in, "m_row"), (s_in,))
        valu(b, "exp2", (s_in,), (p_out,), kind="trans", start=0, length=16)
        b.interleave(P(10), "valu", 5, group)
        b.interleave(P(6), "trans", 3, group)
        b.sched_barrier(0)
        valu(b, "scale_o", ("v_o", "rescale"), ("v_o",))
        b.anchor("anchor_o", ("v_o",), ("v_o",))

    pv_full_cluster(ep, "v_p0", "v_p1", "v_s1", T("end", -4), 6)  # E3
    if setprio:
        ep.setprio(0)
    ep.barrier()
    mem_cluster(ep, lambda b: load_v(b, T("end", -2), 0), lambda b: read_k(b, 0, T("end", -2)), 2 * NDMA)  # E4
    qk(ep, "v_s0", T("end", -2))  # E5
    valu(ep, "apply_l_rescale", ("l_row", "rescale"), ("l_row",))
    finish_p(ep, "v_p1", 7)
    ep.barrier()

    def e6_body(b):
        read_v(b, 1, T("end", -3))
        if causal:
            mask(b, "v_s0", T("end", -2), T("end", -1))
        else:
            valu(b, "seq_pad_mask", ("v_s0",), ("v_s0",), tile=T("end", -2))
        b.wait(lgkm=0)
        b.wait(vm=NDMA)
        b.barrier()

    e6_body(ep)  # E6 (no new DMA)
    pv_full_cluster(ep, "v_p1", "v_p0", "v_s0", T("end", -3), 8)  # E7
    if setprio:
        ep.setprio(0)
    ep.barrier()
    mem_cluster(ep, lambda b: load_v(b, T("end", -1), 1), lambda b: read_k(b, 1, T("end", -1)), NDMA)  # E8
    qk(ep, "v_s1", T("end", -1))  # E9
    valu(ep, "apply_l_rescale", ("l_row", "rescale"), ("l_row",))
    finish_p(ep, "v_p0", 9)
    ep.barrier()
    read_v(ep, 0, T("end", -2))  # E10
    if causal:
        mask(ep, "v_s1", T("end", -1), T("end"))
    else:
        valu(ep, "seq_pad_mask", ("v_s1",), ("v_s1",), tile=T("end", -1))
    ep.wait(lgkm=0)
    ep.wait(vm=0)
    ep.barrier()
    # E11: last rescale, finish the last tile's softmax in place
    pv_all(ep, "v_p0", "v_v", T("end", -2))
    valu(ep, "reduce_max", ("v_s1",), ("m_tile",))
    valu(ep, "rescale_from_tile_max", ("m_row", "m_tile"), ("m_row", "rescale"))
    valu(ep, "sub_m", ("v_s1", "m_row"), ("v_s1",))
    valu(ep, "exp2", ("v_s1",), ("v_p1",), kind="trans", start=0, length=16)
    ep.interleave(P(9), "valu", 6, 10)
    ep.interleave(P(7), "trans", 3, 10)
    ep.sched_barrier(0)
    finish_last = ("exp2", "apply_l_rescale", "reduce_sum", "cast_p")
    valu(ep, finish_last[0], ("v_p1",), ("v_p1",), kind="trans", start=16, length=16)
    valu(ep, finish_last[1], ("l_row", "rescale"), ("l_row",))
    valu(ep, finish_last[2], ("l_row", "v_p1"), ("l_row",))
    valu(ep, finish_last[3], ("v_p1",), ("v_p1",))
    ep.anchor("anchor_p", ("v_p1",), ("v_p1",))
    ep.sched_barrier(0)
    valu(ep, "scale_o", ("v_o", "rescale"), ("v_o",))
    ep.anchor("anchor_o", ("v_o",), ("v_o",))
    ep.barrier(fence="after")
    read_v(ep, 1, T("end", -1))  # E12
    ep.wait(lgkm=0)
    ep.barrier()
    pv_all(ep, "v_p1", "v_v", T("end", -1))  # E13
    valu(ep, "normalize", ("v_o", "l_row"), ("v_o",))
    ep.stagger("close")
    ep.op("vmem_store", "store_o", ins=("v_o", "m_row", "l_row"), count=0)

    lds_bytes = 2 * (g["K_TILE_BYTES"] + g["V_TILE_BYTES"])

    def workload(batch, heads, seq_q, seq_kv=None, kv_heads=None):
        """Per-WG KV tile counts as the kernel computes them (``init_tile_bounds``)."""
        seq_kv = seq_kv or seq_q
        kv_heads = kv_heads or heads
        n_kv = -(-seq_kv // BLOCK_N)
        tiles = []
        for qb in range(-(-seq_q // BLOCK_M)):
            t = n_kv
            if causal:
                end = max(0, (qb + 1) * BLOCK_M + (seq_kv - seq_q))
                t = min(-(-end // BLOCK_N), n_kv)
            t = max(4, (t + 1) // 2 * 2)
            tiles.append(t)
        tiles = tiles * (batch * heads)
        kv_tile_bytes = 2 * BLOCK_N * head_dim * 2
        flops_tile = NUM_WAVES * 2 * 2 * 32 * BLOCK_N * head_dim
        useful = 4 * batch * heads * seq_q * seq_kv * head_dim
        if causal:
            useful = 4 * batch * heads * head_dim * sum(min(seq_kv, i + 1 + seq_kv - seq_q) for i in range(seq_q))
        return {
            "tiles_per_wg": tiles,
            "lds_bytes": lds_bytes,
            "flops": sum(tiles) * flops_tile,
            "useful_flops": useful,
            "hbm_bytes_noreuse": sum(tiles) * kv_tile_bytes + 2 * batch * heads * seq_q * head_dim * 2,
            "hbm_bytes_unique": batch * (2 * heads * seq_q + 2 * kv_heads * seq_kv) * head_dim * 2,
        }

    return Schedule(
        name=f"mha_fwd_dualwave_gfx950_d{head_dim}_{'causal' if causal else 'full'}_{dtype}",
        target="gfx950",
        buffers={
            "lK": Buffer("lK", "lds", "K", 2, g["K_TILE_BYTES"]),
            "lV": Buffer("lV", "lds", "V", 2, g["V_TILE_BYTES"]),
        },
        queues={
            "dma": Queue("dma", "vm", tracked=False),
            "gld": Queue("gld", "vm", tracked=True),
            "ds": Queue("ds", "lgkm", tracked=True),
            "ds_tr": Queue("ds_tr", "lgkm", tracked=False),
        },
        sections=[
            Section("prologue", pro.ops),
            Section(
                "loop",
                lp.ops,
                loop=("j", lambda b: range(b["first"] + 3, b["end"] - 1, 2)),
                carried=("m_row", "l_row", "v_o", "v_p0"),
            ),
            Section("epilogue", ep.ops),
        ],
        groups=2,
        stagger=stagger,
        waves_per_group=NUM_WAVES // 2,
        bases=lambda n: {"first": 0, "end": n},
        valid_tiles=lambda n: n >= 4 and n % 2 == 0,
        tensors={"K": "QK^T consumes every KV tile once", "V": "P*V consumes every KV tile once"},
        meta={
            "verify_tiles": (4, 6, 8, 10),
            "block_m": BLOCK_M,
            "block_n": BLOCK_N,
            "head_dim": head_dim,
            "waves": NUM_WAVES,
            "waves_per_eu": 2,
            "rows_per_wave": 32,
            "loop_tiles_per_iter": 2,
            "max_wg_per_cu": 1,  # 8 waves x waves_per_eu=2 fills the CU
            "workload": workload,
            # defaults for values whose producer does not declare a size
            "value_dwords": {
                "v_o": 16 * DC,
                "v_s0": 32,
                "v_s1": 32,
                "m_row": 1,
                "l_row": 1,
                "m_tile": 1,
                "rescale": 1,
                "v_p0": 32,
                "v_p1": 32,
            },
            "expert_intent": [
                "K/V double-buffered in LDS; DMA for tile t+2 overlaps compute of tile t",
                "two wave groups one cluster apart: one group's memory cluster co-runs with the other's MFMA cluster",
                "softmax exp2/VALU interleaved 1:3 / 1:5 between MFMAs via sched_group_barrier",
                "s_setprio 1 around P*V so the compute group wins issue against the memory group",
            ],
        },
    )
