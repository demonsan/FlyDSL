# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 dual-wave flash-attention forward emitted from its Schedule IR.

The kernel body is not hand-written: ``kernels/attention/sched/mha_gfx950.py``
describes the pipeline (LDS double buffers, DMA/ds_read queues, the
8-cluster dual-wave loop, waits, barriers, sched_group interleaves and
priorities) and ``flydsl.sched.lower.emit`` walks it, calling the bindings
below.  The bindings reuse the gfx950 dual-wave helpers in
``flash_attn_utils``; only the *schedule* moved into the IR.

Scope: dense self/cross attention, D=64/128, bf16/f16, causal or not
(no paged KV, bias, ALiBi, sink, split-K or varlen -- use
``flash_attn_gfx950.py`` for those).  The schedule is verified before the
kernel is built; a schedule with verifier errors is rejected.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.sched.lower import bindings_used
from flydsl.sched.program import emit_program, schedule_fingerprint
from flydsl.sched.verify import verify
from kernels.attention.flash_attn_utils import (
    DualwaveGemmHelper,
    DualwaveKernelContext,
    DualwaveKvGmemToLdsLoader,
    DualwaveKvLdsToVgprLoader,
    DualwaveQLoader,
    DualwaveSoftmaxHelper,
    DualwaveStoreHelper,
    _anchor_v_o,
    _anchor_v_p,
    _make_dualwave_swp_traits,
    _s_barrier,
    _sched_barrier,
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
    _v_pair_to_vec32,
    _v_vec32_to_pair,
    daz_denormal_attr,
)
from kernels.attention.sched.mha_gfx950 import KERNEL_BINDINGS, build_mha_schedule, mha_geometry
from kernels.common.kernels_common import dtype_to_elem_type


# Plain-Python helpers live at module level: inside a @flyc.kernel body the AST
# rewriter turns every `if` into scf dispatch, which is wrong for trace-time logic.
def _tile_resolver(ctx, bases):
    """TileRef -> index value, with the same arithmetic the hand-written kernel uses."""

    def resolve(ref):
        if ref.base == "first":
            return ctx.split_tile(ref.off)
        base = bases[ref.base]
        if ref.off < 0:
            return base - (-ref.off)
        return base + ref.off if ref.off else base

    return resolve


def _require_dense(ctx):
    if ctx.active is not None:
        raise NotImplementedError("schedule-driven MHA covers the dense path only (no active guard)")


# P (a pair of f32x16) is carried across the loop as one f32x32, as the hand-written kernel does.
_CARRY_CODECS = {"v_p0": (lambda p: _v_pair_to_vec32(p), lambda v: _v_vec32_to_pair(v))}


def _check_loop_shape(sched):
    """The kernel supplies loop bounds (3, end-1, 2); the schedule's loop must agree."""
    loop = [s for s in sched.sections if s.loop is not None]
    if len(loop) != 1:
        raise ValueError(f"{sched.name}: expected exactly one loop section, got {len(loop)}")
    for n in sched.meta.get("verify_tiles", (4,)):
        r = loop[0].loop[1](sched.bases(n))
        if (r.start, r.stop, r.step) != (3, n - 1, 2):
            raise ValueError(f"{sched.name}: loop range {r} for {n} tiles differs from the kernel's (3, n-1, 2)")


def build_flash_attn_sched_gfx950_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    schedule=None,
):
    """Build the schedule-driven dense gfx950 flash-attention launcher.

    ``schedule`` overrides the default ``build_mha_schedule(...)`` -- this is the
    hook an agent uses to try a different pipeline without touching kernel code.
    """
    gpu_arch = get_hip_arch()
    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_sched_gfx950 targets gfx950, got {gpu_arch}")
    if dtype_str not in ("bf16", "f16"):
        raise RuntimeError(f"flash_attn_sched_gfx950 supports bf16/f16 only, got {dtype_str}")
    num_kv_heads = num_kv_heads or num_heads
    traits = _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=stagger,
    )
    sched = schedule or build_mha_schedule(
        head_dim, causal, dtype_str, lazy_rescale=lazy_rescale, setprio=setprio, stagger=stagger
    )
    geo = mha_geometry(head_dim)
    if geo["NUM_DMA"] != traits.SMEM_D_RPT or 2 * (geo["K_TILE_BYTES"] + geo["V_TILE_BYTES"]) != (
        traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES
    ):
        raise AssertionError("schedule geometry drifted from _make_dualwave_swp_traits")
    report = verify(sched)
    if not report.ok:
        raise ValueError(f"refusing to build from a schedule with verifier errors:\n{report}")
    missing = sorted(set(bindings_used(sched.all_ops())) - KERNEL_BINDINGS)
    if missing:
        raise ValueError(f"schedule {sched.name} uses ops this kernel does not implement: {missing}")
    _check_loop_shape(sched)
    # The JIT cache must distinguish transformed schedules with the same name.
    cache_tag = (traits.cache_tag, sched.name, schedule_fingerprint(sched))

    lds_elem = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[lds_elem, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_sched_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
    ):
        ctx = DualwaveKernelContext(
            traits, Q, K, V, O, O, O, O, O, seq_len, seq_len_kv, stride_q_n, stride_kv_n, head_dim_runtime, 0
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        if const_expr(traits.CAUSAL):
            ctx.init_causal_lpt_order()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()

        g2l = DualwaveKvGmemToLdsLoader(ctx)
        l2r = DualwaveKvLdsToVgprLoader(ctx)
        q_loader = DualwaveQLoader(ctx)
        gemm = DualwaveGemmHelper(ctx)
        sm = DualwaveSoftmaxHelper(ctx)
        store = DualwaveStoreHelper(ctx)
        elem = ctx.elem_dtype

        def _scale_o(v_o, s):
            sm.scale_o(v_o, s)
            return v_o

        def _normalize(v_o, l_row):
            sm.scale_o(v_o, sm.safe_l_inv(l_row))
            return v_o

        def _stagger_open():
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_one(ctx.stagger_i32)
            else:
                _sched_barrier(0)
                _s_barrier()

        def _stagger_close():
            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_zero(ctx.stagger_i32)
            else:
                _s_barrier()

        # One binding per schedule op name; each is a thin call into the existing helpers.
        bindings = {
            "load_k": lambda tile, buf: g2l.load_k_tile(tile, buf),
            "load_v": lambda tile, buf: g2l.load_v_tile(tile, buf),
            "read_k": lambda buf: l2r.load_k(buf),
            "read_v": lambda buf: l2r.load_v(buf),
            "load_q": lambda: q_loader.load_all(),
            "scale_q": lambda q: q_loader.scale_all(q),
            "qk": lambda v_k, q: gemm.qk(v_k, q),
            "pv_step": lambda p, v, v_o, step: gemm.pv_step_k(step, p, v, v_o),
            "pv": lambda p, v, v_o: gemm.pv(p, v, v_o),
            "causal_mask_prologue": lambda s: sm.causal_mask_prologue_if_needed(s),
            "causal_mask": lambda s, tile, end_tile: sm.causal_mask_prologue_if_needed(s, tile, kv_end_tile=end_tile),
            "seq_pad_mask": lambda s, tile: sm.seq_pad_mask_if_needed(s, tile),
            "to_lists": lambda s: sm.v_s_vec_to_lists(s),
            "reduce_max": lambda s: sm.reduce_max(s),
            "floor_masked_max": lambda m: sm.floor_masked_max(m),
            "sub_m": lambda s, m: sm.sub_m(s, m),
            "exp2": lambda s, start, length: sm.exp2(s, start, length),
            "reduce_sum": lambda l_row, p: sm.reduce_sum(l_row, p),
            "cast_p": lambda p: sm.cast_p(p),
            "anchor_p": lambda p: _anchor_v_p(traits, p, elem_dtype=elem),
            "anchor_o": lambda v_o: _anchor_v_o(traits, v_o),
            "lazy_rescale_o": sm.lazy_rescale_o,
            "rescale_o": sm.rescale_o,
            "rescale_from_tile_max": lambda m_row, m_tile: sm.rescale_from_tile_max(m_row, m_tile),
            "apply_l_rescale": lambda l_row, r: sm.apply_l_rescale(l_row, r),
            "scale_o": _scale_o,
            "normalize": _normalize,
            "init_acc": lambda: (ctx.c_zero_f, [ctx.c_zero_v16f32 for _ in range_constexpr(traits.D_CHUNKS)]),
            "stagger_open": _stagger_open,
            "stagger_close": _stagger_close,
            "store_o": lambda v_o, m_row, l_row: store.store_final_o(v_o, ctx.q_row, m_row, l_row),
        }

        def tile_of(extra):
            return _tile_resolver(ctx, dict(extra, end=ctx.split_t_end))

        _require_dense(ctx)
        # The whole program -- prologue, scf.for over KV tile pairs, epilogue -- comes from the
        # schedule, including whatever a cross-block transformation made the loop carry.
        emit_program(
            sched,
            bindings,
            tile_of,
            loop_bounds=lambda name: (fx.Index(3), ctx.split_t_end - fx.Index(1), fx.Index(2)),
            codecs=_CARRY_CODECS,
            tag=sched.name,  # op -> ISA attribution; only visible with FLYDSL_DEBUG_ENABLE_DEBUG_INFO
        )

    @flyc.jit
    def launch_flash_attn_sched_gfx950(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = cache_tag
        num_q_blocks = (fx.Index(seq_len) + traits.BLOCK_M - 1) // traits.BLOCK_M
        passthrough = [["no-nans-fp-math", "true"], ["unsafe-fp-math", "true"]] if const_expr(traits.DAZ) else None
        flash_attn_sched_gfx950_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough,
                "llvm.denormal_fpenv": (daz_denormal_attr() if const_expr(traits.DAZ) else None),
            },
        ).launch(
            grid=(traits.NUM_HEADS_Q, num_q_blocks, fx.Index(batch_size)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        *,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        stream=None,
    ):
        args = (
            Q,
            K,
            V,
            O,
            batch_size,
            seq_len,
            seq_len if seq_len_kv is None else seq_len_kv,
            traits.DEFAULT_STRIDE_Q_N if stride_q_n is None else stride_q_n,
            traits.DEFAULT_STRIDE_KV_N if stride_kv_n is None else stride_kv_n,
            traits.HEAD_DIM,
        )
        with CompilationContext.compile_hints(compile_hints):
            if stream is None:
                return launch_flash_attn_sched_gfx950(*args)
            return launch_flash_attn_sched_gfx950(*args, stream=stream)

    _launch.schedule = sched
    _launch.verify_report = report
    return _launch
