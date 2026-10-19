# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The schedule-driven kernels emit the same call sequence as the hand-written ones.

Each reference below is transcribed from the hand-written gfx950 source
(``mla_fwd_decode_m16x8_fp8_fp8.py`` / ``flash_attn_gfx950.py``, as of the
commit that introduced ``flydsl.sched``).  ``emit`` runs with recording
primitives and bindings, so these tests need neither MLIR nor a GPU.
"""

import pytest

from flydsl.sched.ir import TileRef
from flydsl.sched.lower import emit
from flydsl.sched.target import load_target
from kernels.attention.sched import mla_gfx950 as M
from kernels.attention.sched.mha_gfx950 import build_mha_schedule

pytestmark = pytest.mark.l0_backend_agnostic

TGT = load_target("gfx950")


def W(vm=None, lgkm=None, exp=None):
    return ("waitcnt", 63 if vm is None else vm, 15 if lgkm is None else lgkm, 7 if exp is None else exp)


class Rec:
    """Recording control-op primitives; tokens match the references below."""

    def __init__(self, out):
        self.out = out

    def wait(self, vm=None, lgkm=None, exp=None):
        self.out.append(W(vm, lgkm, exp))

    def barrier(self, fence="both", vm=None, lgkm=None, asm=False):
        if fence in ("both", "before"):
            self.out.append(("sched_barrier", 0))
        if vm is not None or lgkm is not None:
            self.out.append(W(vm, lgkm))
        self.out.append(("s_barrier",))
        if fence in ("both", "after"):
            self.out.append(("sched_barrier", 0))

    def sched_barrier(self, mask=0):
        self.out.append(("sched_barrier", mask))

    def sched_group(self, cls, size, group):
        self.out.append(("sched_group", TGT.sched_mask(cls), size, group))

    def setprio(self, prio):
        self.out.append(("setprio", prio))

    def nop(self, n, asm=True):
        self.out.append(("s_nop", n))


class RecBindings(dict):
    """Every binding records (name, *args, **kwargs) and returns a fresh token."""

    def __init__(self, out, outs_of):
        super().__init__()
        self.out, self.outs_of = out, outs_of

    def get(self, name, default=None):
        def fn(*args, **kw):
            self.out.append((name,) + tuple(kw[k] for k in sorted(kw)))
            n = self.outs_of.get(name, 1)
            return tuple(f"{name}#{i}" for i in range(n)) if n > 1 else f"{name}#0"

        return fn


def _emit(ops, env, outs_of=None):
    out = []
    emit(ops, RecBindings(out, outs_of or {}), env, tile=lambda r: (r.base, r.off), primitives=Rec(out), target=TGT)
    return out


# ---------------------------------------------------------------------------------- MLA
def ref_mla_gemm1(prefetch=True):
    """_process_tile_gemm1, gfx950, from '_maybe_prefetch(0)' through 's_setprio(14)'."""
    out = []

    def pf(block):
        if prefetch:
            out.append(("prefetch_kv", block, 0))
            out.append(("prefetch_kv", block, 1))

    pf(0)
    for nope_pair in range(8):
        t0, t1 = 2 * nope_pair, 2 * nope_pair + 1
        out += [("read_k", t0 * 32, 16 * h) for h in range(4)]
        out += [("read_k", t1 * 32, 16 * h) for h in range(4)]
        pf(nope_pair + 1)
        out += [("sched_barrier", 0), W(lgkm=4)]
        out += [("qk_mfma", t0, "nope")] * 4
        if nope_pair == 0:
            out.append(("setprio", 15))
        out.append(W(lgkm=0))
        out += [("qk_mfma", t1, "nope")] * 4
    out += [("read_k", 16 * 32, 16 * h) for h in range(4)]
    out += [("read_k", 17 * 32, 16 * h) for h in range(4)]
    out += [("sched_barrier", 0), W(lgkm=4)]
    out += [("qk_mfma", 0, "rope")] * 4
    out.append(W(lgkm=0))
    out += [("qk_mfma", 1, "rope")] * 4
    out.append(("setprio", 14))
    return out


def ref_mla_gemm2_core():
    """_gemm2_core, K_HALVES == 2 (gfx950)."""
    out = [("setprio", 15)]
    for pv_pair in range(8):
        ia, ib = 2 * pv_pair, 2 * pv_pair + 1
        for strip in (ia * 32, ib * 32):
            for half in (0, 32):
                out.append(("read_v_tr", strip, half + 0))
                out.append(("read_v_tr", strip, half + 16))
        for step in range(4):
            out += [("sched_barrier", 0), W(lgkm=(6, 4, 2, 0)[step]), ("v_swap",), ("pv_mfma",), ("pv_mfma",)]
        out.append(("sched_barrier", 0))
        if pv_pair < 7:
            out.append(("s_nop", 1))
    out.append(("setprio", 0))
    return out


@pytest.mark.parametrize("prefetch", [True, False])
def test_mla_gemm1_matches_hand_schedule(prefetch):
    got = _emit(M.gemm1_ops(prefetch=prefetch), {})
    # qk_mfma records (name, q_idx, q_kind) -- sorted kwargs
    assert got == ref_mla_gemm1(prefetch)


def test_mla_gemm1_accumulator_chain():
    """First NoPE step starts from C=0 (no acc input); every later MFMA accumulates into p{h}."""
    ops = [o for o in M.gemm1_ops() if o.binding == "qk_mfma"]
    assert len(ops) == 72
    assert all(len(o.ins) == 1 for o in ops[:4]) and all(len(o.ins) == 2 for o in ops[4:])
    assert [o.outs[0] for o in ops[:8]] == ["p0", "p1", "p2", "p3"] * 2


def test_mla_gemm2_core_matches_hand_schedule():
    env = {"p_lo": "P_LO", "p_hi": "P_HI", **{f"o{i}": f"O{i}" for i in range(32)}}
    got = _emit(M.gemm2_core_ops(), env, outs_of={"v_swap": 2})
    assert got == ref_mla_gemm2_core()


def test_mla_gemm2_core_operand_routing():
    """Step s of PV pair k feeds (iter, p) = [(2k,lo),(2k,hi),(2k+1,lo),(2k+1,hi)][s] into O[2*iter], O[2*iter+1]."""
    mf = [o for o in M.gemm2_core_ops() if o.binding == "pv_mfma"]
    assert len(mf) == 64
    for k in range(8):
        for s, (it, p) in enumerate([(2 * k, "p_lo"), (2 * k, "p_hi"), (2 * k + 1, "p_lo"), (2 * k + 1, "p_hi")]):
            a, b = mf[8 * k + 2 * s], mf[8 * k + 2 * s + 1]
            assert (a.ins[1], a.outs[0], b.ins[1], b.outs[0]) == (p, f"o{2 * it}", p, f"o{2 * it + 1}")


# ---------------------------------------------------------------------------------- MHA
def ref_mha_loop(head_dim=128, causal=True):
    """flash_attn_gfx950.py main loop body, dense path, C0..C7 (lazy rescale, setprio, stagger)."""
    ndma = head_dim // 64
    out = []

    def pairs(n, cnt, grp, cls):
        n = max(1, (n + 1) // 2) if head_dim == 64 else n
        for _ in range(n):
            out.append(("sched_group", 8, 1, grp))
            out.append(("sched_group", TGT.sched_mask(cls), cnt, grp))

    def sync():
        out.extend([("sched_barrier", 0), ("s_barrier",), ("sched_barrier", 0)])

    def mem(load, read):
        out.extend([("s_nop", 7), ("sched_barrier", 0), load, read])

    def mem_end():
        out.extend([W(lgkm=0), W(vm=2 * ndma)])
        sync()

    j = lambda o: ("j", o)  # noqa: E731
    # C0
    mem(("load_v", 1, j(-2)), ("read_k", 1))
    mem_end()
    # C1
    out += [("qk",), ("exp2", 16, 16), ("reduce_sum",), ("cast_p",), ("anchor_p",)]
    pairs(6, 3, 1, "trans")
    pairs(10, 5, 1, "valu")
    sync()
    # C2
    mem(("load_k", 1, ("j", 0)), ("read_v", 0))
    mem_end()
    # C3
    out += [("setprio", 1), ("pv_step", 0), ("to_lists",), ("reduce_max",)]
    pairs(4, 6, 2, "valu")
    out += [("lazy_rescale_o",), ("pv_step", 1), ("pv_step", 2), ("pv_step", 3), ("sub_m",), ("exp2", 16, 0)]
    pairs(6, 6, 2, "valu")
    pairs(6, 3, 2, "trans")
    out.append(("setprio", 0))
    sync()
    # C4
    mem(("load_v", 0, j(-1)), ("read_k", 0))
    mem_end()
    # C5
    out += [("qk",), ("exp2", 16, 16), ("reduce_sum",), ("cast_p",), ("anchor_p",)]
    pairs(6, 3, 3, "trans")
    pairs(10, 5, 3, "valu")
    sync()
    # C6
    mem(("load_k", 0, j(1)), ("read_v", 1))
    out.append(("causal_mask", j(0), j(-1)) if causal else ("to_lists",))
    mem_end()
    # C7
    out += [("setprio", 1), ("pv_step", 0), ("reduce_max",)]
    pairs(4, 6, 4, "valu")
    out += [("lazy_rescale_o",), ("pv_step", 1), ("pv_step", 2), ("pv_step", 3), ("sub_m",), ("exp2", 16, 0)]
    pairs(6, 5, 4, "valu")
    pairs(6, 3, 4, "trans")
    out.append(("setprio", 0))
    sync()
    return out


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_mha_loop_body_matches_hand_schedule(head_dim, causal):
    s = build_mha_schedule(head_dim, causal)
    env = {n: n for n in ("q", "m_row", "l_row", "v_o", "v_p0")}
    got = _emit(s.section("loop").ops, env, outs_of={"lazy_rescale_o": 4})
    assert got == ref_mha_loop(head_dim, causal)


def test_emit_rejects_missing_bindings_and_values():
    s = build_mha_schedule(128, True)
    with pytest.raises(KeyError, match="no binding"):
        emit(s.section("prologue").ops, {}, {}, primitives=Rec([]), target=TGT)
    with pytest.raises(KeyError, match="undefined values"):
        _emit(s.section("loop").ops, {})


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_mha_schedule_only_uses_kernel_bindings(causal, lazy):
    from flydsl.sched.lower import bindings_used
    from kernels.attention.sched.mha_gfx950 import KERNEL_BINDINGS

    s = build_mha_schedule(128, causal, lazy_rescale=lazy)
    assert set(bindings_used(s.all_ops())) <= KERNEL_BINDINGS


def test_stagger_bindings_take_no_arguments():
    """Kernels implement stagger_open/close as zero-arg callables (the phase is in the name)."""
    calls = []
    b = {"stagger_open": lambda: calls.append("open"), "stagger_close": lambda: calls.append("close")}
    ops = [o for o in build_mha_schedule(128, True).all_ops() if o.kind == "stagger"]
    emit(ops, b, {}, primitives=Rec([]), target=TGT)
    assert calls == ["open", "close"]


def test_tileref_resolution_is_kernel_owned():
    assert TileRef("j", -2).eval({"j": 7}) == 5
