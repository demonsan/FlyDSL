# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Per-wave-group programs: IR, verifier (barrier pairing, cross-group LDS races) and emission (no GPU)."""

import pytest

from flydsl.sched.cost import estimate_cost
from flydsl.sched.ir import Buffer, Builder, Op, Queue, Schedule, Section, SlotUse, T
from flydsl.sched.program import emit_program, python_group_dispatch, python_loop, schedule_fingerprint
from flydsl.sched.resource import estimate_resources
from flydsl.sched.target import load_target
from flydsl.sched.verify import trace, verify
from kernels.attention.sched.mha_gfx950 import build_mha_schedule
from kernels.attention.sched.mha_gfx950_2prog import build_mha_2prog_schedule

pytestmark = pytest.mark.l0_backend_agnostic

TGT = load_target("gfx950")


# ------------------------------------------------------------------ a tiny two-group protocol
def _dma(b, slot=0, tile=None):
    b.op("lds_dma", "load", count=1, queue="dma", writes=(SlotUse("L", slot, tile),), label=f"dma L[{slot}]")


def _read(b, slot=0, tile=None, out="x"):
    b.op("ds_read", "read", outs=(out,), count=1, queue="ds", reads=(SlotUse("L", slot, tile),), label=f"rd L[{slot}]")


def _two(g0_ops, g1_ops, **kw):
    return Schedule(
        "two",
        "gfx950",
        {"L": Buffer("L", "lds", "X", 1, 64)},
        {"dma": Queue("dma", "vm"), "ds": Queue("ds", "lgkm", tracked=True)},
        sections=[],
        groups=2,
        group_sections=[[Section("g0", g0_ops)], [Section("g1", g1_ops)]],
        **kw,
    )


def _producer_consumer(consumer_reads_before_barrier=False, producer_overwrites=False, extra_barrier=False):
    """Group 0 fills L[0] by DMA; group 1 reads it.  Both groups write their half of L[0]."""
    p, c = Builder(), Builder()
    _dma(p)
    _dma(c)
    p.wait(vm=0)
    c.wait(vm=0)
    p.barrier()
    if consumer_reads_before_barrier:
        _read(c)
    c.barrier()
    if not consumer_reads_before_barrier:
        _read(c)
    c.wait(lgkm=0)
    if producer_overwrites:  # group 0 refills L[0] while group 1 may still be reading it
        _dma(p)
        p.wait(vm=0)
    p.barrier()
    c.barrier()
    if extra_barrier:
        c.barrier()
    return _two(p.ops, c.ops)


def test_cross_group_protocol_verifies():
    r = verify(_producer_consumer())
    assert r.ok, str(r)


def test_cross_group_raw_race():
    r = verify(_producer_consumer(consumer_reads_before_barrier=True))
    assert "SYNC.RAW_LDS" in r.rules("error"), str(r)
    assert any(f.where.startswith("g1:") for f in r.errors)


def test_cross_group_war_race():
    """Group 0 refills L[0] between the same two barriers as group 1 reads it: the two are concurrent.

    Ops of different groups in one barrier phase have no order, so the race is reported from
    whichever side the merge visits second (here: group 1's read sees group 0's new write)."""
    r = verify(_producer_consumer(producer_overwrites=True))
    assert r.rules("error") in (["SYNC.WAR_LDS"], ["SYNC.RAW_LDS"]), str(r)
    assert "dma L[0]" in r.errors[0].message


def test_barrier_count_mismatch_is_a_deadlock():
    r = verify(_producer_consumer(extra_barrier=True))
    assert r.rules("error") == ["SYNC.BARRIER_MISMATCH"], str(r)
    assert "group 0: 2, group 1: 3" in r.errors[0].message


def test_slot_needs_every_writer_group():
    """Group 1 reads L[0] after group 0 wrote its half but before group 1 wrote its own."""
    p, c = Builder(), Builder()
    _dma(p)
    p.wait(vm=0)
    p.barrier()
    c.barrier()
    _read(c)
    c.wait(lgkm=0)
    _dma(c)
    c.wait(vm=0)
    p.barrier()
    c.barrier()
    assert "SYNC.UNINIT_READ" in verify(_two(p.ops, c.ops)).rules("error")


def test_scopes_and_coverage_are_per_group():
    p, c = Builder(), Builder()
    p.op("valu", "init", outs=("a",))
    c.op("valu", "use", ins=("a",), outs=("b",))  # "a" exists only in group 0's program
    r = verify(_two(p.ops, c.ops))
    assert "SCOPE.UNDEFINED" in r.rules("error") and r.errors[0].where.startswith("g1:")


def test_group_programs_must_match_group_count():
    with pytest.raises(ValueError):
        Schedule("x", "gfx950", {}, {}, [], groups=2, group_sections=[[Section("a", [])]])


def test_stagger_close_alone_deadlocks_single_program():
    b = Builder()
    b.stagger("open")
    s = Schedule("st", "gfx950", {}, {}, [Section("main", b.ops)], groups=2, stagger=True)
    assert verify(s).rules("error") == ["SYNC.BARRIER_MISMATCH"]


# ------------------------------------------------------------------ the MHA two-program variant
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("prio", ["toggle", "static"])
def test_mha_2prog_verifies(head_dim, causal, prio):
    s = build_mha_2prog_schedule(head_dim, causal, prio=prio)
    r = verify(s)
    assert r.ok, str(r)
    base = verify(build_mha_schedule(head_dim, causal))
    assert r.rules("warn") == base.rules("warn")
    assert not any(op.kind == "stagger" for op in s.all_ops())


def test_mha_2prog_groups_pair_the_same_clusters_as_stagger():
    """Barrier k joins the same (group 0 op, group 1 op) phases in both formulations."""
    from flydsl.sched.verify import _barrier_row, barrier_counts

    base, two = build_mha_schedule(128, True), build_mha_2prog_schedule(128, True)
    for n in (4, 6, 10):
        ev = trace(base, n)
        rows = barrier_counts(base, ev)
        for g in (0, 1):
            ev2 = trace(two, n, g)
            row2 = _barrier_row(two, ev2, g)
            ph = [(e.op.label, rows[g][i]) for i, e in enumerate(ev) if e.op.kind not in ("stagger", "barrier")]
            ph2 = [(e.op.label, row2[i]) for i, e in enumerate(ev2) if e.op.kind != "barrier"]
            ph2 = [x for x in ph2 if not x[0].startswith("setprio")]
            ph = [x for x in ph if not x[0].startswith("setprio")]
            assert ph == ph2


def test_mha_2prog_missing_offset_barrier_deadlocks():
    s = build_mha_2prog_schedule(128, True)
    pro = s.group_sections[1][0]
    pro.ops = [op for op in pro.ops if "phase offset" not in op.label]
    assert "SYNC.BARRIER_MISMATCH" in verify(s).rules("error")


def test_mha_2prog_offset_moved_to_the_end_is_the_unstaggered_schedule():
    """Moving group 1's offset barrier to its end keeps the counts equal and removes the phase
    offset: that is the plain non-staggered schedule, which is also legal."""
    s = build_mha_2prog_schedule(128, True)
    pro1 = s.group_sections[1][0]
    off = next(op for op in pro1.ops if "phase offset" in op.label)
    pro1.ops.remove(off)
    s.group_sections[1][2].ops.append(off)
    assert verify(s).ok


# ------------------------------------------------------------------ emission
class _Rec(dict):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def get(self, name, default=None):
        def fn(*ins, **kw):
            self.out.append(name)
            acc = ["o0", "o1"]
            shaped = {
                "lazy_rescale_o": (acc, "m", "l", "p"),
                "init_acc": (0.0, acc),
                "rescale_from_tile_max": ("m", "r"),
            }
            if name in shaped:
                return shaped[name]
            return acc if name in ("pv", "pv_step", "scale_o", "normalize", "anchor_o") else f"{name}#"

        return fn


class _Prim:
    def __init__(self, out):
        self.out = out

    def __getattr__(self, name):
        return lambda *a, **k: self.out.append(f"<{name}>")


def _interpret(sched, n, group):
    out = []
    emit_program(
        sched,
        _Rec(out),
        tile_of=lambda extra: (lambda r: r.eval({"first": 0, "end": n, **extra})),
        loop_bounds=lambda name: (3, n - 1, 2),
        loop_impl=python_loop,
        primitives=_Prim(out),
        target=TGT,
        group_id=group,
        dispatch=python_group_dispatch,
    )
    return out


@pytest.mark.parametrize("n", [4, 8])
def test_emit_dispatches_each_group_its_program(n):
    two = build_mha_2prog_schedule(128, True)
    g0, g1 = _interpret(two, n, 0), _interpret(two, n, 1)
    work = [x for x in g0 if not x.startswith(("<", "stagger"))]
    assert work == [x for x in g1 if not x.startswith(("<", "stagger"))]
    assert g0.count("<barrier>") == g1.count("<barrier>")
    # group 1 crosses its offset barrier before its first QK^T, group 0 its own one at the very end
    assert g1[: g1.index("qk")].count("<barrier>") == g0[: g0.index("qk")].count("<barrier>") + 1
    assert g0[-2:] == ["<barrier>", "store_o"] and g1[-2:] != ["<barrier>", "store_o"]


def test_emit_requires_group_id_and_fingerprints_differ():
    two = build_mha_2prog_schedule(128, True)
    with pytest.raises(ValueError):
        emit_program(two, {}, tile_of=lambda e: None, loop_bounds=lambda n: (0, 0, 1))
    static = build_mha_2prog_schedule(128, True, prio="static")
    fps = {schedule_fingerprint(s) for s in (two, static, build_mha_schedule(128, True))}
    assert len(fps) == 3


def test_all_ops_cover_every_group():
    p, c = [Op("valu", "a", binding="only_g0")], [Op("valu", "b", binding="only_g1")]
    s = _two(p, c)
    assert {o.binding for o in s.all_ops()} == {"only_g0", "only_g1"}
    assert [sec.name for sec in s.sections] == ["g0"]


def test_tile_refs_resolve_per_group():
    b = Builder()
    _dma(b, tile=T("first"))
    assert trace(_two(b.ops, list(b.ops), bases=lambda n: {"first": 0}), 4, 1)[0].bases == {"first": 0}


# ------------------------------------------------------------------ resource / cost per group

MM = "mfma_f32_16x16x32_f16"


def _asym(swap_g1=True, g1_mfma=8, g1_dwords=64):
    """Group 0 loop: {MFMA x4} | {DMA}; group 1 loop: {DMA} | {MFMA x g1_mfma} (or the same order as group 0)."""

    def loop(order, mfma_n, dwords):
        b = Builder()
        for part in order:
            if part == "mfma":
                b.op("mfma", "mm", ins=("a",), outs=("acc",), count=mfma_n, mfma=MM, out_dwords={"acc": dwords})
            else:
                _dma(b)
            b.barrier()
        return Section("loop", b.ops, loop=("i", lambda bases: range(2)), carried=())

    pro = Section("pro", [Op("valu", "init", binding="init", outs=("a",), out_dwords={"a": 4})])
    g0 = [pro, loop(("mfma", "dma"), 4, 16)]
    g1 = [pro, loop(("dma", "mfma") if swap_g1 else ("mfma", "dma"), g1_mfma, g1_dwords)]
    return Schedule(
        "asym",
        "gfx950",
        {"L": Buffer("L", "lds", "X", 1, 64)},
        {"dma": Queue("dma", "vm")},
        sections=[],
        groups=2,
        meta={"waves_per_eu": 2, "verify_tiles": (4,)},
        group_sections=[g0, g1],
    )


def test_cost_pairs_asymmetric_group_loops_by_barrier_phase():
    c = estimate_cost(_asym())
    assert [p["status"] for p in c.phases] == ["ok", "ok"]  # each slot: one group computes, the other loads
    assert c.groups[0]["mfma_cycles_per_iter"] == 4 * TGT.mfma_cycles(MM)
    assert c.groups[1]["mfma_cycles_per_iter"] == 8 * TGT.mfma_cycles(MM)
    assert c.mfma_cycles_per_iter_per_wave == 8 * TGT.mfma_cycles(MM)  # the busier group bounds the pair
    same = estimate_cost(_asym(swap_g1=False))
    assert [p["status"] for p in same.phases] == ["contended", "idle"]
    assert "group 1" in str(same)


def test_cost_flags_loops_without_static_pairing():
    s = _asym()
    s.group_sections[1][1].ops = s.group_sections[1][1].ops + [Op("barrier", "extra", args={"fence": "both"})]
    c = estimate_cost(s)
    assert not c.phases and any("no static dual-wave pairing" in h for h in c.hints)


def test_resource_vgpr_peak_is_the_hungriest_group():
    r = estimate_resources(_asym(g1_dwords=128))
    assert r.vgpr_per_group == {0: 4 + 16, 1: 4 + 128}
    assert r.vgpr_live_peak == 132 and r.vgpr_peak_where.startswith("g1:")
    assert r.lds_bytes == 64  # LDS is shared, counted once


@pytest.mark.parametrize("prio", ["toggle", "static"])
def test_mha_2prog_cost_and_resources_match_the_single_program(prio):
    one, two = build_mha_schedule(128, True), build_mha_2prog_schedule(128, True, prio=prio)
    c1, c2 = estimate_cost(one), estimate_cost(two)
    assert [(p["a"], p["b"], p["status"]) for p in c1.phases] == [(p["a"], p["b"], p["status"]) for p in c2.phases]
    assert c1.mfma_cycles_per_iter_per_wave == c2.mfma_cycles_per_iter_per_wave
    r1, r2 = estimate_resources(one), estimate_resources(two)
    assert r1.vgpr_live_peak == r2.vgpr_live_peak and r1.lds_bytes == r2.lds_bytes
    assert r1.vgpr_per_group == {} and set(r2.vgpr_per_group) == {0, 1}
