# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""flydsl.sched analyzers on the gfx950 MHA / MLA schedules (no GPU, no MLIR)."""

import pytest

from flydsl.sched.cost import estimate_cost
from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T
from flydsl.sched.resource import estimate_resources
from flydsl.sched.target import UnknownFact, load_target
from flydsl.sched.verify import trace, verify
from kernels.attention.sched import mla_gfx950
from kernels.attention.sched.mha_gfx950 import build_mha_schedule

pytestmark = pytest.mark.l0_backend_agnostic


# ---------------------------------------------------------------- target facts
def test_target_facts_carry_sources_and_refuse_unknowns(monkeypatch):
    # The public description only: private packs (FLYDSL_SCHED_TARGET_PATH) may override these facts.
    monkeypatch.delenv("FLYDSL_SCHED_TARGET_PATH", raising=False)
    load_target.cache_clear()
    t = load_target("gfx950")
    assert t.counter_max("lgkm") == 15 and t.counter_max("vm") == 63
    assert t.mfma_cycles("mfma_f32_32x32x16_bf16") == 32
    assert t.fact("lds_bytes_per_cu").source == "cdna4-isa"
    assert t.fact("latency_cycles").value["ds_read_b128"] is None  # unknown stays unknown
    with pytest.raises(UnknownFact):
        t.require("does_not_exist")
    with pytest.raises(UnknownFact):
        load_target("gfx1250")  # no description -> no silent substitution


# ------------------------------------------------------------------- MHA (dual-wave)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("stagger", [True, False])
def test_mha_hand_schedule_verifies(head_dim, causal, stagger):
    r = verify(build_mha_schedule(head_dim, causal, stagger=stagger))
    assert r.ok, str(r)


def test_mha_d64_epilogue_interleave_over_asks():
    """E11 asks sched_group_barrier for 9 MFMA slots with only 8 MFMAs at D=64."""
    r = verify(build_mha_schedule(64, True))
    assert r.rules("warn") == ["SCHED.GROUP_UNSATISFIABLE"]
    assert verify(build_mha_schedule(128, True)).rules("warn") == []


def _loop(s):
    return s.section("loop").ops


def _mutated(fn, **kw):
    s = build_mha_schedule(128, True, **kw)
    fn(s)
    return verify(s)


def test_mha_loose_dma_wait_is_a_raw_hazard():
    def f(s):
        next(o for o in _loop(s) if o.kind == "wait" and o.args.get("vm") is not None).args["vm"] = 8

    assert "SYNC.RAW_LDS" in _mutated(f).rules("error")


def test_mha_missing_cluster_barrier_breaks_both_directions():
    def f(s):
        ops = _loop(s)
        del ops[[i for i, o in enumerate(ops) if o.kind == "barrier"][1]]

    errs = _mutated(f).rules("error")
    assert "SYNC.RAW_LDS" in errs and "SYNC.WAR_LDS" in errs


def test_mha_wrong_prefetch_slot_is_stale_data():
    def f(s):
        o = [o for o in _loop(s) if o.binding == "load_k"][0]
        o.writes = (SlotUse("lK", 0, o.writes[0].tile),)

    assert "DATA.STALE_TILE" in _mutated(f).rules("error")


def test_mha_dropped_lgkm_wait_before_untracked_ds_read_tr():
    def f(s):
        ops = s.section("epilogue").ops
        del ops[[i for i, o in enumerate(ops) if o.kind == "wait" and o.args.get("lgkm") == 0][1]]

    assert "SYNC.REG_NOT_READY" in _mutated(f).rules("error")


def test_mha_uncarried_loop_value_is_a_scope_error():
    s = build_mha_schedule(128, True)
    sec = s.section("loop")
    sec.carried = tuple(c for c in sec.carried if c != "l_row")
    assert "SCOPE.UNCARRIED" in verify(s).rules("error")


def test_mha_stagger_needs_two_barriers_between_dma_wait_and_cross_group_read():
    """With the phase shift, a DMA drained in cluster k is safe to read only in cluster k+2.

    lV[0] (tile j-3) is drained by C0's vmcnt wait and read in C2.  Reading it in C1
    instead is fine for a single wave group but races the other, shifted group.
    """

    def hoist_v_read_into_c1(s):
        ops = _loop(s)
        read = ops.pop(next(i for i, o in enumerate(ops) if o.binding == "read_v"))
        ops.insert([i for i, o in enumerate(ops) if o.kind == "barrier"][0] + 1, read)

    staggered = build_mha_schedule(128, True, stagger=True)
    hoist_v_read_into_c1(staggered)
    assert "SYNC.RAW_LDS" in verify(staggered).rules("error")

    plain = build_mha_schedule(128, True, stagger=False)
    hoist_v_read_into_c1(plain)
    assert "SYNC.RAW_LDS" not in verify(plain).rules("error")


def test_mha_trace_consumes_every_tile_once():
    s = build_mha_schedule(128, True)
    for n in (4, 6, 8, 12):
        ks = [ref.eval(ev.bases) for ev in trace(s, n) for t, ref in ev.op.consumes if t == "K"]
        assert sorted(ks) == list(range(n))


def test_mha_resources_and_cost():
    s = build_mha_schedule(128, True)
    r = estimate_resources(s)
    assert r.lds_bytes == 68096 and r.fits
    c = estimate_cost(s, shape=dict(batch=1, heads=64, seq_q=8192))
    assert [cl.mfma for cl in c.loop_clusters] == [0, 16, 0, 16, 0, 16, 0, 16]
    assert all(p["status"] == "ok" for p in c.phases)  # every phase pairs one MFMA cluster with one memory cluster
    assert c.mfma_cycles_per_tile_per_wave == 1024
    # Causal: rounds are as long as their slowest WG, so the bound sits below peak.
    assert c.estimate["tflops_at_mfma_bound"] < 2500
    # Non-causal with a whole number of WG rounds reaches the documented ~2.5 PF bf16 peak exactly.
    full = estimate_cost(build_mha_schedule(128, False), shape=dict(batch=4, heads=64, seq_q=8192))
    assert full.estimate["wg_rounds"] == 32 and 2500 < full.estimate["tflops_at_mfma_bound"] < 2530


# ------------------------------------------------------------------------- MLA
def test_mla_hand_schedule_verifies():
    r = verify(mla_gfx950.build_mla_schedule())
    assert r.ok, str(r)
    assert r.rules("warn") == []


def _mla_mutated(patch):
    s = mla_gfx950.build_mla_schedule()
    orig = s.program

    def prog(n):
        secs = orig(n)
        patch(secs)
        return secs

    s.program = prog
    return verify(s).rules("error")


def test_mla_tile_barrier_without_vmcnt0():
    def patch(secs):
        if len(secs) > 2:  # tile 1 exists: its barrier no longer waits for the prefetch
            secs[2].ops[0].args.update(vm=None)

    assert "SYNC.RAW_LDS" in _mla_mutated(patch)


def test_mla_output_bounce_into_the_v_slot():
    def patch(secs):
        last = secs[-1]
        cur = (len(secs) - 2) % 2
        for o in last.ops:
            if o.binding == "store_o":
                o.writes = (SlotUse("lKV", cur),)

    assert "SYNC.WAR_LDS" in _mla_mutated(patch)


def test_mla_v_drain_off_by_one():
    def patch(secs):
        for sec in secs[1:]:
            for o in sec.ops:
                if o.kind == "wait" and o.args.get("lgkm") == 6:
                    o.args["lgkm"] = 7

    assert "SYNC.REG_NOT_READY" in _mla_mutated(patch)


def test_mla_cost_points_at_the_faster_fp8_mfma():
    c = estimate_cost(mla_gfx950.build_mla_schedule(), shape=dict(num_seqs=64, kv_len=8192))
    assert c.mfma_cycles_per_tile_per_wave == 72 * 16 + 64 * 16
    assert any("mfma_f32_16x16x128_f8f6f4" in h for h in c.hints)


# ------------------------------------------------------------------ small schedules
def _tiny(write_slot, wait_vm):
    pro = Builder()
    pro.op("lds_dma", "load", count=1, queue="dma", writes=(SlotUse("b", write_slot, T("t")),))
    pro.wait(vm=wait_vm)
    pro.barrier()
    pro.op("ds_read", "read", outs=("x",), count=1, queue="ds", reads=(SlotUse("b", 0, T("t")),))
    pro.wait(lgkm=0)
    return Schedule(
        "tiny",
        "gfx950",
        {"b": Buffer("b", "lds", "X", 1, 64)},
        {"dma": Queue("dma", "vm"), "ds": Queue("ds", "lgkm", tracked=True)},
        [Section("p", pro.ops)],
        bases=lambda n: {"t": 0},
    )


def test_tiny_schedule_rules():
    assert verify(_tiny(0, 0)).ok
    assert "SYNC.RAW_LDS" in verify(_tiny(0, 1)).rules("error")
    assert "HW.WAITCNT_RANGE" in verify(_tiny(0, 64)).rules("error")
