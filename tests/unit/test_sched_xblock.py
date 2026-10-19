# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cross-block scheduling: moves across barriers / the loop back-edge, wait synthesis,
the hoist pass, and whole-program emission of transformed schedules (CPU only)."""

from collections import Counter

import pytest

from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T, TileRef
from flydsl.sched.program import emit_program, python_loop, schedule_fingerprint
from flydsl.sched.target import load_target
from flydsl.sched.transform import (
    IllegalTransform,
    cover_summary,
    hoist_async,
    move_op,
    rotate_down,
    rotate_up,
    synthesize_waits,
    where_is,
)
from flydsl.sched.verify import trace, verify
from kernels.attention.sched.mha_gfx950 import build_mha_schedule

pytestmark = pytest.mark.l0_backend_agnostic
TGT = load_target("gfx950")


def _instances(sched, n):
    """Multiset of executed (binding, resolved tile args, slots) -- the work a program does."""
    out = Counter()
    for ev in trace(sched, n):
        op = ev.op
        if op.binding is None:
            continue
        args = tuple(sorted((k, v.eval(ev.bases) if isinstance(v, TileRef) else v) for k, v in op.args.items()))
        slots = tuple((u.buffer, u.slot) for u in op.writes + op.reads)
        out[(op.binding, args, slots)] += 1
    return out


def _perturbed():
    """Hand schedule with two K prefetches pushed late: K[j+1] into the next iteration, K[j] into C4."""
    s = build_mha_schedule(128, True)
    i = where_is(s, lambda o: str(o) == "dma K[j+1]->lK[0]")[0][1]
    p = rotate_down(s, "loop", i)
    i = where_is(p, lambda o: str(o) == "dma K[j]->lK[1]")[0][1]
    j = next(k for k, o in enumerate(p.section("loop").ops) if str(o) == "dma V[j-1]->lV[0]")
    return s, synthesize_waits(move_op(p, ("loop", i), ("loop", j)))


@pytest.fixture(scope="module")
def perturbed():
    return _perturbed()


@pytest.fixture(scope="module")
def recovered(perturbed):
    return hoist_async(perturbed[1])


# ----------------------------------------------------------------------------- moves
def test_rotations_preserve_the_executed_work():
    s = build_mha_schedule(128, True)
    i = where_is(s, lambda o: str(o) == "dma K[j+1]->lK[0]")[0][1]
    down = rotate_down(s, "loop", i)
    up = rotate_up(down, "loop", 0)
    for n in (4, 6, 8, 12):
        assert _instances(down, n) == _instances(s, n)
        assert _instances(up, n) == _instances(s, n)
    # rotating back puts the op at the body end (same instant as the next iteration's head)
    orig = [str(o) for o in s.section("loop").ops]
    orig.append(orig.pop(i))
    assert [str(o) for o in up.section("loop").ops] == orig


def test_rotation_moves_the_first_instance_into_straight_line_code():
    s = build_mha_schedule(128, True)
    i = where_is(s, lambda o: str(o) == "dma K[j+1]->lK[0]")[0][1]
    down = rotate_down(s, "loop", i)
    assert str(down.section("loop").ops[0]) == "dma K[j-1]->lK[0]"
    assert str(down.section("epilogue").ops[0]) == "dma K[end-2]->lK[0]"  # last instance, after the loop
    assert not any(str(o) == "dma K[first+2]->lK[0]" for o in down.section("prologue").ops)  # twin absorbed


def test_moves_that_change_trip_counts_are_refused():
    s = build_mha_schedule(128, True)
    with pytest.raises(IllegalTransform):
        move_op(s, ("loop", 2), ("prologue", 0))
    i = where_is(s, lambda o: o.binding == "read_k")[1][1]  # ds_read in the loop: no peeled twin
    with pytest.raises(IllegalTransform):
        rotate_down(s, "loop", i)


# ----------------------------------------------------------------------------- waits
def test_wait_synthesis_keeps_hand_loop_waits_and_only_loosens():
    s = build_mha_schedule(128, True)
    w = synthesize_waits(s)
    assert verify(w).ok
    for a, b in zip(s.all_ops(), w.all_ops()):
        for c in ("vm", "lgkm"):
            if a.args.get(c) is not None:
                assert b.args[c] >= a.args[c]
    loop_a = [(o.args.get("vm"), o.args.get("lgkm")) for o in s.section("loop").ops if o.kind == "wait"]
    loop_b = [(o.args.get("vm"), o.args.get("lgkm")) for o in w.section("loop").ops if o.kind == "wait"]
    assert loop_a == loop_b  # the hand loop waits are already the loosest safe values


def test_wait_synthesis_repairs_an_illegal_schedule(perturbed):
    _, p = perturbed
    assert verify(p).ok


def test_multi_instruction_entry_needs_all_its_instructions_retired():
    b = Builder()
    b.op("lds_dma", "load", count=1, queue="dma", writes=(SlotUse("x", 0, T("t")),))
    b.wait(vm=0)
    b.barrier()
    b.op("ds_read_tr", "read", outs=("v",), count=8, queue="tr", reads=(SlotUse("x", 0, T("t")),))
    b.wait(lgkm=4)  # 4 of the 8 reads may still be in flight
    b.op("valu", "use", ins=("v",), count=0)
    b.wait(lgkm=0)
    s = Schedule(
        "tiny",
        "gfx950",
        {"x": Buffer("x", "lds", "X", 1, 64)},
        {"dma": Queue("dma", "vm"), "tr": Queue("tr", "lgkm")},
        [Section("p", b.ops)],
        bases=lambda n: {"t": 0},
    )
    assert "SYNC.REG_NOT_READY" in verify(s).rules("error")


# ----------------------------------------------------------------------------- hoist pass
def test_hoist_cannot_beat_the_hand_schedule_and_says_why():
    s = build_mha_schedule(128, True)
    h, log = hoist_async(s)
    assert verify(h).ok
    loop_dma = {k: v for k, v in cover_summary(h).items() if k.startswith("dma")}
    assert loop_dma == {k: v for k, v in cover_summary(s).items() if k.startswith("dma")}
    reasons = [st.detail for st in log.steps if st.action == "stopped" and "dma" in st.op and "[j" in st.op]
    assert reasons and all("SYNC.WAR_LDS" in r for r in reasons)  # 2 LDS slots bound the prefetch distance


def _key(sched, sec):
    return [(o.kind, str(o), tuple(sorted((k, str(v)) for k, v in o.args.items()))) for o in sched.section(sec).ops]


def test_hoist_restores_the_hand_loop_exactly(perturbed, recovered):
    """Back-edge rotation + compound moves over own-counter waits rebuild the expert loop,
    including its vmcnt(4) waits; GPU: perturbed +3.6% causal, recovered == hand (docs)."""
    hand, _ = perturbed
    back, _ = recovered
    assert _key(back, "loop") == _key(hand, "loop")
    assert _key(back, "epilogue") == _key(hand, "epilogue")


def test_hoist_never_lands_memory_ops_in_compute_clusters(recovered):
    from flydsl.sched.transform import cluster_roles

    back, _ = recovered
    roles = cluster_roles(back)
    assert all(roles[id(o)] == "memory" for o in back.section("loop").ops if o.kind == "lds_dma")


def test_hoist_recovers_the_prefetch_distance_across_the_back_edge(perturbed, recovered):
    hand, p = perturbed
    back, log = recovered
    assert verify(back).ok
    before, after, ref = cover_summary(p), cover_summary(back), cover_summary(hand)
    assert min(v[0] for k, v in before.items() if k.startswith("dma")) < 1024
    assert all(after[k][0] >= ref[k][0] for k in ref if k.startswith("dma") and k in after)
    assert any(st.action == "rotate_up" for st in log.steps)
    for n in (4, 6, 8, 12):
        assert _instances(back, n) == _instances(hand, n)


# ----------------------------------------------------------------------------- emission
class _Rec(dict):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def get(self, name, default=None):
        def fn(*ins, **kw):
            self.out.append((name, tuple(sorted(kw.items()))))
            acc = ["o0", "o1"]  # v_o is a list of accumulators, as in the kernel
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
    def __getattr__(self, name):
        return lambda *a, **k: None


def _interpret(sched, n):
    out = []
    emit_program(
        sched,
        _Rec(out),
        tile_of=lambda extra: (lambda r: r.eval({"first": 0, "end": n, **extra})),
        loop_bounds=lambda name: (3, n - 1, 2),
        loop_impl=python_loop,
        primitives=_Prim(),
        target=TGT,
    )
    return out


@pytest.mark.parametrize("n", [4, 6, 10])
def test_emitted_transformed_program_does_the_same_work(recovered, n):
    hand = build_mha_schedule(128, True)
    a, b = _interpret(hand, n), _interpret(recovered[0], n)
    assert Counter(a) == Counter(b)
    # compute consumes tiles in the same order; only the memory ops moved
    compute = ("qk", "pv", "pv_step")
    assert [x for x in a if x[0] in compute] == [x for x in b if x[0] in compute]


def test_fingerprint_distinguishes_transformed_schedules(recovered):
    assert schedule_fingerprint(build_mha_schedule(128, True)) != schedule_fingerprint(recovered[0])
    assert schedule_fingerprint(build_mha_schedule(128, True)) == schedule_fingerprint(build_mha_schedule(128, True))
