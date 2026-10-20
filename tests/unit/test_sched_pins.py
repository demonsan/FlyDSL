# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register pins: Schedule IR declarations, PIN.* verifier rules, lowering to hints, ISA report.

CPU only: the lowering is driven with a fake hint emitter.  The hint ops themselves
(``rocdl.sched.{vgpr,agpr,sgpr}``) are covered by ``tests/unit/test_schedule_gpr.py``.
"""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

from flydsl.sched import pins
from flydsl.sched.ir import Builder, Op, Queue, Schedule, Section
from flydsl.sched.ir import T as Tile
from flydsl.sched.program import emit_program, python_loop
from flydsl.sched.target import TargetDesc, load_target
from flydsl.sched.verify import verify

pytestmark = [pytest.mark.l0_backend_agnostic]

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def acc_schedule(carried_pin=None, op_pin=None, acc_dw=16, use_after=True):
    """prologue: init acc, load x | loop k: load a, mma(a, acc) | epilogue: store acc (and x)."""
    pro = Builder()
    pro.op("valu", "init", outs=("acc",), out_dwords={"acc": acc_dw})
    pro.op("vmem_load", "load_x", outs=("x",), queue="gld", out_dwords={"x": 4})
    if op_pin:
        pro.ops[-1].pins = {"x": op_pin}
    body = Builder()
    body.op("vmem_load", "load_a", outs=("a",), args={"k": Tile("k")}, queue="gld", out_dwords={"a": 4})
    body.op("mfma", "mma", ins=("a", "acc"), outs=("acc",), mfma="mfma_f32_32x32x16_bf16", out_dwords={"acc": acc_dw})
    ep = Builder()
    ep.op("vmem_store", "store", ins=("acc", "x") if use_after else ("acc",), count=1)
    loop = Section("k", body.ops, loop=("k", lambda bases: range(0, 4)), carried=("acc",))
    if carried_pin:
        loop.pins = {"acc": carried_pin}
    return Schedule(
        name="pin_test",
        target="gfx950",
        buffers={},
        queues={"gld": Queue("gld", "vm", tracked=True)},
        sections=[Section("prologue", pro.ops), loop, Section("epilogue", ep.ops)],
        meta={"verify_tiles": (4,), "waves_per_eu": 2},
    )


def rules(report, severity=None):
    return set(report.rules(severity))


def target_without_agpr():
    data = {"arch": "gfx950", "facts": {k: {"value": f.value} for k, f in load_target("gfx950")._facts.items()}}
    data["facts"]["acc_vgpr_max_per_wave"] = {"value": 0}
    return TargetDesc(data)


# ---------------------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------------------


def test_reg_range_and_spec_parse():
    r = pins.RegRange.parse("v[192:255]")
    assert (r.file, r.lo, r.hi, r.dwords) == ("v", 192, 255, 64)
    assert str(r) == "v[192:255]" and str(pins.RegRange.parse("a7")) == "a7"
    assert str(pins.RegRange.parse("s[4:5]")) == "s[4:5]"
    assert str(r.sub(16, 16)) == "v[208:223]"
    assert r.overlaps(pins.RegRange.parse("v255")) and not r.overlaps(pins.RegRange.parse("a[192:255]"))
    for bad in ("v[3:1]", "x[0:1]", "v[1:2", "x5"):
        with pytest.raises(ValueError):
            pins.RegRange.parse(bad)
    with pytest.raises(ValueError):
        r.sub(60, 8)
    assert pins.PinSpec.parse("v[8:11]") == pins.PinSpec(pins.RegRange("v", 8, 11), True)
    assert pins.PinSpec.parse("v[8:11] soft").strict is False and str(pins.PinSpec.parse("v8 soft")) == "v8 soft"
    with pytest.raises(ValueError):
        pins.PinSpec.parse("v8 maybe")


# ---------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------


def test_no_pins_no_findings():
    rep = verify(acc_schedule())
    assert not [f for f in rep.findings if f.rule.startswith("PIN.")]


def test_carried_pin_ok():
    rep = verify(acc_schedule(carried_pin="v[128:143]"))
    assert rep.ok, rep.render()
    assert "PIN.CARRIED" in rules(rep, "info") and "PIN.RESERVED" not in rules(rep)
    assert pins.carried_pins(acc_schedule(carried_pin="v[128:143]")) == {"acc": "v[128:143]"}


def test_width_must_match_dwords():
    assert "PIN.WIDTH" in rules(verify(acc_schedule(carried_pin="v[128:135]")), "error")


def test_widths_without_hint_are_split():
    rep = verify(acc_schedule(carried_pin="v[128:159]", acc_dw=32))
    assert rep.ok and "PIN.SPLIT" in rules(rep, "info")
    rep = verify(acc_schedule(carried_pin="v[128:130]", acc_dw=3))  # 96 bits: no B96 hint
    assert "PIN.SPLIT" in rules(rep, "info")


def test_out_of_file_and_budget():
    rep = verify(acc_schedule(carried_pin="v[248:263]"))
    assert "PIN.OUT_OF_FILE" in rules(rep, "error") and "PIN.BUDGET" in rules(rep, "warn")
    s = acc_schedule()
    s.sections[0].ops[0].pins = {"acc": "s[100:115]"}
    assert "PIN.OUT_OF_FILE" in rules(verify(s), "error")


def test_agpr_needs_an_accvgpr_file():
    s = acc_schedule(carried_pin="a[0:15]")
    assert verify(s).ok
    assert "PIN.NO_AGPR" in rules(pins.check(s, target_without_agpr()), "error")


def test_alignment_strict_vs_soft_and_abi():
    assert "PIN.ALIGN" in rules(verify(acc_schedule(carried_pin="v[129:144]")), "error")
    assert "PIN.ALIGN" in rules(verify(acc_schedule(carried_pin="v[129:144] soft")), "warn")
    assert "PIN.ABI" in rules(verify(acc_schedule(carried_pin="v[0:15]")), "warn")


def test_overlap_between_live_pins():
    # x is defined before the loop and read after it: live across the loop, like the carried acc
    rep = verify(acc_schedule(carried_pin="v[128:143]", op_pin="v[140:143]"))
    assert "PIN.OVERLAP" in rules(rep, "error")
    rep = verify(acc_schedule(carried_pin="v[128:143]", op_pin="v[140:143] soft"))
    assert "PIN.OVERLAP" in rules(rep, "warn") and rep.ok
    assert verify(acc_schedule(carried_pin="v[128:143]", op_pin="v[144:147]")).ok


def test_overlap_respects_lifetimes():
    b = Builder()
    b.op("valu", "f", outs=("p",), out_dwords={"p": 4})
    b.op("valu", "g", ins=("p",), outs=("q",), out_dwords={"q": 4})
    b.op("valu", "h", ins=("q",), outs=("r",), out_dwords={"r": 4})
    b.op("valu", "use", ins=("r",))
    b.ops[0].pins = {"p": "v[64:67]"}
    b.ops[2].pins = {"r": "v[64:67]"}  # p is dead once g has read it
    s = Schedule("lt", "gfx950", {}, {}, [Section("s", b.ops)])
    assert verify(s).ok
    b.ops[1].pins = {"q": "v[66:69]"}  # q is live while r is written: overlap
    assert "PIN.OVERLAP" in rules(verify(s), "error")


def test_unknown_values_and_conflicts():
    s = acc_schedule(carried_pin="v[128:143]")
    s.sections[1].ops[1].pins = {"acc": "v[128:143]"}  # an op pin agreeing with the loop: fine
    assert verify(s).ok
    s.sections[1].ops[1].pins = {"acc": "v[144:159]"}
    assert "PIN.CONFLICT" in rules(verify(s), "error")
    s = acc_schedule()
    s.sections[0].ops[0].pins = {"nope": "v[128:143]"}
    assert "PIN.UNKNOWN_VALUE" in rules(verify(s), "error")
    s = acc_schedule()
    s.sections[1].pins = {"x": "v[128:131]"}
    assert "PIN.UNKNOWN_VALUE" in rules(verify(s), "error")
    s = acc_schedule()
    s.sections[0].ops[0].pins = {"acc": "v[1:x]"}
    assert "PIN.SYNTAX" in rules(verify(s), "error")


def test_async_and_sched_interactions():
    s = acc_schedule(carried_pin="v[128:143]")
    s.queues["gld"] = Queue("gld", "vm", tracked=False)
    s.sections[1].ops.insert(0, Op(kind="wait", label="w", args={"vm": 0}))
    s.sections[1].ops.append(Op(kind="sched_group", label="sg", args={"cls": "mfma", "size": 1, "group": 0}))
    rep = pins.check(s)
    assert "PIN.ASYNC_SPILL" in rules(rep, "warn") and "PIN.SCHED" in rules(rep, "info")
    s.sections[1].pins = {"acc": "v[128:143] soft"}
    assert "PIN.ASYNC_SPILL" not in rules(pins.check(s))  # only strict hints restrict the allocator


def test_pinned_mha_schedule_verifies():
    from kernels.attention.sched.mha_gfx950_pinned import PRESETS, build_pinned_mha_schedule

    assert {"o", "ml", "o_soft", "ml_soft"} <= set(PRESETS)
    for preset in PRESETS:
        s = build_pinned_mha_schedule(preset, head_dim=128, causal=False)
        rep = verify(s)
        assert rep.ok, rep.render()
        assert s.name.endswith(f"_pin_{preset}") and pins.has_pins(s)
    base = build_pinned_mha_schedule({"v_o": "v[192:255]"}).sections
    assert [getattr(sec, "pins", {}) for sec in base if sec.loop is None] == [{}, {}]


def test_fingerprint_sees_pins():
    from flydsl.sched.program import schedule_fingerprint

    assert schedule_fingerprint(acc_schedule()) != schedule_fingerprint(acc_schedule(carried_pin="v[128:143]"))
    assert schedule_fingerprint(acc_schedule(carried_pin="v[128:143]")) != schedule_fingerprint(
        acc_schedule(carried_pin="v[128:143] soft")
    )


# ---------------------------------------------------------------------------------------------
# lowering (fake hints)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def fake_hints(monkeypatch):
    calls = []

    def mark(leaf, reg, strict):
        calls.append((str(reg), strict, leaf))
        return f"h({leaf})@{reg}{'' if strict else '~'}"

    monkeypatch.setattr(pins, "hint", mark)
    monkeypatch.setattr(pins, "leaf_dwords", lambda leaf: 16)
    monkeypatch.setattr(pins, "support_error", lambda: "")
    return calls


def test_pin_value_splits_leaves(fake_hints):
    out = pins.pin_value(["o0", ("o1", "o2"), "o3"], "v[192:255] soft")
    assert out == ["h(o0)@v[192:207]~", ("h(o1)@v[208:223]~", "h(o2)@v[224:239]~"), "h(o3)@v[240:255]~"]
    with pytest.raises(ValueError):
        pins.pin_value(["o0", "o1"], "v[192:255]")  # 32 of 64 dwords


def test_carried_pin_codecs_through_emit_program(fake_hints):
    sched = acc_schedule(carried_pin="v[128:143]")
    log = []
    bindings = {
        "init": lambda: "zero",
        "load_x": lambda: "x",
        "load_a": lambda k: f"a{k}",
        "mma": lambda a, acc: f"mma({a},{acc})",
        "store": lambda acc, x: log.append(acc),
    }
    emit_program(
        sched,
        bindings,
        tile_of=lambda extra: (lambda ref: extra[ref.base] + ref.off),
        loop_bounds=lambda name: (0, 2, 1),
        loop_impl=python_loop,
        primitives=object(),
    )
    # init hinted before the loop; per iteration the argument at the top and the new value at the end; the result
    assert [r for r, _, _ in fake_hints] == ["v[128:143]"] * 6 and all(s for _, s, _ in fake_hints)
    assert log == [
        "h(h(mma(a1,h(h(mma(a0,h(h(zero)@v[128:143])@v[128:143]))@v[128:143])@v[128:143]))@v[128:143])@v[128:143]"
    ]


def test_op_pins_hint_the_definition(fake_hints):
    env = {"p": "P", "q": "Q"}
    op = Op(kind="valu", binding="f", outs=("p", "q"), pins={"p": "v[64:79]"})
    pins.pin_outputs(op, env)
    assert env == {"p": "h(P)@v[64:79]", "q": "Q"}
    use = Op(kind="valu", binding="g", ins=("p", "q"))
    assert pins.pin_inputs(use, env) == ["h(P)@v[64:79]", "Q"]  # reads need nothing


def test_op_pins_through_lower_emit(fake_hints):
    from flydsl.sched.lower import emit

    b = Builder()
    b.op("valu", "f", outs=("p",), out_dwords={"p": 16})
    b.ops[0].pins = {"p": "v[64:79]"}
    b.op("valu", "g", ins=("p",), outs=("q",))
    seen = []
    emit(b.ops, {"f": lambda: "P", "g": lambda p: seen.append(p) or "Q"}, {}, primitives=object())
    assert seen == ["h(P)@v[64:79]"]


def test_no_pins_is_a_no_op():
    codecs = {"v_p0": (str, str)}
    assert pins.pin_codecs(acc_schedule(), codecs) is codecs
    env = {"a": 1}
    op = Op(kind="valu", binding="f", ins=("a",), outs=("b",))
    assert pins.pin_inputs(op, env) == [1]
    pins.pin_outputs(op, env)
    assert env == {"a": 1}


def test_support_error_names_the_missing_ops(monkeypatch):
    fake = types.ModuleType("flydsl.expr.rocdl")
    fake.sched_vgpr = object()
    expr = types.ModuleType("flydsl.expr")
    expr.rocdl = fake
    monkeypatch.setitem(sys.modules, "flydsl.expr", expr)
    monkeypatch.setitem(sys.modules, "flydsl.expr.rocdl", fake)
    assert "sched_agpr" in pins.support_error() and "sched_sgpr" in pins.support_error()
    fake.sched_agpr = fake.sched_sgpr = object()
    assert pins.support_error() == ""


def test_required_llvm_patch_is_declared():
    info = json.loads((_ROOT / "thirdparty" / "llvm-build-info.json").read_text(encoding="utf-8"))["upstream"]
    assert "schedule-bank-and-gpr.patch" in [e["file"] for e in info["required_patches"]]
    patch = (_ROOT / "thirdparty" / "llvm-extensions" / "schedule-bank-and-gpr.patch").read_text(encoding="utf-8")
    for name in ("int_amdgcn_sched_vgpr", "int_amdgcn_sched_agpr", "int_amdgcn_sched_sgpr", "V_SCHEDULE_VGPR_B512"):
        assert name in patch


# ---------------------------------------------------------------------------------------------
# ISA report
# ---------------------------------------------------------------------------------------------

_ASM = """\
k:
	v_mov_b64_e32 v[128:129], 0
.LBB0_1:                                ; =>This Inner Loop Header
	v_mfma_f32_16x16x32_bf16 v[128:131], v[2:5], v[6:9], v[128:131]
	v_mov_b64_e32 v[10:11], v[132:133]
	v_mfma_f32_32x32x16_bf16 v[12:27], v[2:5], v[6:9], v[12:27]
	v_mov_b64_e32 v[132:133], v[12:13]
	s_cbranch_scc1 .LBB0_1
	buffer_store_dwordx4 v[128:131], v1, s[0:3], 0 offen
	s_endpgm
.Lfunc_end0:
"""


def test_isa_pin_report():
    rep = pins.isa_pin_report(_ASM, ["v[128:131]", "v[132:147] soft", "v[200:201]"], kernel="k")
    acc = rep["v[128:131]"].summary()
    assert acc["placed"] and acc["in_place"] == {"v_mfma_f32_16x16x32_bf16": 1}
    assert acc["copies_in"] == acc["copies_out"] == 0  # the immediate init is not a copy
    assert acc["readers"]["buffer_store_dwordx4"] == 1
    other = rep["v[132:147] soft"].summary()
    assert (other["copies_in"], other["copies_out"]) == (1, 1) and other["copy_blocks"] == {".LBB0_1": 2}
    assert not rep["v[200:201]"].placed
    assert pins.mfma_accumulators(_ASM) == {"v[128:131]": 1, "v[12:27]": 1}


def test_pending_lds_uses():
    ok = "ds_read_b64_tr_b16 v[4:5], v1\ns_waitcnt lgkmcnt(0)\nv_mov_b64_e32 v[8:9], v[4:5]\n"
    bad = "ds_read_b64_tr_b16 v[4:5], v1\nscratch_store_dwordx2 off, v[4:5], off\ns_waitcnt lgkmcnt(0)\n"
    partial = (
        "ds_read_b128 v[4:7], v1\nds_read_b128 v[8:11], v1\ns_waitcnt lgkmcnt(1)\nv_mov_b32 v0, v4\nv_mov_b32 v0, v8\n"
    )
    assert pins.pending_lds_uses(ok) == []
    assert pins.pending_lds_uses(bad) == ["scratch_store_dwordx2 off, v[4:5], off"]
    assert pins.pending_lds_uses(partial) == ["v_mov_b32 v0, v8"]


def test_schedule_copy_is_independent():
    s = acc_schedule()
    c = copy.deepcopy(s)
    c.sections[1].pins = {"acc": "v[128:143]"}
    assert not pins.has_pins(s) and pins.has_pins(c)
