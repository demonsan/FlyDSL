# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register pins: Schedule IR declarations, PIN.* verifier rules, lowering markers, LLVM IR rewrite, ISA report.

CPU only: the lowering is driven with a fake marker, and ``flydsl/compiler/named_reg.py``
(stdlib only) is loaded by path so the test does not import the MLIR-backed compiler package.
"""

import copy
import importlib.util
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
_SPEC = importlib.util.spec_from_file_location("named_reg", _ROOT / "python" / "flydsl" / "compiler" / "named_reg.py")
named_reg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(named_reg)


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
# RegRange
# ---------------------------------------------------------------------------------------------


def test_reg_range_parse_and_print():
    r = pins.RegRange.parse("v[192:255]")
    assert (r.file, r.lo, r.hi, r.dwords) == ("v", 192, 255, 64)
    assert str(r) == "v[192:255]" and str(pins.RegRange.parse("a7")) == "a7"
    assert str(r.sub(16, 16)) == "v[208:223]"
    assert r.overlaps(pins.RegRange.parse("v255")) and not r.overlaps(pins.RegRange.parse("a[192:255]"))
    for bad in ("v[3:1]", "s[0:1]", "v[1:2", "x5"):
        with pytest.raises(ValueError):
            pins.RegRange.parse(bad)
    with pytest.raises(ValueError):
        r.sub(60, 8)


# ---------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------


def test_no_pins_no_findings():
    rep = verify(acc_schedule())
    assert not [f for f in rep.findings if f.rule.startswith("PIN.")]


def test_carried_pin_ok_is_reserved_for_the_function():
    rep = verify(acc_schedule(carried_pin="v[128:143]"))
    assert rep.ok, rep.render()
    assert "PIN.RESERVED" in rules(rep, "info")
    assert pins.carried_pins(acc_schedule(carried_pin="v[128:143]")) == {"acc": "v[128:143]"}


def test_width_must_match_dwords():
    rep = verify(acc_schedule(carried_pin="v[128:135]"))
    assert "PIN.WIDTH" in rules(rep, "error")


def test_width_without_register_class_is_split():
    rep = verify(acc_schedule(carried_pin="v[128:159]", acc_dw=32))
    assert rep.ok and "PIN.SPLIT" in rules(rep, "info")


def test_out_of_file_and_budget():
    rep = verify(acc_schedule(carried_pin="v[248:263]"))
    assert "PIN.OUT_OF_FILE" in rules(rep, "error")
    assert "PIN.BUDGET" in rules(rep, "warn")


def test_agpr_needs_an_accvgpr_file():
    s = acc_schedule(carried_pin="a[0:15]")
    assert verify(s).ok
    rep = pins.check(s, target_without_agpr())
    assert "PIN.NO_AGPR" in rules(rep, "error")


def test_alignment_and_abi():
    rep = verify(acc_schedule(carried_pin="v[129:144]"))
    assert "PIN.ALIGN" in rules(rep, "warn")
    rep = verify(acc_schedule(carried_pin="v[0:15]"))
    assert "PIN.ABI" in rules(rep, "warn")


def test_overlap_between_live_pins():
    # x is defined before the loop and read after it: live across the loop, like the carried pin
    rep = verify(acc_schedule(carried_pin="v[128:143]", op_pin="v[140:143]"))
    assert "PIN.OVERLAP" in rules(rep, "error")
    assert "PIN.RESERVED" in rules(rep, "info")
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


def test_unknown_values_and_carried_op_pins():
    s = acc_schedule()
    s.sections[1].ops[1].pins = {"acc": "v[128:143]"}  # acc is loop-carried: pin the loop instead
    assert "PIN.CARRIED_OP_PIN" in rules(verify(s), "error")
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
    assert "PIN.ASYNC_SPILL" in rules(rep, "warn")
    assert "PIN.SCHED" in rules(rep, "info")


def test_pinned_mha_schedule_verifies():
    from kernels.attention.sched.mha_gfx950_pinned import PRESETS, build_pinned_mha_schedule

    for preset in PRESETS:
        s = build_pinned_mha_schedule(preset, head_dim=128, causal=False)
        rep = verify(s)
        assert rep.ok, rep.render()
        assert s.name.endswith(f"_pin_{preset}")
        assert pins.has_pins(s)
    base = build_pinned_mha_schedule({"v_o": "v[192:255]"}).sections
    assert [getattr(sec, "pins", {}) for sec in base if sec.loop is None] == [{}, {}]


def test_fingerprint_sees_pins():
    from flydsl.sched.program import schedule_fingerprint

    assert schedule_fingerprint(acc_schedule()) != schedule_fingerprint(acc_schedule(carried_pin="v[128:143]"))


# ---------------------------------------------------------------------------------------------
# lowering (fake markers)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def fake_markers(monkeypatch):
    calls = []

    def mark(leaf, reg, mode):
        calls.append((mode, str(reg), leaf))
        return f"{mode}({leaf})@{reg}"

    monkeypatch.setattr(pins, "marker", mark)
    monkeypatch.setattr(pins, "leaf_dwords", lambda leaf: 16)
    return calls


def test_pin_value_splits_leaves(fake_markers):
    out = pins.pin_value(["o0", ("o1", "o2"), "o3"], "v[192:255]", "write")
    assert out == ["write(o0)@v[192:207]", ("write(o1)@v[208:223]", "write(o2)@v[224:239]"), "write(o3)@v[240:255]"]
    with pytest.raises(ValueError):
        pins.pin_value(["o0", "o1"], "v[192:255]", "read")  # 32 of 64 dwords


def test_carried_pin_codecs_through_emit_program(fake_markers):
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
    modes = [m for m, _, _ in fake_markers]
    # init written before the loop; per iteration read at the top, written at the end; read after the loop
    assert modes == ["write", "read", "write", "read", "write", "read"]
    assert {r for _, r, _ in fake_markers} == {"v[128:143]"}
    assert log == [
        "read(write(mma(a1,read(write(mma(a0,read(write(zero)@v[128:143])@v[128:143]))@v[128:143])@v[128:143]))@v[128:143])@v[128:143]"
    ]


def test_op_pins_read_at_every_use(fake_markers):
    env = {}
    op_def = Op(kind="valu", binding="f", outs=("p", "q"), pins={"p": "v[64:79]"})
    env.update(p="P", q="Q")
    pins.pin_outputs(op_def, env)
    assert env["p"] == "write(P)@v[64:79]" and env["q"] == "Q"
    use = Op(kind="valu", binding="g", ins=("p", "q"), outs=("p",))
    assert pins.pin_inputs(use, env) == ["read(write(P)@v[64:79])@v[64:79]", "Q"]
    env["p"] = "P2"  # redefined without a pin: later reads are plain
    pins.pin_outputs(use, env)
    assert pins.pin_inputs(Op(kind="valu", binding="h", ins=("p",)), env) == ["P2"]


def test_op_pins_through_lower_emit(fake_markers):
    from flydsl.sched.lower import emit

    b = Builder()
    b.op("valu", "f", outs=("p",), out_dwords={"p": 16})
    b.ops[0].pins = {"p": "v[64:79]"}
    b.op("valu", "g", ins=("p",), outs=("q",))
    seen = []
    emit(b.ops, {"f": lambda: "P", "g": lambda p: seen.append(p) or "Q"}, {}, primitives=object())
    assert seen == ["read(write(P)@v[64:79])@v[64:79]"]


def test_no_pins_is_a_no_op():
    codecs = {"v_p0": (str, str)}
    assert pins.pin_codecs(acc_schedule(), codecs) is codecs
    env = {"a": 1}
    op = Op(kind="valu", binding="f", ins=("a",), outs=("b",))
    assert pins.pin_inputs(op, env) == [1]
    pins.pin_outputs(op, env)
    assert pins.LIVE_KEY not in env


# ---------------------------------------------------------------------------------------------
# LLVM IR rewrite
# ---------------------------------------------------------------------------------------------

_LL = """\
define amdgpu_kernel void @k(ptr addrspace(1) %0) {
  %2 = call <16 x float> asm ";flydsl.pin.write v[128:143]", "=v,0"(<16 x float> zeroinitializer)
  br label %3
3:
  %4 = phi <16 x float> [ %2, %1 ], [ %8, %3 ]
  %5 = call <16 x float> asm ";flydsl.pin.read v[128:143]", "=v,0"(<16 x float> %4), !dbg !7
  %6 = fadd <16 x float> %5, %5
  %8 = call <16 x float> asm ";flydsl.pin.write v[128:143]", "=v,0"(<16 x float> %6)
  %9 = call i32 asm sideeffect ";flydsl.pin.read a5", "=v,0"(i32 %x)
  %10 = call <8 x bfloat> asm ";flydsl.pin.write v[4:7]", "=v,0"(<8 x bfloat> splat (bfloat 0xR3F80))
  br label %3
}
!7 = !{}
"""


def test_rewrite_markers_to_named_register_intrinsics():
    res = named_reg.rewrite_markers(_LL)
    t = res.text
    assert (res.reads, res.writes) == (2, 3)
    assert res.regs == {"v[128:143]": 16, "a5": 1, "v[4:7]": 4}
    assert ";flydsl.pin" not in t
    assert "%pin_2_i = bitcast <16 x float> zeroinitializer to <16 x i32>" in t
    assert "call void @llvm.write_register.v16i32(metadata !8, <16 x i32> %pin_2_i)" in t
    assert "%2 = bitcast <16 x float> zeroinitializer to <16 x float>" in t
    assert "%pin_5_r = call <16 x i32> @llvm.read_register.v16i32(metadata !8), !dbg !7" in t
    assert "%5 = bitcast <16 x i32> %pin_5_r to <16 x float>, !dbg !7" in t
    assert "%pin_9_r = call i32 @llvm.read_register.i32(metadata !9)" in t  # i32 needs no bitcast type change
    assert "bitcast <8 x bfloat> splat (bfloat 0xR3F80) to <4 x i32>" in t
    assert "declare void @llvm.write_register.v16i32(metadata, <16 x i32>)" in t
    assert "declare <16 x i32> @llvm.read_register.v16i32(metadata)" in t
    assert "declare void @llvm.write_register.v4i32(metadata, <4 x i32>)" in t
    for i, reg in ((8, "v[128:143]"), (9, "a5"), (10, "v[4:7]")):
        assert f'!{i} = !{{!"{reg}"}}' in t


def test_rewrite_strip_and_errors():
    t = named_reg.rewrite_markers(_LL, mode="strip").text
    assert "read_register" not in t and ";flydsl.pin" not in t
    assert "%5 = bitcast <16 x float> %4 to <16 x float>, !dbg !7" in t
    assert named_reg.ir_type_dwords("<8 x bfloat>") == 4 and named_reg.ir_type_dwords("double") == 2
    with pytest.raises(ValueError):
        named_reg.ir_type_dwords("<3 x half>")
    with pytest.raises(ValueError):
        named_reg.rewrite_markers(_LL, mode="nope")


# ---------------------------------------------------------------------------------------------
# ISA checks
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
    rep = named_reg.isa_pin_report(_ASM, ["v[128:131]", "v[132:147]", "v[200:201]"], kernel="k")
    acc = rep["v[128:131]"].summary()
    assert acc["placed"] and acc["in_place"] == {"v_mfma_f32_16x16x32_bf16": 1}
    assert acc["copies_in"] == acc["copies_out"] == 0  # the immediate init is not a copy
    assert acc["readers"]["buffer_store_dwordx4"] == 1
    other = rep["v[132:147]"].summary()
    assert (other["copies_in"], other["copies_out"]) == (1, 1)
    assert other["copy_blocks"] == {".LBB0_1": 2}
    assert not rep["v[200:201]"].placed


def test_pending_lds_uses():
    ok = "ds_read_b64_tr_b16 v[4:5], v1\ns_waitcnt lgkmcnt(0)\nv_mov_b64_e32 v[8:9], v[4:5]\n"
    bad = "ds_read_b64_tr_b16 v[4:5], v1\nscratch_store_dwordx2 off, v[4:5], off\ns_waitcnt lgkmcnt(0)\n"
    partial = (
        "ds_read_b128 v[4:7], v1\nds_read_b128 v[8:11], v1\ns_waitcnt lgkmcnt(1)\nv_mov_b32 v0, v4\nv_mov_b32 v0, v8\n"
    )
    assert named_reg.pending_lds_uses(ok) == []
    assert named_reg.pending_lds_uses(bad) == ["scratch_store_dwordx2 off, v[4:5], off"]
    assert named_reg.pending_lds_uses(partial) == ["v_mov_b32 v0, v8"]


def test_schedule_copy_is_independent():
    s = acc_schedule()
    c = copy.deepcopy(s)
    c.sections[1].pins = {"acc": "v[128:143]"}
    assert not pins.has_pins(s) and pins.has_pins(c)


def test_validated_pin_llvm_is_recorded_consistently():
    """The LLVM revision pins were validated with is stated once in code and matches pins / docs."""
    rec = named_reg.PIN_LLVM
    assert rec["branch"] == "amdgpu-named-reg-pin-std" and rec["llvm_version"] == "24.0.0git"
    assert len(rec["commit"]) == 40 and rec["commit"] in pins.BRANCH
    doc = (_ROOT / "docs" / "sched_ir_guide.md").read_text(encoding="utf-8")
    for key in ("commit", "upstream_base", "llvm_version"):
        assert rec[key] in doc
