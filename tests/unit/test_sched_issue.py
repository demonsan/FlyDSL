# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Issue programs (flydsl.sched.issue): planning, checks, resolve/emit, calibration, conform.

CPU only: synthetic schedules and synthetic ISA snippets in the form of
``*_final_isa.s`` with ``emit(tag=...)`` debug locations.
"""

import json
import sys

import pytest

from flydsl.sched.__main__ import main as cli
from flydsl.sched.conform import conform, parse_isa
from flydsl.sched.ir import Builder, Schedule, Section
from flydsl.sched.issue import Calibration, calibrate, issue_args, resolve
from flydsl.sched.lower import emit
from flydsl.sched.verify import verify

pytestmark = pytest.mark.l0_backend_agnostic

CAL = {
    "by_binding": {
        "mm": {"mfma": 4},
        "exp": {"trans": 6},
        "sum": {"valu": 6},
        "dep": {"valu": 4},
        "br": {"valu": 2, "branch": 1},
        "pk": {"valu": 6, "raw:valu": 12},
    }
}


def _sched(*body, gap=(2, 4), lead=0, mode="template", slots=None):
    """One section: an MFMA op, then the given (binding, ins, outs) fillers, then an issue op."""
    b = Builder()
    b.op("mfma", "mm", ins=("a",), outs=("s",), count=4, mfma="mfma_f32_32x32x16_bf16")
    for binding, ins, outs in body:
        b.op("valu", binding, ins=ins, outs=outs)
    kw = {"slots": slots, "gap": gap} if slots else {"gap": gap, "lead": lead}
    b.issue(mode, group=3, **kw)
    b.barrier()
    return Schedule("t", "gfx950", {}, {}, [Section("main", b.ops)])


def _resolve(sched, cal=CAL):
    return resolve(sched, Calibration.from_dict(cal))


def _info(sched):
    return next(o.args["issue"] for o in sched.section("main").ops if o.kind == "sched_group")


def _groups(sched):
    return [(o.args["cls"], o.args["size"]) for o in sched.section("main").ops if o.kind == "sched_group"]


# ------------------------------------------------------------------------- program / plan
def test_issue_args_validate():
    with pytest.raises(ValueError, match="template|explicit"):
        issue_args("loose")
    with pytest.raises(ValueError, match="cannot be selected"):
        issue_args("template", fill=("ds_read_tr",))
    with pytest.raises(ValueError, match="need slots"):
        issue_args("explicit")


def test_template_spreads_fillers_between_spine_instructions():
    s, rep = _resolve(_sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",))))
    assert rep.ok, rep.render()
    # 12 fillers over 4 MFMAs: 3 after each, TRANS (earlier op) before VALU
    assert _groups(s) == [("mfma", 1), ("trans", 3), ("mfma", 1), ("trans", 3), ("mfma", 1), ("valu", 3),
                          ("mfma", 1), ("valu", 3)]  # fmt: skip
    info = _info(s)
    assert info["gaps"] == [3, 3, 3, 3] and info["id"] == "main/3"
    assert all(o.args["group"] == 3 for o in s.section("main").ops if o.kind == "sched_group")


def test_lead_puts_fillers_before_the_first_spine_instruction():
    s, rep = _resolve(_sched(("exp", ("x",), ("x",)), lead=2, gap=(1, 4)))
    assert rep.ok and _groups(s)[0] == ("trans", 2)


def test_gap_bounds_are_checked():
    _, rep = _resolve(_sched(("exp", ("x",), ("x",)), gap=(3, 6)))  # 6 fillers / 4 MFMAs
    assert "ISSUE.GAP" in rep.rules("error")


def test_tail_takes_the_overflow_when_gaps_would_exceed_the_bound():
    s, rep = _resolve(_sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",)), ("dep", ("y",), ("z",)), gap=(1, 3)))
    assert rep.ok, rep.render()
    gaps = _info(s)["gaps"]
    assert max(gaps[:-1]) <= 3 and gaps[-1] > 3 and "ISSUE.GAP" in rep.rules("warn")


def test_raw_weights_balance_unpacked_instructions():
    # 6 units that issue as 12 instructions (unpacked v_pk_*_f32) vs 6 plain TRANS
    s, rep = _resolve(_sched(("exp", ("x",), ("x",)), ("pk", ("x",), ("y",)), gap=(2, 6)))
    assert rep.ok, rep.render()
    units = _info(s)["gaps"]
    assert sum(units) == 12 and units[0] > units[-1]  # fewer pk units per gap than TRANS


def test_control_flow_and_uncalibrated_ops_are_errors():
    _, rep = _resolve(_sched(("br", ("x",), ("x",))))
    assert "ISSUE.CONTROL_FLOW" in rep.rules("error")
    _, rep = _resolve(_sched(("unknown", ("x",), ("x",))))
    assert "ISSUE.UNCALIBRATED" in rep.rules("error")


def test_filler_reading_the_spine_must_issue_after_it():
    # "dep" reads s, the MFMA's result: it cannot sit in the first gaps
    _, rep = _resolve(_sched(("dep", ("s",), ("z",)), ("exp", ("x",), ("x",)), gap=(1, 6)))
    assert "ISSUE.DEPENDS" in rep.rules("error")
    f = [x for x in rep.findings if x.rule == "ISSUE.DEPENDS"][0]
    assert "reads the result of mfma #4" in f.message


def test_spine_reading_a_late_filler_is_an_error():
    b = Builder()
    b.op("valu", "exp", ins=("x",), outs=("a",))
    b.op("mfma", "mm", ins=("a",), outs=("s",), count=4)
    b.issue(group=0, gap=(0, 6))
    s = Schedule("t", "gfx950", {}, {}, [Section("main", b.ops)])
    _, rep = _resolve(s)
    assert "ISSUE.DEPENDS" in rep.rules("error")
    b.ops[-1].args["lead"] = 6
    _, rep = _resolve(s)
    assert rep.ok, rep.render()


def test_explicit_program_counts_are_checked():
    good = [("mfma", 1), ("trans", 2), ("mfma", 1), ("trans", 2), ("mfma", 1), ("trans", 2), ("mfma", 1)]
    s, rep = _resolve(_sched(("exp", ("x",), ("x",)), mode="explicit", slots=good))
    assert rep.ok and _groups(s) == good
    bad = good[:-1] + [("mfma", 2)]
    _, rep = _resolve(_sched(("exp", ("x",), ("x",)), mode="explicit", slots=bad))
    assert "ISSUE.COUNT" in rep.rules("error")


def test_inline_asm_fillers_are_reported():
    cal = {"by_binding": dict(CAL["by_binding"], exp={"trans": 8, "asm:valu": 4})}
    _, rep = _resolve(_sched(("exp", ("x",), ("x",))), cal)
    assert "ISSUE.ASM" in rep.rules("warn")


class _Rec:
    def __init__(self, out):
        self.out = out

    def __getattr__(self, name):
        return lambda *a: self.out.append((name,) + a)


def test_emit_lowers_resolved_programs_and_refuses_unresolved_ones():
    sched = _sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",)))
    bindings = {n: (lambda *a, **k: n) for n in ("mm", "exp", "sum")}
    with pytest.raises(ValueError, match="unresolved issue program"):
        emit(sched.section("main"), bindings, {"a": 1, "x": 2}, primitives=_Rec([]))
    s, _ = _resolve(sched)
    out = []
    emit(s.section("main"), bindings, {"a": 1, "x": 2}, primitives=_Rec(out))
    assert [o for o in out if o[0] == "sched_group"][:2] == [
        ("sched_group", "mfma", 1, 3),
        ("sched_group", "trans", 3, 3),
    ]


# ------------------------------------------------------------------------- calibrate / conform
def _isa(rows, name="k"):
    out = ["\t.text", f"\t.type\t{name},@function", f"{name}:", '\t.file\t1 "sched:t" "main"']
    for op, text in rows:
        out += [f"\t.loc\t1 {op + 1} 0", "\t" + text]
    out += [".Lfunc_end0:", f"\t.size\t{name}, .Lfunc_end0-{name}"]
    return "\n".join(out) + "\n"


M = "v_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]"
T = "v_exp_f32_e32 v40, v41"
PK = ["v_add_f32_e64 v50, v60, v70", "v_add_f32_e64 v51, v61, v71"]  # one unpacked v_pk_add_f32


def test_calibrate_counts_units_and_keeps_the_layout():
    sched = _sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",)))
    rows = [(0, M), (1, T), (1, T), (2, PK[0]), (2, PK[1]), (0, M)]
    cal = calibrate(parse_isa(_isa(rows)), sched)
    assert cal.by_binding["mm"] == {"mfma": 2}
    assert cal.by_binding["sum"] == {"valu": 1, "raw:valu": 2}
    assert cal.layout == {"main": ["mm", "exp", "sum"]} and cal.matches(sched.section("main"))
    # exact per-op counts are used for a section with the same layout
    cal.by_op["main#1"] = {"trans": 4}
    s, rep = resolve(sched, cal)
    assert rep.ok and sum(n for c, n in _groups(s) if c == "trans") == 4


def test_conform_reports_achieved_against_requested_issue_order():
    sched, _ = _resolve(_sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",))))
    ops = sched.section("main").ops
    barrier = len(ops) - 1
    V = "v_mul_f32_e32 v80, v81, v82"
    good = [(0, M), (1, T), (1, T), (1, T), (0, M), (1, T), (1, T), (1, T), (0, M), (2, V), (2, V), (2, V), (0, M),
            (2, V), (2, V), (2, V), (barrier, "s_barrier")]  # fmt: skip
    rep = conform(parse_isa(_isa(good)), sched)
    f = [x for x in rep.findings if x.rule == "CONFORM.ISSUE"]
    assert len(f) == 1 and f[0].severity == "info" and "exact" in f[0].message and "0 back-to-back" in f[0].message
    bad = [(0, M), (0, M)] + [r for r in good[1:] if r[1] != M]
    bad = bad[:-1] + [(0, M), (0, M), (barrier, "s_barrier")]
    rep = conform(parse_isa(_isa(bad)), sched)
    f = [x for x in rep.findings if x.rule == "CONFORM.ISSUE"][0]
    assert f.severity == "warn" and "diverges" in f.message and "2 back-to-back" in f.message


def test_unpacked_pairs_are_one_unit():
    k = parse_isa(_isa([(0, PK[0]), (0, PK[1]), (0, "v_add_f32_e64 v53, v62, v72")]))
    assert [i.unit for i in k.insts] == [1, 0, 1]


# ------------------------------------------------------------------------- MHA schedule
def test_mha_issue_variants_resolve_and_verify():
    from kernels.attention.sched.mha_gfx950 import build_mha_schedule

    for kw in ({}, {"issue_split": False}, {"rescale": "deferred"}):
        for causal in (False, True):
            s = build_mha_schedule(128, causal, p_cast="cvt", issue="default", **kw)
            assert verify(s).ok and s.meta["issue_report"].ok
            assert not any(o.kind == "issue" for o in s.all_ops())
            assert any("issue" in o.args for o in s.all_ops() if o.kind == "sched_group")


def test_mha_issue_reports_asm_casts_and_needs_d128():
    from kernels.attention.sched.mha_gfx950 import CALIBRATION_D128, build_mha_schedule

    # inline-asm casts float outside the chain: reported, not silently counted
    cal = {"by_binding": dict(CALIBRATION_D128["by_binding"], cast_p={"asm:valu": 16}, anchor_p={})}
    with pytest.raises(ValueError, match="ISSUE.GAP") as e:
        build_mha_schedule(128, False, p_cast="asm", issue=cal)
    assert "ISSUE.ASM" in str(e.value)  # the 16 asm casts are not counted as fillers
    with pytest.raises(ValueError, match="head_dim=128"):
        build_mha_schedule(64, False, p_cast="cvt", issue="default")
    plain = build_mha_schedule(128, False)
    assert not any("issue" in o.args for o in plain.all_ops()) and plain.name.endswith("_bf16")


def test_mha_deferred_rescale_carries_the_correction():
    from kernels.attention.sched.mha_gfx950 import build_mha_schedule

    s = build_mha_schedule(128, False, rescale="deferred")
    assert s.section("loop").carried[-1] == "corr" and verify(s).ok
    applies = [o for o in s.all_ops() if o.binding == "scale_o_if"]
    assert len(applies) == 3  # C0, C4 and E0
    assert not any(o.binding == "lazy_rescale_o" for o in s.all_ops())


# ------------------------------------------------------------------------- CLI
def test_cli_calibrate_and_issue(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "_issue_sched_for_test", sys.modules[__name__])
    isa = tmp_path / "k.s"
    isa.write_text(_isa([(0, M), (1, T), (2, PK[0]), (2, PK[1])]))
    out = tmp_path / "cal.json"
    spec = "_issue_sched_for_test:_cli_sched"
    assert cli(["calibrate", str(isa), "--schedule", spec, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["by_binding"]["sum"] == {"raw:valu": 2, "valu": 1}
    assert "mm" in capsys.readouterr().out
    out.write_text(json.dumps(CAL))
    assert cli(["issue", spec, "--calib", str(out)]) == 0
    assert "ISSUE.PLAN" in capsys.readouterr().out


def _cli_sched():
    return _sched(("exp", ("x",), ("x",)), ("sum", ("x",), ("y",)))
