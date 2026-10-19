# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ISA conformance checker on small synthetic ISA snippets (no MLIR, no GPU).

The snippets mimic ``*_final_isa.s`` compiled with ``FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1``
from a kernel that called ``emit(..., tag="t")``: LLVM splits the pseudo file
``sched:t/main`` into directory ``sched:t`` and name ``main``, and ``.loc``
lines are schedule op index + 1.
"""

import sys

import pytest

from flydsl.sched.__main__ import main as cli
from flydsl.sched.conform import conform, diff_isa, op_table, parse_isa, waitcnt_values
from flydsl.sched.ir import Builder, Schedule, Section
from flydsl.sched.lower import emit, tag_file

pytestmark = pytest.mark.l0_backend_agnostic


def _schedule():
    b = Builder()
    b.op("lds_dma", "load", count=2)  # 0
    b.op("ds_read", "read", outs=("x",), count=2)  # 1
    b.wait(lgkm=0)  # 2
    b.wait(vm=2)  # 3
    b.barrier()  # 4
    b.setprio(1)  # 5
    b.op("mfma", "mm", ins=("x",), outs=("y",), count=2)  # 6
    b.op("trans", "exp", ins=("y",), outs=("y",), count=0)  # 7
    b.interleave(2, "trans", 2, group=0)  # 8..11
    b.setprio(0)  # 12
    b.barrier()  # 13
    return Schedule("t", "gfx950", {}, {}, [Section("main", b.ops)])


# One line per instruction: (op index or None for user code, instruction text).
GOOD = [
    (None, "s_load_dwordx2 s[0:1], s[4:5], 0x0"),
    (0, "buffer_load_dwordx4 v1, s[0:3], 0 offen lds"),
    (0, "buffer_load_dwordx4 v2, s[0:3], 0 offen lds"),
    (1, "ds_read_b128 v[4:7], v3"),
    (1, "ds_read_b128 v[8:11], v3 offset:16"),
    (2, "s_waitcnt lgkmcnt(0)"),
    (3, "s_waitcnt vmcnt(2)"),
    (4, "s_barrier"),
    (5, "s_setprio 1"),
    (6, "v_mfma_f32_32x32x16_bf16 v[16:31], v[4:7], v[8:11], 0"),
    (7, "v_exp_f32_e32 v40, v16"),
    (7, "v_exp_f32_e32 v41, v17"),
    (6, "v_mfma_f32_32x32x16_bf16 v[16:31], v[4:7], v[8:11], v[16:31]"),
    (7, "v_exp_f32_e32 v42, v18"),
    (7, "v_exp_f32_e32 v43, v19"),
    (12, "s_setprio 0"),
    (13, "s_barrier"),
    (None, "s_endpgm"),
]


def _isa(rows, kernarg=64, name="k"):
    out = [
        "\t.text",
        f"\t.type\t{name},@function",
        f"{name}:",
        ".Lfunc_begin0:",
        '\t.file\t1 "/src" "kernel.py"',
        "\t.loc\t1 10 0",
        '\t.file\t2 "sched:t" "main"',
    ]
    for n, (op, text) in enumerate(rows):
        out.append(f".Ltmp{n}:")
        out.append("\t.loc\t1 12 4" if op is None else f"\t.loc\t2 {op + 1} 0")
        if text.startswith("asm:"):
            out += ["\t;;#ASMSTART", "\t" + text[4:], "\t;;#ASMEND"]
        elif text.endswith(":"):
            out.append(text)
        else:
            out.append("\t" + text)
    out += [
        ".Lfunc_end0:",
        f"\t.size\t{name}, .Lfunc_end0-{name}",
        "\t.section\t.rodata",
        f".amdhsa_kernel {name}",
        "\t.amdhsa_group_segment_fixed_size 1024",
        f"\t.amdhsa_kernarg_size {kernarg}",
        "\t.amdhsa_next_free_vgpr 64",
        ".end_amdhsa_kernel",
    ]
    return "\n".join(out) + "\n"


def _check(rows, sched=None):
    return conform(parse_isa(_isa(rows)), sched or _schedule())


def _replace(rows, i, new):
    rows = list(rows)
    if new is None:
        del rows[i]
    else:
        rows[i] = new
    return rows


# ------------------------------------------------------------------------- parsing
def test_parse_attributes_instructions_to_schedule_ops():
    k = parse_isa(_isa(GOOD))
    assert k.name == "k" and k.tags == ["t"] and len(k.insts) == len(GOOD)
    assert [(i.section, i.op) for i in k.insts[1:4]] == [("main", 0), ("main", 0), ("main", 1)]
    assert k.insts[0].section is None and k.insts[0].src == ("/src/kernel.py", 12, 4)
    assert [i.cls for i in k.insts[1:10]] == [
        "lds_dma", "lds_dma", "ds_read", "ds_read", "waitcnt", "waitcnt", "barrier", "setprio", "mfma",
    ]  # fmt: skip
    assert k.insts[10].cls == "trans"
    assert k.meta["kernarg_size"] == "64" and k.meta["next_free_vgpr"] == "64"
    assert {i.block for i in k.insts} == {"k"}  # .Ltmp labels are not blocks


def test_parse_inline_asm_blocks_and_waitcnt_fields():
    rows = [(1, "asm:ds_read_b64_tr_b16 v[2:3], v1 offset:0"), (None, ".LBB0_1:"), (2, "s_waitcnt vmcnt(3) lgkmcnt(0)")]
    k = parse_isa(_isa(rows))
    assert k.insts[0].asm and k.insts[0].cls == "ds_read_tr"
    assert k.insts[1].block == ".LBB0_1" and not k.insts[1].asm
    assert waitcnt_values(k.insts[1]) == {"vm": 3, "lgkm": 0, "exp": 7}


def test_parse_picks_kernel_by_name():
    text = _isa(GOOD, name="a") + _isa(GOOD[:3], name="b")
    assert len(parse_isa(text, "b").insts) == 3
    with pytest.raises(ValueError, match="not in ISA"):
        parse_isa(text, "c")


# ------------------------------------------------------------------------- conformance
def test_conforming_isa_has_no_warnings():
    r = _check(GOOD)
    assert r.ok
    assert not [f for f in r.findings if f.severity != "info"], r.render()
    assert "CONFORM.INTERLEAVE" in r.rules("info")


def test_no_tags_is_an_error():
    rows = [(None, t) for _, t in GOOD]
    r = _check(rows)
    assert r.rules("error") == ["CONFORM.NO_TAGS"]


def test_merged_wait_is_reported_and_accepted():
    rows = _replace(_replace(GOOD, 6, None), 5, (2, "s_waitcnt vmcnt(2) lgkmcnt(0)"))
    r = _check(rows)
    assert r.ok and "CONFORM.WAIT_MERGED" in r.rules("info")
    assert "CONFORM.WAIT_TIGHTENED" not in r.rules()


def test_dropped_wait():
    r = _check(_replace(GOOD, 6, None))
    assert "CONFORM.WAIT_DROPPED" in r.rules("warn")


def test_looser_and_stricter_waits():
    r = _check(_replace(GOOD, 6, (3, "s_waitcnt vmcnt(4)")))
    assert r.rules("error") == ["CONFORM.WAIT_LOOSENED"]
    r = _check(_replace(GOOD, 6, (3, "s_waitcnt vmcnt(0)")))
    assert r.ok and "CONFORM.WAIT_TIGHTENED" in r.rules("warn")


def test_compiler_inserted_wait():
    rows = list(GOOD)
    rows.insert(12, (6, "s_waitcnt lgkmcnt(1)"))
    r = _check(rows)
    f = [f for f in r.findings if f.rule == "CONFORM.WAIT_INSERTED"]
    assert len(f) == 1 and f[0].where == "main/6"


def test_missing_barrier():
    r = _check(_replace(GOOD, 16, None))
    assert r.rules("error") == ["CONFORM.BARRIER_COUNT"]


def test_code_moved_across_a_fence():
    rows = list(GOOD)
    rows.insert(5, rows.pop(9))  # first MFMA hoisted above the waits and the barrier
    r = _check(rows)
    assert "CONFORM.ORDER" in r.rules("warn")
    assert "CONFORM.SETPRIO_PLACEMENT" in r.rules("warn")  # and above s_setprio 1


def test_code_in_another_block_is_not_an_order_violation():
    rows = list(GOOD)
    tail = rows.pop(13)  # one exp2 placed out of line after the kernel's last barrier
    rows[-1:-1] = [(None, ".LBB0_9:"), tail]
    r = _check(rows)
    assert "CONFORM.ORDER" not in r.rules()


def test_interleave_divergence():
    rows = _replace(_replace(GOOD, 10, (6, GOOD[12][1])), 12, (7, GOOD[10][1]))  # M M T T T T
    r = _check(rows)
    f = [f for f in r.findings if f.rule == "CONFORM.INTERLEAVE"]
    assert f and f[0].severity == "warn" and "issued M2 T4" in f[0].message


def test_interleave_with_too_few_instructions_is_not_a_divergence():
    rows = _replace(GOOD, 14, None)  # 3 exp2 for 4 TRANS slots
    r = _check(rows)
    f = [f for f in r.findings if f.rule == "CONFORM.INTERLEAVE"][0]
    assert f.severity == "info" and "'trans': (4, 3)" in f.message


def test_instruction_counts_and_missing_ops():
    r = _check(_replace(GOOD, 4, None))  # one ds_read short
    assert "CONFORM.COUNT" in r.rules("warn")
    r = _check(_replace(_replace(GOOD, 4, None), 3, None))
    assert "CONFORM.OP_MISSING" in r.rules("warn")


def test_setprio_value():
    r = _check(_replace(GOOD, 8, (5, "s_setprio 2")))
    assert "CONFORM.SETPRIO_VALUE" in r.rules("warn")


def test_unattributed_summary_and_op_table():
    r = _check(GOOD)
    f = [f for f in r.findings if f.rule == "CONFORM.UNATTRIBUTED"][0]
    assert f.message.startswith("2 of 18 instructions")
    table = op_table(parse_isa(_isa(GOOD)), _schedule())
    assert table[0].split()[:3] == ["main", "0", "load"] and "lds_dma2" in table[0]


# ------------------------------------------------------------------------- diff
def test_diff_identical_and_changed_regions():
    a = parse_isa(_isa(GOOD, kernarg=608))
    d = diff_isa(a, parse_isa(_isa(GOOD, kernarg=212)))
    assert ("kernarg_size", "608", "212") in d.meta
    assert d.rows == [] and "3 identical" in d.findings[0]
    rows = list(GOOD)
    rows.insert(12, (6, "s_waitcnt lgkmcnt(1)"))
    d = diff_isa(a, parse_isa(_isa(rows)))
    assert len(d.rows) == 1 and "waitcnt 0->1" in d.rows[0]
    swapped = _replace(_replace(GOOD, 10, (6, GOOD[12][1])), 12, (7, GOOD[10][1]))
    d = diff_isa(a, parse_isa(_isa(swapped)))
    assert "order differs" in d.rows[0]
    assert "isa diff" in d.render()


# ------------------------------------------------------------------------- CLI
def test_cli_conform_and_against(tmp_path, capsys, monkeypatch):
    good = tmp_path / "good.s"
    good.write_text(_isa(GOOD))
    bad = tmp_path / "bad.s"
    bad.write_text(_isa(_replace(GOOD, 16, None)))
    monkeypatch.setitem(sys.modules, "_conform_sched_for_test", sys.modules[__name__])
    spec = "_conform_sched_for_test:_schedule"
    assert cli(["conform", str(good), "--schedule", spec, "--ops"]) == 0
    assert "main         0 load" in capsys.readouterr().out
    assert cli(["conform", str(bad), "--schedule", spec, "--against", str(good)]) == 1
    out = capsys.readouterr().out
    assert "CONFORM.BARRIER_COUNT" in out and "isa diff" in out
    assert cli(["conform", str(good)]) == 0  # no schedule: attribution summary only


# ------------------------------------------------------------------------- emit tags
class _Prim:
    def __init__(self, out):
        self.out = out

    def __getattr__(self, name):
        return lambda *a, **k: self.out.append((name,) + a)


def test_emit_tag_wraps_every_op_in_its_location():
    s = _schedule()
    out, locs = [], []

    class Loc:
        def __init__(self, filename, line):
            self.at = (filename, line)

        def __enter__(self):
            locs.append(self.at)
            out.append(("enter",) + self.at)

        def __exit__(self, *exc):
            out.append(("exit",) + self.at)

    bindings = {n: (lambda n: lambda *a, **k: out.append((n,)) or n)(n) for n in ("load", "read", "mm", "exp")}
    emit(s.section("main"), bindings, {}, primitives=_Prim(out), tag="t", location=Loc)
    assert locs == [("sched:t/main", i + 1) for i in range(len(s.section("main").ops))]
    assert out[:3] == [("enter", "sched:t/main", 1), ("load",), ("exit", "sched:t/main", 1)]
    assert tag_file("t") == "sched:t" and tag_file("t", "loop") == "sched:t/loop"


def test_emit_without_tag_never_builds_locations():
    s = _schedule()

    def boom(*a):
        raise AssertionError("location used without tag")

    bindings = {n: (lambda *a, **k: None) for n in ("load", "read", "mm", "exp")}
    emit(s.section("main").ops, bindings, {}, primitives=_Prim([]), location=boom)
    out = []
    emit(s.section("main").ops, bindings, {}, primitives=_Prim(out), tag="t", section="main", location=_null_loc(out))
    assert out[0] == ("loc", "sched:t/main", 1)


def _null_loc(out):
    from contextlib import contextmanager

    @contextmanager
    def loc(filename, line):
        out.append(("loc", filename, line))
        yield

    return loc
