# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target packs (FLYDSL_SCHED_TARGET_PATH overlay, provenance) and target-driven hazard checks.

Every pack here is a fake written to tmp_path; its numbers are made up for the test.
"""

import json
import os

import pytest

from flydsl.sched.ir import Builder, Op, Queue, Schedule, Section
from flydsl.sched.target import TARGET_PATH_ENV, UnknownFact, load_target
from flydsl.sched.verify import verify

pytestmark = pytest.mark.l0_backend_agnostic

MFMA = "mfma_f32_16x16x32_f16"  # described (passes) by the public gfx950.json


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(TARGET_PATH_ENV, raising=False)
    load_target.cache_clear()
    yield
    load_target.cache_clear()


def _pack(d, name, facts, arch="gfx950", **top):
    d.mkdir(parents=True, exist_ok=True)
    data = {"schema": "flydsl.sched.target/v1", "arch": arch, "facts": facts, **top}
    (d / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return d


def _use(monkeypatch, *dirs):
    monkeypatch.setenv(TARGET_PATH_ENV, os.pathsep.join(map(str, dirs)))


def _hz(**rules):
    return {"matrix_hazards": {"value": rules, "source": "fake:hazards", "confidence": "source-reported"}}


# ------------------------------------------------------------------ loader / provenance
def test_pack_overlays_per_key_and_records_provenance(tmp_path, monkeypatch):
    d = _pack(
        tmp_path / "p",
        "gfx950",
        {
            "latency_cycles": {"value": {"ds_read_b128": 1234}, "source": "fake:lat p1", "confidence": "measured"},
            "brand_new": {"value": 7, "source": "fake:new", "confidence": "inferred", "note": "n"},
        },
    )
    assert load_target("gfx950").fact("latency_cycles").value["ds_read_b128"] is None
    _use(monkeypatch, d)
    t = load_target("gfx950")
    lat = t.fact("latency_cycles").value
    assert lat["ds_read_b128"] == 1234 and lat["ds_read_tr16_b64"] is None  # merged per key
    p = t.provenance("latency_cycles", "ds_read_b128")
    assert p.origin.endswith("gfx950.json") and str(d) in p.origin
    assert (p.source, p.confidence) == ("fake:lat p1", "measured")
    assert t.provenance("latency_cycles", "s_barrier").origin == "builtin:gfx950.json"  # untouched key
    assert t.provenance("lds_banks").origin == "builtin:gfx950.json"
    assert t.get("brand_new") == 7 and t.counter_max("vm") == 63  # public facts survive
    measured = t.facts_with_confidence("measured")
    assert "latency_cycles.ds_read_b128" in measured and "lds_banks" not in measured
    assert len(t.layers) == 2


def test_later_dir_wins_and_sku_pack_applies(tmp_path, monkeypatch):
    a = _pack(tmp_path / "a", "gfx950", {"x": {"value": 1, "source": "fake:a", "confidence": "inferred"}})
    b = _pack(tmp_path / "b", "gfx950", {"x": {"value": 2, "source": "fake:b", "confidence": "inferred"}})
    _pack(b, "gfx950@mi355x", {"y": {"value": 3, "source": "fake:sku", "confidence": "inferred"}})
    _use(monkeypatch, a, b)
    t = load_target("gfx950")
    assert t.get("x") == 2 and t.get("y") == 3  # sku from the public description selects the sku pack
    other = load_target("gfx950@other")
    assert other.get("y") is None and other.key == "gfx950@other"


def test_cache_follows_the_env_var(tmp_path, monkeypatch):
    d = _pack(tmp_path / "p", "gfx950", {"x": {"value": 1, "source": "fake", "confidence": "inferred"}})
    assert load_target("gfx950").get("x") is None
    _use(monkeypatch, d)
    assert load_target("gfx950").get("x") == 1
    monkeypatch.delenv(TARGET_PATH_ENV)
    assert load_target("gfx950").get("x") is None


def test_pack_only_arch_and_exact_match(tmp_path, monkeypatch):
    wave = {"wave_size": {"value": 32, "source": "fake", "confidence": "inferred"}}
    d = _pack(tmp_path / "p", "gfx9999", wave, arch="gfx9999")
    with pytest.raises(UnknownFact):
        load_target("gfx9999")
    _use(monkeypatch, d)
    assert load_target("gfx9999").get("wave_size") == 32
    with pytest.raises(UnknownFact):
        load_target("gfx9998")  # no substitution of a neighbouring arch
    _pack(d, "gfx9998", {}, arch="gfx9999")
    with pytest.raises(ValueError):
        load_target("gfx9998")  # file name and content disagree


# ------------------------------------------------------------------ matrix hazards
def _mm_then_valu(nops=0):
    b = Builder()
    b.op("valu", "init", outs=("a", "b", "acc"), count=0)
    b.op("mfma", "mm", ins=("a", "b", "acc"), outs=("acc",), count=1, mfma=MFMA)
    if nops:
        b.nop(nops - 1)
    b.op("valu", "use", ins=("acc",), outs=("r",), count=1)
    return Schedule("mm", "gfx950", {}, {}, [Section("main", b.ops)])


def _hz_pack(tmp_path, monkeypatch, **entry):
    _use(monkeypatch, _pack(tmp_path / "p", "gfx950", _hz(matrix_to_valu=entry)))


def test_matrix_dependency_without_interlock_is_an_error(tmp_path, monkeypatch):
    _hz_pack(tmp_path, monkeypatch, wait=5, interlock=False)
    r = verify(_mm_then_valu())
    assert "HAZARD.MATRIX_DEP" in r.rules("error"), str(r)
    assert "fake:hazards" in r.errors[0].message
    assert verify(_mm_then_valu(nops=5)).ok  # s_nop 4 = 5 wait states


def test_undescribed_ops_only_bound_the_distance(tmp_path, monkeypatch):
    _hz_pack(tmp_path, monkeypatch, wait=5, interlock=False)
    s = _mm_then_valu()
    s.sections[0].ops.insert(2, Op("valu", "other", outs=("z",), count=0))  # size not described
    r = verify(s)
    assert r.ok and r.rules("warn") == ["HAZARD.MATRIX_DEP"], str(r)
    s.sections[0].ops[-1].count = 0  # the consumer itself is undescribed: no claim at all
    assert "HAZARD.MATRIX_DEP" not in verify(s).rules()


def test_matrix_dependency_by_passes_and_interlock(tmp_path, monkeypatch):
    _hz_pack(tmp_path, monkeypatch, wait=1, by_passes={"4": 9}, interlock=True)
    r = verify(_mm_then_valu(nops=5))  # MFMA has 4 passes -> 9 required, 5 given
    assert r.ok and "PERF.MATRIX_DEP_STALL" in r.rules("info")


def test_matrix_dependency_refuses_null_facts(tmp_path, monkeypatch):
    _hz_pack(tmp_path, monkeypatch, wait=None, interlock=False)
    r = verify(_mm_then_valu())
    assert r.ok and "TARGET.UNKNOWN" in r.rules("info") and "HAZARD.MATRIX_DEP" not in r.rules()


def test_targets_without_hazard_facts_skip_cleanly():
    r = verify(_mm_then_valu())
    assert r.ok and not [f for f in r.findings if f.rule.startswith(("HAZARD", "PERF.MATRIX", "TARGET"))]


def test_loop_body_wraps_into_next_iteration(tmp_path, monkeypatch):
    _use(
        monkeypatch,
        _pack(tmp_path / "p", "gfx950", _hz(matrix_to_matrix_srcc_exact={"wait": 3, "interlock": False})),
    )
    b = Builder()
    b.op("valu", "init", outs=("a", "b", "acc"))
    pro = list(b.ops)
    b = Builder()
    b.op("mfma", "mm", ins=("a", "b", "acc"), outs=("acc",), mfma=MFMA)
    b.op("salu", "s", count=1)
    loop = Section("loop", b.ops, loop=("i", lambda bases: range(2)), carried=("acc",))
    s = Schedule("w", "gfx950", {}, {}, [Section("pro", pro), loop])
    r = verify(s)
    assert r.rules("error") == ["HAZARD.MATRIX_DEP"] and "next iteration" in r.errors[0].where, str(r)


# ------------------------------------------------------------------ counters
COUNTERS = {
    "counters": {
        "value": {
            "load": {"max": 9, "in_order": True, "kinds": ["vmem_load"]},
            "ds": {"max": 9, "in_order": False, "kinds": ["ds_read", "ds_write"]},
        },
        "source": "fake:counters",
        "confidence": "source-reported",
    },
    "supported_ops": {"value": ["vmem_load", "ds_read", "wait", "valu"], "source": "fake", "confidence": "inferred"},
}


def test_counter_table_generalizes_queue_counters(tmp_path, monkeypatch):
    _use(monkeypatch, _pack(tmp_path / "p", "gfx9999", COUNTERS, arch="gfx9999"))
    ops = [
        Op("vmem_load", "ld", outs=("x",), queue="ld", count=1),
        Op("ds_read", "rd", outs=("y",), queue="lds", count=1),
        Op("wait", "wait ds 1", args={"ds": 1}),
        Op("wait", "wait ds 0", args={"ds": 0, "load": 0}),
    ]
    good = {"ld": Queue("ld", "load"), "lds": Queue("lds", "ds")}
    r = verify(Schedule("c", "gfx9999", {}, good, [Section("main", ops)]))
    # (the simulator drains only vm/lgkm waits, so it also reports these split counters as undrained)
    assert [x for x in r.rules("error") if x.startswith("HW.")] == ["HW.WAIT_PARTIAL_OOO"], str(r)

    bad = {"ld": Queue("ld", "ds"), "lds": Queue("lds", "vm")}  # vm does not exist here
    r = verify(Schedule("c", "gfx9999", {}, bad, [Section("main", ops[:2])]))
    assert {"HW.COUNTER_MISMATCH", "HW.COUNTER_UNKNOWN"} <= set(r.rules("error")), str(r)


# ------------------------------------------------------------------ LDS banks
def _ds(stride, access=4):
    op = Op("ds_read", "rd", outs=("y",), args={"lds_stride_bytes": stride, "lds_access_bytes": access})
    return Schedule("l", "gfx950", {}, {}, [Section("main", [op])])


def test_lds_bank_conflict_from_declared_stride(tmp_path, monkeypatch):
    r = verify(_ds(256))
    assert "TARGET.UNKNOWN" in r.rules("info")  # lanes per cycle is not in the public description
    lanes = {"lds_lanes_per_cycle": {"value": {"default": 16}, "source": "fake", "confidence": "inferred"}}
    _use(monkeypatch, _pack(tmp_path / "p", "gfx950", lanes))
    assert "PERF.LDS_BANK_CONFLICT" in verify(_ds(256)).rules("info")
    assert "PERF.LDS_BANK_CONFLICT" not in verify(_ds(4)).rules("info")
