# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Waits on gfx12-style split counters are simulated (fake target pack, no private data)."""

import json

import pytest

from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T
from flydsl.sched.target import load_target
from flydsl.sched.transform import synthesize_waits
from flydsl.sched.verify import verify

pytestmark = pytest.mark.l0_backend_agnostic


@pytest.fixture
def split_target(tmp_path, monkeypatch):
    pack = {
        "arch": "gfxtest",
        "facts": {
            "counters": {
                "value": {"tensor": {"max": 63, "in_order": True}, "ds": {"max": 63, "in_order": True}},
                "source": "test",
                "confidence": "test",
            },
            "supported_ops": {
                "value": ["lds_dma", "ds_read", "wait", "barrier", "valu"],
                "source": "test",
                "confidence": "test",
            },
        },
    }
    (tmp_path / "gfxtest.json").write_text(json.dumps(pack))
    monkeypatch.setenv("FLYDSL_SCHED_TARGET_PATH", str(tmp_path))
    load_target.cache_clear()
    yield "gfxtest"
    load_target.cache_clear()


def _sched(target, tensor_wait):
    b = Builder()
    b.op("lds_dma", "tdm", count=1, queue="tdm", writes=(SlotUse("x", 0, T("t")),))
    b.op("lds_dma", "tdm", count=1, queue="tdm", writes=(SlotUse("y", 0, T("t")),))
    b.wait(tensor=tensor_wait)
    b.barrier()
    b.op("ds_read", "read", outs=("v",), count=1, queue="ds", reads=(SlotUse("x", 0, T("t")),))
    b.wait(ds=0)
    b.op("valu", "use", ins=("v",), count=0)
    b.wait(tensor=0)
    return Schedule(
        "split",
        target,
        {"x": Buffer("x", "lds", "X", 1, 64), "y": Buffer("y", "lds", "Y", 1, 64)},
        {"tdm": Queue("tdm", "tensor"), "ds": Queue("ds", "ds")},
        [Section("p", b.ops)],
        bases=lambda n: {"t": 0},
    )


def test_split_counter_waits_are_simulated(split_target):
    assert verify(_sched(split_target, 1)).ok  # x's DMA (older) retired, y's may still fly
    assert "SYNC.RAW_LDS" in verify(_sched(split_target, 2)).rules("error")


def test_wait_synthesis_on_split_counters(split_target):
    w = synthesize_waits(_sched(split_target, 0))
    assert [o.args.get("tensor") for o in w.section("p").ops if o.kind == "wait"][0] == 1
