# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Deterministic pre-compile resource estimate: LDS, VGPR lower bound, occupancy."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir import Schedule
from .target import TargetDesc, load_target
from .verify import trace


@dataclass
class ResourceReport:
    schedule: str
    target: str
    lds_bytes: int
    lds_budget: int
    lds_buffers: Dict[str, int]
    vgpr_live_peak: int  # declared live values only (lower bound, per lane)
    vgpr_peak_where: str
    vgpr_budget: Optional[int]  # per wave at the schedule's waves_per_eu
    waves_per_eu: Optional[int]
    notes: List[str] = field(default_factory=list)
    # per-group programs: group -> live peak of that group's program (the report's peak is the max)
    vgpr_per_group: Dict[int, int] = field(default_factory=dict)

    @property
    def fits(self):
        ok = self.lds_bytes <= self.lds_budget
        if self.vgpr_budget is not None:
            ok = ok and self.vgpr_live_peak <= self.vgpr_budget
        return ok

    def __str__(self):
        lines = [f"resource {self.schedule} @{self.target}"]
        bufs = ", ".join(f"{k} {v} B" for k, v in self.lds_buffers.items())
        pct = 100.0 * self.lds_bytes / self.lds_budget
        lines.append(f"  LDS   {self.lds_bytes} / {self.lds_budget} B ({pct:.0f}%)  [{bufs}]")
        budget = (
            f" / {self.vgpr_budget} (arch+acc) per wave at waves_per_eu={self.waves_per_eu}" if self.vgpr_budget else ""
        )
        lines.append(f"  VGPR  >= {self.vgpr_live_peak} live dwords/lane{budget}  (peak at {self.vgpr_peak_where})")
        for n in self.notes:
            lines.append(f"  note  {n}")
        return "\n".join(lines)


def _vgpr_live_peak(sched: Schedule, n_tiles: int, group: int = 0):
    """Peak of the sum of declared value sizes over def..last-use intervals (in ``group``'s program)."""
    events = trace(sched, n_tiles, group)
    default = dict(sched.meta.get("value_dwords", {}))
    intervals = []  # (start, end, dwords, name) -- one per definition
    open_def: Dict[str, tuple] = {}  # name -> (start, dwords)
    last_use: Dict[str, int] = {}
    for i, ev in enumerate(events):
        for name in ev.op.ins:
            last_use[name] = i
        for name in ev.op.outs:
            if name in open_def:
                start, d = open_def[name]
                end = last_use.get(name, start)
                if name in ev.op.ins and end == i:
                    end = i - 1  # in-place update (x -> x) reuses the registers
                intervals.append((start, max(start, end), d, name))
            open_def[name] = (i, ev.op.out_dwords.get(name, default.get(name, 0)))
            last_use[name] = i
    for name, (start, d) in open_def.items():
        intervals.append((start, max(start, last_use.get(name, start)), d, name))
    peak, where = 0, ""
    for i, ev in enumerate(events):
        live = sum(d for s, e, d, _ in intervals if s <= i <= e)
        if live > peak:
            peak, where = live, ev.where
    unsized = sorted({n for _, _, d, n in intervals if d == 0})
    return peak, where, unsized


def estimate_resources(sched: Schedule, target: Optional[TargetDesc] = None, n_tiles: Optional[int] = None):
    target = target or load_target(sched.target)
    lds_buffers = {b.name: b.slots * b.bytes_per_slot for b in sched.buffers.values() if b.tier == "lds"}
    lds_bytes = sum(lds_buffers.values()) + int(sched.meta.get("lds_extra_bytes", 0))
    if sched.meta.get("lds_extra_bytes"):
        lds_buffers["other"] = int(sched.meta["lds_extra_bytes"])
    n = n_tiles or max(sched.meta.get("verify_tiles", (4,)))
    per_group: Dict[int, int] = {}
    if sched.group_sections is None:
        peak, where, unsized = _vgpr_live_peak(sched, n)
    else:  # registers are per wave: the budget must hold the hungriest group's program
        peak, where, unsized = 0, "", []
        for g in range(sched.groups):
            p, w, u = _vgpr_live_peak(sched, n, g)
            per_group[g] = p
            unsized = sorted(set(unsized) | set(u))
            if p > peak or not where:
                peak, where = p, f"g{g}:{w}"
    wpe = sched.meta.get("waves_per_eu")
    budget = None
    notes = []
    if wpe:
        per_lane = target.require("vgpr_file_per_lane") // wpe
        gran = target.require("vgpr_alloc_granule")
        budget = per_lane - per_lane % gran
        arch_max = target.require("arch_vgpr_max_per_wave")
        if peak > arch_max:
            notes.append(
                f"live values exceed the {arch_max} arch VGPRs a wave may hold; the rest must sit in AccVGPRs "
                "(MFMA accumulators) or spill"
            )
    notes.append("VGPR figure counts declared schedule values only; addresses, masks and temporaries come on top")
    if unsized:
        notes.append(f"values without a declared size: {', '.join(unsized)}")
    if per_group:
        notes.append("per-group live peaks: " + ", ".join(f"group {g} {p}" for g, p in sorted(per_group.items())))
    return ResourceReport(
        schedule=sched.name,
        target=target.key,
        lds_bytes=lds_bytes,
        lds_budget=target.require("lds_bytes_per_cu"),
        lds_buffers=lds_buffers,
        vgpr_live_peak=peak,
        vgpr_peak_where=where,
        vgpr_budget=budget,
        waves_per_eu=wpe,
        notes=notes,
        vgpr_per_group=per_group,
    )
