# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Issue programs: the exact instruction-class order of a fence region.

A plain ``sched_group`` interleave is a hint.  If it asks for more instructions
of a class than the region has, or the region is split by control flow,
LLVM fills what it can and leaves the rest to its own heuristics.  An
*issue program* is a checked statement about one region:

* **template**: a *spine* class (default MFMA) and *filler* classes, with
  ``gap=(lo, hi)`` filler instructions between consecutive spine instructions.
  The fillers are the region's own instructions, taken in op order.
* **explicit**: an ordered list of ``(class, n)`` slots.

``resolve(schedule, calibration)`` turns each ``issue`` op into a chain of
``sched_group`` ops (``sched_group_barrier(mask, n, group)``), one group per
slot.  The chain is built from *calibrated* per-op instruction counts, so every
group can be filled exactly and every grouped instruction has exactly one
place.  Checks: the counts are known and agree with the program, the gaps are
feasible, data dependencies between spine and fillers allow the order, and the
region contains no control flow.  ``resolve`` reports problems as
``verify.Finding``s.

Why sched_group chains rather than emitting one instruction at a time between
``sched_barrier`` fences: the chain leaves register allocation and operand
choice to LLVM, needs no change to the kernel's helpers, and is checked
against the real ISA by ``conform`` (``CONFORM.ISSUE``).  An instruction class
the chain cannot name (inline asm, SALU, s_nop hazard padding) stays free.  It
is reported, not hidden.

Calibration (``calibrate``) maps every binding to the instructions it produced
in a compiled kernel, read back through ``emit(tag=...)`` locations:
``python -m flydsl.sched calibrate kernel_final_isa.s --schedule m:builder``.
"""

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import Op, Schedule, Section
from .verify import Finding, Report

ISSUE_KIND = "issue"
# instruction classes a sched_group_barrier mask can select (conform's class names)
GROUPABLE = ("mfma", "valu", "trans", "salu", "ds_read", "ds_write", "lds_dma", "vmem_load", "vmem_store")
# conform class -> target sched_group_mask name, where they differ
MASK_NAME = {"lds_dma": "ldsdma", "vmem_load": "vmem_read", "vmem_store": "vmem_write"}


def fence_regions(ops: Sequence[Op]) -> List[int]:
    """Region index of every op (same rule as ``conform``): code may not move between regions."""
    out, r = [], 0
    for op in ops:
        fence = None
        if op.kind == "sched_barrier" and op.args.get("mask", 0) == 0:
            fence = "both"
        elif op.kind == "barrier":
            fence = op.args.get("fence", "both")
        elif op.kind == "stagger":
            fence = "both"
        if fence in ("both", "before"):
            r += 1
        out.append(r)
        if fence in ("both", "after"):
            r += 1
    return out


# --------------------------------------------------------------------------- calibration


@dataclass
class Calibration:
    """Instructions per op, by class, measured from a compiled kernel.

    ``by_op["<section>#<n>"]`` (n-th op with a binding) is exact for a section whose
    binding sequence equals ``layout[section]`` (the compiled one); ``by_binding`` is
    the most common count of a binding over its occurrences, the fallback for other
    schedules.  Counts are scheduler *units*: an unpacked ``v_pk_*_f32`` pair counts
    once (``conform.mark_unpacked``).  Classes are conform's (``mfma``, ``valu``,
    ``trans``, ...); inline-asm instructions are counted as ``asm:<class>`` because no
    sched_group mask selects them.
    """

    by_binding: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_op: Dict[str, Dict[str, int]] = field(default_factory=dict)
    varies: Dict[str, List[Dict[str, int]]] = field(default_factory=dict)  # binding -> differing counts
    source: str = ""
    schedule: str = ""  # name of the calibrated schedule (informational)
    layout: Dict[str, List[str]] = field(default_factory=dict)  # section -> bindings of its code ops, in order

    def counts(self, op: Op, where: Optional[str] = None, exact: bool = False) -> Optional[Dict[str, int]]:
        """Counts of ``op``: exact (``by_op``) when its section matches ``layout``, else per binding."""
        if exact and where is not None and where in self.by_op:
            return self.by_op[where]
        if op.binding is not None and op.binding in self.by_binding:
            return self.by_binding[op.binding]
        return None

    def matches(self, sec: Section) -> bool:
        return self.layout.get(sec.name) == [o.binding for o in sec.ops if o.binding is not None]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "schedule": self.schedule,
            "layout": self.layout,
            "by_binding": self.by_binding,
            "by_op": self.by_op,
            "varies": self.varies,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            d.get("by_binding", {}),
            d.get("by_op", {}),
            d.get("varies", {}),
            d.get("source", ""),
            d.get("schedule", ""),
            d.get("layout", {}),
        )


def code_keys(sec: Section) -> Dict[int, str]:
    """op index -> ``"<section>#<n>"``, n counting only ops with a binding (ops that emit code).

    Stable under ``resolve``, which only replaces code-less issue / sched_group ops.
    """
    out, n = {}, 0
    for k, op in enumerate(sec.ops):
        if op.binding is not None:
            out[k] = f"{sec.name}#{n}"
            n += 1
    return out


def calibrate(isa, schedule: Schedule, tag: Optional[str] = None, source: str = "") -> Calibration:
    """Per-op instruction counts of ``schedule`` from a tagged ISA (``conform.parse_isa`` result)."""
    if not isa.tagged:
        raise ValueError(
            "ISA carries no sched: locations; compile with emit(tag=...) and FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1"
        )
    tag = tag or isa.tags[0]
    secs = {s.name: s for s in schedule.all_sections()}
    keys = {name: code_keys(s) for name, s in secs.items()}
    by_op: Dict[str, Counter] = {}
    for i in isa.insts:
        if i.tag != tag or i.section not in secs or i.op not in keys[i.section]:
            continue
        key = keys[i.section][i.op]
        # units, not raw instructions: an unpacked v_pk_*_f32 is one instruction to the scheduler;
        # the raw count is kept as "raw:<class>" where it differs
        c = by_op.setdefault(key, Counter())
        cls = ("asm:" if i.asm else "") + i.cls
        c[cls] += i.unit
        c["raw:" + cls] += 1
    for c in by_op.values():
        for k in [k for k in c if k.startswith("raw:")]:
            if c[k] == c[k[4:]]:
                del c[k]
    per_binding: Dict[str, List[Dict[str, int]]] = {}
    for sec in secs.values():
        for k, op in enumerate(sec.ops):
            if op.binding is None or op.kind in ("stagger",):
                continue
            per_binding.setdefault(op.binding, []).append(dict(by_op.get(keys[sec.name][k], {})))
    by_binding, varies = {}, {}
    for b, seen in per_binding.items():
        keyed = Counter(tuple(sorted(c.items())) for c in seen)
        by_binding[b] = dict(keyed.most_common(1)[0][0])
        if len(keyed) > 1:
            varies[b] = [dict(k) for k in keyed]
    layout = {name: [o.binding for o in s.ops if o.binding is not None] for name, s in secs.items()}
    return Calibration(by_binding, {k: dict(v) for k, v in by_op.items()}, varies, source, schedule.name, layout)


# --------------------------------------------------------------------------- planning


@dataclass
class Plan:
    slots: List[Tuple[str, int]]
    spine: int
    fillers: int
    gaps: List[int]  # fillers after each spine instruction (last = tail)
    lead: int
    free: Dict[str, int]  # instructions in the region no group names
    where: str = ""
    raw_gaps: List[int] = field(default_factory=list)  # expected raw instructions per gap (unpacking)

    def pattern(self) -> str:
        from .conform import SHORT

        return " ".join(f"{SHORT.get(c, c)}{n}" for c, n in self.slots)


def _closure(region: Sequence[Tuple[int, Op]]) -> Dict[int, set]:
    """op index -> indices of earlier region ops it depends on, transitively (through ins/outs)."""
    deps: Dict[int, set] = {}
    last_writer: Dict[str, int] = {}
    for k, op in region:
        d = set()
        for v in op.ins:
            w = last_writer.get(v)
            if w is not None:
                d |= {w} | deps[w]
        deps[k] = d
        for v in op.outs:
            last_writer[v] = k
    return deps


def plan_region(
    region: Sequence[Tuple[int, Op]],
    prog: Dict,
    counts: Dict[int, Dict[str, int]],
    report: Report,
    where: str,
) -> Optional[Plan]:
    """Slots for one ``issue`` op over the ``(index, op)`` pairs of its region (ops before it)."""

    def err(rule, msg, hint="", sev="error"):
        report.add((rule, where, msg), Finding(rule, sev, msg, where=where, hint=hint))

    ok = True
    for k, op in region:
        c = counts.get(k)
        if c is None:
            if op.binding is not None:
                err("ISSUE.UNCALIBRATED", f"{where}: no calibrated instruction counts for op {k} {op}",
                    hint="python -m flydsl.sched calibrate <final_isa.s> --schedule <module:builder>")  # fmt: skip
                ok = False
            continue
        if c.get("branch", 0):
            err("ISSUE.CONTROL_FLOW", f"{where}: op {k} {op} contains control flow ({c['branch']} branch)",
                hint="sched_group_barrier acts inside one basic block; make the op branchless or move it out")  # fmt: skip
            ok = False
    if not ok:
        return None
    total = Counter()
    for k, _ in region:
        total.update({c: n for c, n in counts.get(k, {}).items() if not c.startswith("raw:")})
    spine = prog["spine"]
    fill = tuple(prog["fill"])
    free = {c: n for c, n in total.items() if c != spine and c not in fill and n}
    asm = {c: n for c, n in free.items() if c.startswith("asm:") and c[4:] in (spine,) + fill}
    if asm:
        err("ISSUE.ASM", f"{where}: {sum(asm.values())} inline-asm instruction(s) {asm} cannot be grouped and will "
            "float", sev="warn", hint="use a non-asm form of the instruction so a sched_group mask can select it")  # fmt: skip

    if prog["mode"] == "explicit":
        slots = [tuple(s) for s in prog["slots"]]
        want = Counter()
        for c, n in slots:
            want[c] += n
        for c in set(want) | {spine} | set(fill):
            if want[c] != total.get(c, 0):
                err("ISSUE.COUNT", f"{where}: program has {want[c]} {c} slot(s), the region issues {total.get(c, 0)}")
                ok = False
        gaps, lead, cur, seen = [], 0, 0, False
        for c, n in slots:
            if c == spine:
                for _ in range(n):
                    if seen:
                        gaps.append(cur)
                    else:
                        lead = cur
                    seen, cur = True, 0
            else:
                cur += n
        gaps.append(cur)
        plan = Plan(slots, total.get(spine, 0), sum(want[c] for c in fill), gaps, lead, free, where, list(gaps))
        _check_gaps(plan, prog, err)
        return plan if ok else None

    # template
    S = total.get(spine, 0)
    if S == 0:
        err("ISSUE.NO_SPINE", f"{where}: the region issues no {spine}")
        return None
    deps = _closure(region)
    stream: List[Tuple[str, int]] = []  # (class, op index) in op order
    weight: List[float] = []  # raw instructions per unit (unpacked pairs weigh 2)
    for k, _ in region:
        c = counts.get(k, {})
        for cls in fill:
            n = c.get(cls, 0)
            stream += [(cls, k)] * n
            weight += [c.get("raw:" + cls, n) / n if n else 1.0] * n
    F = len(stream)
    lead = min(prog.get("lead", 0), F)
    # split the stream into S gaps of (about) equal *raw* size, what the hardware issues; when
    # that exceeds the gap bound, cap the gaps between spine instructions and let the tail
    # (after the last spine instruction) take the rest
    lo_gap, hi_gap = prog.get("gap", (0, 1 << 30))

    def split(per):
        gaps, raw_gaps, acc = [0] * S, [0.0] * S, 0.0
        for p in range(lead, F):
            g = min(S - 1, int((acc + weight[p] / 2) / per)) if per else 0
            gaps[g] += 1
            raw_gaps[g] += weight[p]
            acc += weight[p]
        return gaps, raw_gaps

    per = sum(weight[lead:]) / S
    gaps, raw_gaps = split(per)
    while per > lo_gap and S > 1 and max(raw_gaps[:-1]) > hi_gap + 0.5:
        per -= 0.25
        gaps, raw_gaps = split(per)
    # spine instruction j belongs to op spine_op[j]
    spine_op = [k for k, _ in region for _ in range(counts.get(k, {}).get(spine, 0))]
    # filler position -> number of spine instructions issued before it
    before: List[int] = []
    for j, g in enumerate([lead] + gaps):
        before += [j] * g
    # dependencies: a filler after the spine op it reads, a spine op after the fillers it reads
    for p, (cls, k) in enumerate(stream):
        need = max((j + 1 for j, s in enumerate(spine_op) if s in deps[k]), default=0)
        if before[p] < need:
            err("ISSUE.DEPENDS", f"{where}: {cls} of op {k} {region_op(region, k)} would issue after {before[p]} "
                f"{spine} but reads the result of {spine} #{need}", hint="move the op after the spine op it reads, or "
                "give the program a larger lead/gap")  # fmt: skip
            ok = False
            break
    first_spine = {}
    for j, s in enumerate(spine_op):
        first_spine.setdefault(s, j)
    for s, j in first_spine.items():
        for p, (cls, k) in enumerate(stream):
            if k in deps[s] and before[p] > j:
                err("ISSUE.DEPENDS", f"{where}: {spine} op {s} {region_op(region, s)} reads op {k} "
                    f"{region_op(region, k)}, whose {cls} would issue after {spine} #{before[p]}",
                    hint="put the producer earlier in the region or give the program a larger lead")  # fmt: skip
                ok = False
                break
    slots: List[Tuple[str, int]] = []

    def run(tokens):
        for cls, _ in tokens:
            if slots and slots[-1][0] == cls and slots[-1][0] != spine:
                slots[-1] = (cls, slots[-1][1] + 1)
            else:
                slots.append((cls, 1))

    run(stream[:lead])
    pos = lead
    for g in gaps:
        slots.append((spine, 1))
        run(stream[pos : pos + g])
        pos += g
    plan = Plan(slots, S, F, gaps, lead, free, where, [round(x) for x in raw_gaps])
    _check_gaps(plan, prog, err)
    return plan if ok else None


def region_op(region, k):
    for i, op in region:
        if i == k:
            return op
    return None


def _check_gaps(plan: Plan, prog: Dict, err):
    """Gap bounds apply to raw instructions (what is issued), estimated from calibrated unpacking."""
    lo, hi = prog.get("gap", (0, 1 << 30))
    raw = plan.raw_gaps or plan.gaps
    inner = raw[:-1]
    bad = [g for g in inner if not lo <= g <= hi]
    if bad:
        err("ISSUE.GAP", f"{plan.where}: {plan.fillers} filler units over {plan.spine} {prog['spine']} give about "
            f"{min(inner)}..{max(inner)} instructions per gap, outside {lo}..{hi}",
            hint="change the gap bounds, move work into or out of the region, or use an explicit program")  # fmt: skip
    if raw and raw[-1] > hi:
        err("ISSUE.GAP", f"{plan.where}: about {raw[-1]} fillers after the last {prog['spine']} (tail) exceed {hi}",
            sev="warn")  # fmt: skip


# --------------------------------------------------------------------------- resolve


def resolve(sched: Schedule, calibration: Calibration) -> Tuple[Schedule, Report]:
    """Replace every ``issue`` op with its ``sched_group`` chain; returns (new schedule, findings).

    The chain's ops carry ``args["issue"]`` (the program, its slots and an id) so
    ``conform`` can report achieved against requested order.
    """
    report = Report(sched.name, sched.target, [])
    out = copy.copy(sched)
    mapping = {}
    n_issue = 0
    for sec in sched.all_sections():
        if id(sec) in mapping:
            continue
        regions = fence_regions(sec.ops)
        keys = code_keys(sec)
        exact = calibration.matches(sec)
        new_ops: List[Op] = []
        for k, op in enumerate(sec.ops):
            if op.kind != ISSUE_KIND:
                new_ops.append(op)
                continue
            where = f"{sec.name}/{k}"
            region = [(i, o) for i, o in enumerate(sec.ops[:k]) if regions[i] == regions[k] and o.kind != ISSUE_KIND]
            counts = {}
            for i, o in region:
                c = calibration.counts(o, keys.get(i), exact)
                if c is not None:
                    counts[i] = c
            plan = plan_region(region, op.args, counts, report, where)
            if plan is None:
                new_ops.append(op)  # left unresolved: lower.emit refuses it
                continue
            n_issue += 1
            info = dict(op.args, slots=plan.slots, id=f"{sec.name}/{k}", gaps=plan.gaps, free=plan.free)
            for cls, n in plan.slots:
                new_ops.append(
                    Op(
                        kind="sched_group",
                        label=f"issue[{op.args['group']}] {n}x{cls}",
                        args={"cls": MASK_NAME.get(cls, cls), "size": n, "group": op.args["group"], "issue": info},
                    )
                )
            report.add(
                ("ISSUE.PLAN", where),
                Finding("ISSUE.PLAN", "info", f"{where}: {plan.pattern()} (units; about "
                        f"{min(plan.raw_gaps[:-1] or [0])}..{max(plan.raw_gaps[:-1] or [0])} instructions per gap, "
                        f"tail {plan.raw_gaps[-1]}, free {plan.free or {}})", where=where),
            )  # fmt: skip
        mapping[id(sec)] = Section(sec.name, new_ops, sec.loop, sec.carried)
    out.sections = [mapping[id(s)] for s in sched.sections]
    if getattr(sched, "group_sections", None) is not None:
        out.group_sections = [[mapping[id(s)] for s in secs] for secs in sched.group_sections]
    return out, report


def issue_args(
    mode: str,
    spine: str = "mfma",
    fill: Sequence[str] = ("trans", "valu"),
    gap: Tuple[int, int] = (3, 6),
    group: int = 0,
    slots: Optional[Sequence[Tuple[str, int]]] = None,
    lead: int = 0,
) -> Dict:
    if mode not in ("template", "explicit"):
        raise ValueError(f"issue mode must be template|explicit, got {mode!r}")
    if mode == "explicit" and not slots:
        raise ValueError("explicit issue programs need slots")
    for c in (spine,) + tuple(fill) + tuple(s[0] for s in slots or ()):
        if c not in GROUPABLE:
            raise ValueError(f"class {c!r} cannot be selected by a sched_group mask; choose from {GROUPABLE}")
    return {
        "mode": mode,
        "spine": spine,
        "fill": tuple(fill),
        "gap": tuple(gap),
        "group": group,
        "slots": [tuple(s) for s in slots] if slots else None,
        "lead": lead,
    }
