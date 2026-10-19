# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Pre-compile verifier: simulate every wave group over concrete tile counts.

Model (documented so findings can be argued with):

* Each wave group runs the same op trace, or -- with ``Schedule.group_sections``
  -- its own program.  Group ``g`` counts the ``s_barrier`` instructions it has
  executed; ``bidx_g(i)`` is that count just before position ``i`` of its trace.
  With ``stagger`` enabled, group 1 runs one extra barrier at ``stagger open``
  and group 0 one extra at ``stagger close`` (the dual-wave phase shift);
  otherwise both are plain barriers.  Barrier number ``k`` matches across
  groups, so groups that execute different barrier totals deadlock
  (``SYNC.BARRIER_MISMATCH``).
* Position ``a`` happens-before position ``b`` *for every wave of every
  group involved* iff ``max_g bidx_g(a) < min_g bidx_g(b)`` (over the groups
  executing ``a`` resp. ``b``): some barrier they all cross lies after ``a``
  and before ``b``.  Per-group programs are merged by barrier phase; counters,
  registers and tile coverage are per group, LDS slots are shared and hold a
  tile only once every group that writes the slot has written its part.
* Async ops enter a FIFO per hardware counter (vm / lgkm), one entry per op
  carrying its instruction count, and complete at the ``wait`` that drains
  them (``wait n`` leaves at most ``n`` instructions outstanding).  Entries
  on a compiler-tracked queue are also drained by the implicit wait the
  compiler inserts before their first consumer; untracked entries are not.
* An LDS slot written by an async op may be read at ``r`` only when the
  write's completion happens-before ``r`` (RAW); it may be overwritten at
  ``w`` only when every earlier read's completion happens-before ``w``
  (WAR).  A read that declares a tile must find that tile in the slot.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .ir import Op, Schedule
from .target import TargetDesc, UnknownFact, load_target


@dataclass
class Finding:
    rule: str
    severity: str  # "error" | "warn" | "info"
    message: str
    where: str = ""
    hint: str = ""
    n_tiles: Optional[int] = None  # first tile count that showed it
    occurrences: int = 1


@dataclass
class Report:
    schedule: str
    target: str
    tile_counts: List[int]
    findings: List[Finding] = field(default_factory=list)
    _index: Dict[tuple, Finding] = field(default_factory=dict, repr=False)

    def add(self, key: tuple, finding: Finding):
        """Record ``finding`` once per ``key``; repeats only bump ``occurrences``."""
        if key in self._index:
            self._index[key].occurrences += 1
            return
        self._index[key] = finding
        self.findings.append(finding)

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self):
        return not self.errors

    def rules(self, severity: Optional[str] = None):
        return sorted({f.rule for f in self.findings if severity is None or f.severity == severity})

    def to_json(self):
        d = asdict(self)
        d.pop("_index")
        return json.dumps(d, indent=2)

    def __str__(self):
        return self.render(verbose=False)

    def render(self, verbose: bool = True) -> str:
        n_info = sum(f.severity == "info" for f in self.findings)
        head = f"verify {self.schedule} @{self.target} tiles={self.tile_counts}: "
        head += "OK" if self.ok else f"{len(self.errors)} error(s)"
        if n_info and not verbose:
            head += f"  ({n_info} info finding(s) hidden)"
        lines = [head]
        order = {"error": 0, "warn": 1, "info": 2}
        for f in sorted(self.findings, key=lambda f: order[f.severity]):
            if f.severity == "info" and not verbose:
                continue
            loc = f" [{f.where}]" if f.where else ""
            ctx = [f"n_tiles={f.n_tiles}"] if f.n_tiles is not None else []
            if f.occurrences > 1:
                ctx.append(f"x{f.occurrences}")
            nt = f" ({', '.join(ctx)})" if ctx else ""
            lines.append(f"  {f.severity:5s} {f.rule}{loc}{nt}: {f.message}")
            if f.hint:
                lines.append(f"        hint: {f.hint}")
        return "\n".join(lines)


@dataclass
class Event:
    op: Op
    bases: Dict[str, int]
    where: str


def trace(sched: Schedule, n_tiles: int, group: int = 0) -> List[Event]:
    """Flatten wave group ``group``'s program into the concrete op sequence for ``n_tiles`` tiles."""
    bases = dict(sched.bases(n_tiles))
    events: List[Event] = []
    for sec in sched.sections_of(group, n_tiles):
        if sec.loop is None:
            events.extend(Event(op, bases, f"{sec.name}/{op}") for op in sec.ops)
            continue
        var, rng = sec.loop
        for it, v in enumerate(rng(bases)):
            b = dict(bases, **{var: v})
            events.extend(Event(op, b, f"{sec.name}#{it}[{var}={v}]/{op}") for op in sec.ops)
    return events


def _barrier_row(sched: Schedule, events: Sequence[Event], g: int) -> List[int]:
    """``row[i]`` = barriers group ``g`` executed before position ``i`` of ``events`` (``row[-1]``: total)."""
    cnt, row = 0, []
    for ev in events:
        row.append(cnt)
        op = ev.op
        if op.kind == "barrier":
            cnt += 1
        elif op.kind == "stagger":
            phase = op.args["phase"]
            if not sched.stagger or (phase == "open" and g == 1) or (phase == "close" and g == 0):
                cnt += 1
    row.append(cnt)
    return row


def barrier_counts(sched: Schedule, events: Sequence[Event]) -> List[List[int]]:
    """``out[g][i]`` = barriers group ``g`` executed before position ``i`` of a program every group runs."""
    return [_barrier_row(sched, events, g) for g in range(sched.groups)]


@dataclass
class _Item:
    """One op of one *lane*: a program executed by the groups ``lanes[lane]``."""

    ev: Event
    lane: int
    pos: int
    phase: Dict[int, int]  # group -> barriers it executed before this op


class _Sim:
    """Simulate the programs of all wave groups together.

    A *lane* is one program and the groups that run it: one lane executed by every group
    when they share the program, one lane per group with per-group programs.  Counters,
    registers and coverage are per lane (per wave); LDS slots are shared, and a slot is
    complete only when every lane that writes it has written its part.  Lanes are merged
    by barrier phase, so ops of different groups between the same two barriers are
    concurrent: only a barrier both cross orders them.
    """

    def __init__(self, sched: Schedule, target: TargetDesc, n_tiles: int, report: Report):
        self.s, self.t, self.n, self.r = sched, target, n_tiles, report
        self.per_group = sched.group_sections is not None
        if self.per_group:
            self.lanes = [(g,) for g in range(sched.groups)]
            traces = [trace(sched, n_tiles, g) for g in range(sched.groups)]
            rows = [{g: _barrier_row(sched, traces[g], g)} for g in range(sched.groups)]
        else:
            self.lanes = [tuple(range(sched.groups))]
            traces = [trace(sched, n_tiles)]
            rows = [dict(enumerate(barrier_counts(sched, traces[0])))]
        self.ev = traces[0]
        self.totals = {g: row[-1] for lane_rows in rows for g, row in lane_rows.items()}
        items = [
            _Item(ev, li, i, {g: row[i] for g, row in rows[li].items()})
            for li, evs in enumerate(traces)
            for i, ev in enumerate(evs)
        ]
        self.items = sorted(items, key=lambda it: (min(it.phase.values()), it.lane, it.pos))
        self.writers: Dict[tuple, set] = {}  # (buf, slot) -> lanes whose program writes it
        for li, evs in enumerate(traces):
            for ev in evs:
                for use in ev.op.writes:
                    self.writers.setdefault((use.buffer, use.slot), set()).add(li)
        self.cmax = {k: v["max"] for k, v in target.require("counters").items()}
        nl = len(self.lanes)
        self.fifo: List[Dict[str, List[dict]]] = [{k: [] for k in self.cmax} for _ in range(nl)]
        self.value_entry: List[Dict[str, dict]] = [{} for _ in range(nl)]  # SSA value -> async entry
        self.slot_write: Dict[tuple, Dict[int, dict]] = {}  # (buf, slot) -> lane -> {"entry", "tile"}
        # (buf, slot) -> writer lane -> read entries since that lane's last write
        self.slot_reads: Dict[tuple, Dict[int, List[dict]]] = {}
        self.consumed: List[Dict[str, Dict[int, int]]] = [{} for _ in range(nl)]
        self.lane = 0

    def find(self, rule, sev, msg, op=None, where="", hint=""):
        # One finding per (rule, op): the same op misbehaving in every loop iteration
        # and every tile count is one problem, reported with an occurrence count.
        # Info findings are keyed by message so generated per-tile copies collapse too.
        if self.per_group:
            where = f"g{self.lane}:{where}"
        if sev == "info":
            key = (rule, msg)
        else:
            key = (rule, id(op)) if op is not None else (rule, where)
            if self.per_group:
                key += (self.lane,)
        self.r.add(key, Finding(rule, sev, msg, where, hint, self.n))

    @staticmethod
    def hb(done: Optional[Dict[int, int]], it: _Item) -> bool:
        """Completion ``done`` (a phase) happens-before ``it`` for every wave of every group involved."""
        if done is None:
            return False
        return max(done.values()) < min(it.phase.values())

    def drain(self, counter: str, keep: int, phase: Dict[int, int]):
        """``s_waitcnt <counter>(keep)``: the oldest instructions retire until at most ``keep``
        remain.  An entry is complete only when *all* of its instructions retired."""
        q = self.fifo[self.lane][counter]
        total = sum(e["count"] for e in q)
        while q and total - q[0]["count"] >= keep:
            total -= q[0]["count"]
            q.pop(0)["done"] = phase

    def _barrier_mismatch(self) -> bool:
        if len(set(self.totals.values())) <= 1:
            return False
        per = ", ".join(f"group {g}: {c}" for g, c in sorted(self.totals.items()))
        self.r.add(
            ("SYNC.BARRIER_MISMATCH", tuple(sorted(self.totals.items()))),
            Finding(
                "SYNC.BARRIER_MISMATCH",
                "error",
                f"wave groups execute different numbers of s_barrier ({per}); "
                "the group with more barriers waits forever (deadlock)",
                "barriers",
                "every group must execute the same number of workgroup barriers for every tile count",
                self.n,
            ),
        )
        return True

    def run(self):
        if self._barrier_mismatch():
            return
        for it in self.items:
            self.lane = it.lane
            ev, op = it.ev, it.ev.op
            if not self.t.supports(op.kind):
                self.find("HW.UNSUPPORTED", "error", f"op kind {op.kind!r} has no {self.t.key} lowering", op, ev.where)
            if op.kind == "wait" or (op.kind == "barrier" and any(op.args.get(c) is not None for c in self.cmax)):
                self._wait(it)
            if op.kind == "wait":
                continue
            if op.kind == "setprio" and not 0 <= op.args["prio"] <= 3:
                self.find(
                    "HW.SETPRIO_RANGE",
                    "info",
                    f"s_setprio {op.args['prio']} keeps only SIMM16[1:0] -> effective priority {op.args['prio'] & 3}",
                    op,
                    ev.where,
                    "write the effective priority (0..3) so the schedule states what the hardware does",
                )
            entry = self._entry(it)
            self._inputs(it)
            self._reads(it, entry)
            self._writes(it, entry)
            consumed = self.consumed[it.lane]
            for tensor, ref in op.consumes:
                t = ref.eval(ev.bases)
                consumed.setdefault(tensor, {}).setdefault(t, 0)
                consumed[tensor][t] += 1
            self._issue(it, entry)
        self._finish()

    def _wait(self, it: _Item):
        # Every counter the target declares (vm/lgkm on CDNA; load/ds/tensor/... split counters on gfx12).
        ev = it.ev
        for counter in self.cmax:
            n = ev.op.args.get(counter)
            if n is None:
                continue
            if n > self.cmax[counter]:
                self.find(
                    "HW.WAITCNT_RANGE",
                    "error",
                    f"{counter}cnt({n}) exceeds the {self.cmax[counter]} encodable on {self.t.key}",
                    ev.op,
                    ev.where,
                )
            self.drain(counter, n, it.phase)

    def _entry(self, it: _Item) -> dict:
        """Completion record for ``op``: async ops complete at a later wait, others at issue."""
        op = it.ev.op
        if op.queue is None:
            return {"label": str(op), "done": it.phase, "count": op.count, "queue": None}
        if op.queue not in self.s.queues:
            raise KeyError(f"{self.s.name}: op {op} uses undeclared queue {op.queue!r}")
        return {"label": str(op), "done": None, "count": op.count, "queue": op.queue}

    def _inputs(self, it: _Item):
        ev = it.ev
        for name in ev.op.ins:
            e = self.value_entry[it.lane].get(name)
            if e is None or e["done"] is not None:
                continue
            q = self.s.queues[e["queue"]]
            if q.tracked:  # the compiler inserts the minimal in-order wait for this value
                fifo = self.fifo[it.lane][q.counter]
                after = fifo[fifo.index(e) + 1 :]
                self.drain(q.counter, sum(x["count"] for x in after), it.phase)
            else:
                self.find(
                    "SYNC.REG_NOT_READY",
                    "error",
                    f"{ev.op} uses {name!r} while its {q.counter} load ({e['label']}) is still outstanding; "
                    f"the compiler does not wait for queue {q.name!r}",
                    ev.op,
                    ev.where,
                    f"insert a {q.counter}cnt wait that drains {e['label']} before this op",
                )

    def _reads(self, it: _Item, entry):
        ev = it.ev
        for use in ev.op.reads:
            key = (use.buffer, use.slot)
            writes = self.slot_write.get(key, {})
            if not writes:
                self.find("SYNC.UNINIT_READ", "error", f"{ev.op} reads {use} before any write", ev.op, ev.where)
            for wl in sorted(self.writers.get(key, ())) if writes else ():
                w = writes.get(wl)
                if w is None:
                    self.find(
                        "SYNC.UNINIT_READ",
                        "error",
                        f"{ev.op} reads {use} before group(s) {list(self.lanes[wl])} wrote their part of it",
                        ev.op,
                        ev.where,
                    )
                    continue
                if not self.hb(w["entry"]["done"], it):
                    state = (
                        "still in flight"
                        if w["entry"]["done"] is None
                        else "complete but not yet visible to every wave"
                    )
                    by = f" (group(s) {list(self.lanes[wl])})" if self.per_group else ""
                    self.find(
                        "SYNC.RAW_LDS",
                        "error",
                        f"{ev.op} reads {use} while the write by {w['entry']['label']}{by} is {state}",
                        ev.op,
                        ev.where,
                        "drain the writer's counter, then cross a barrier every group executes, before this read",
                    )
                if use.tile is not None and w["tile"] is not None and use.tile.eval(ev.bases) != w["tile"]:
                    self.find(
                        "DATA.STALE_TILE",
                        "error",
                        f"{ev.op} expects {self.s.buffers[use.buffer].tensor} tile {use.tile.eval(ev.bases)} "
                        f"in {key[0]}[{key[1]}] but it holds tile {w['tile']} (written by {w['entry']['label']})",
                        ev.op,
                        ev.where,
                    )
            reads = self.slot_reads.setdefault(key, {})
            for wl in self.writers.get(key, ()):
                reads.setdefault(wl, []).append(entry)

    def _writes(self, it: _Item, entry):
        ev = it.ev
        for use in ev.op.writes:
            key = (use.buffer, use.slot)
            for rd in self.slot_reads.get(key, {}).get(it.lane, []):
                if rd is not entry and not self.hb(rd["done"], it):
                    state = "still in flight" if rd["done"] is None else "not yet complete in every wave"
                    self.find(
                        "SYNC.WAR_LDS",
                        "error",
                        f"{ev.op} overwrites {use} while the read by {rd['label']} is {state}",
                        ev.op,
                        ev.where,
                        "issue the overwrite after a barrier that follows the readers' lgkmcnt drain",
                    )
            self.slot_reads.setdefault(key, {})[it.lane] = []
            tile = use.tile.eval(ev.bases) if use.tile is not None else None
            self.slot_write.setdefault(key, {})[it.lane] = {"entry": entry, "tile": tile}

    def _issue(self, it: _Item, entry):
        if entry["queue"] is None:
            return
        ev = it.ev
        counter = self.s.queues[entry["queue"]].counter
        fifo = self.fifo[it.lane][counter]
        fifo.append(entry)
        outstanding = sum(e["count"] for e in fifo)
        if outstanding > self.cmax[counter]:
            self.find(
                "PERF.COUNTER_SATURATED",
                "info",
                f"{outstanding} {counter} instructions outstanding after {ev.op} (> {self.cmax[counter]}); "
                "issue stalls until one retires",
                ev.op,
                ev.where,
            )
        for name in ev.op.outs:
            self.value_entry[it.lane][name] = entry

    def _finish(self):
        for lane in range(len(self.lanes)):
            self.lane = lane
            for counter, q in self.fifo[lane].items():
                for e in q:
                    tracked = self.s.queues[e["queue"]].tracked
                    self.find(
                        "SYNC.UNDRAINED",
                        "warn" if tracked else "error",
                        f"{e['label']} is still outstanding on {counter} at the end of the program",
                        where=f"end:{e['label']}",
                        hint=f"end with a {counter}cnt(0) wait",
                    )
            for tensor, policy in self.s.tensors.items():
                got = self.consumed[lane].get(tensor, {})
                missing = sorted(set(range(self.n)) - set(got))
                dup = sorted(t for t, c in got.items() if c > 1)
                extra = sorted(set(got) - set(range(self.n)))
                if missing or dup or extra:
                    self.find(
                        "SEM.COVERAGE",
                        "error",
                        f"{tensor} ({policy}): missing tiles {missing}, consumed twice {dup}, out of range {extra}",
                        where=f"coverage:{tensor}",
                    )


def _sections(sched: Schedule, counts: Sequence[int]):
    if sched.program is None:
        return sched.all_sections()
    seen, out = set(), []
    for n in counts:
        for sec in sched.program(n):
            if id(sec) not in seen:
                seen.add(id(sec))
                out.append(sec)
    return out


def _sched_groups(sections, report: Report):
    """``sched_group_barrier`` MFMA requests must be satisfiable inside their region."""
    for sec in sections:
        region, start = [], 0
        for idx, op in enumerate(list(sec.ops) + [None]):
            boundary = (
                op is None
                or op.kind in ("barrier", "stagger")
                or (op.kind == "sched_barrier" and op.args.get("mask", 0) == 0)
            )
            if not boundary:
                region.append(op)
                continue
            have = sum(o.count for o in region if o.kind == "mfma")
            want = sum(o.args["size"] for o in region if o.kind == "sched_group" and o.args["cls"] == "mfma")
            if want > have:
                report.add(
                    ("SCHED.GROUP_UNSATISFIABLE", sec.name, start),
                    Finding(
                        "SCHED.GROUP_UNSATISFIABLE",
                        "warn",
                        f"sched_group_barrier asks for {want} MFMA but the region issues {have}; "
                        "LLVM silently leaves the extra groups unmatched",
                        f"{sec.name}/ops[{start}:{idx}] '{region[0]}' .. '{region[-1]}'",
                        "make the interleave pattern match the region's MFMA count",
                    ),
                )
            region, start = [], idx + 1


def _scopes(sections, report: Report, prefix: str = ""):
    """SSA scoping across ``scf.for``: what the emitter can actually reach.

    Inside a loop body a value is readable if defined earlier in the body, carried,
    or defined before the loop and never redefined in it.  After the loop the same
    holds with "earlier in the body" replaced by "carried out of the loop".
    """
    defined = set()
    for sec in sections:
        if sec.loop is None:
            for op in sec.ops:
                for name in op.ins:
                    if name not in defined:
                        report.add(
                            ("SCOPE.UNDEFINED", prefix + sec.name, name),
                            Finding(
                                "SCOPE.UNDEFINED",
                                "error",
                                f"{op} reads {name!r} before any definition",
                                f"{prefix}{sec.name}/{op}",
                            ),
                        )
                defined.update(op.outs)
            continue
        redefined = {n for op in sec.ops for n in op.outs}
        visible = (defined - redefined) | set(sec.carried)
        for c in sec.carried:
            if c not in defined:
                report.add(
                    ("SCOPE.CARRY_INIT", prefix + sec.name, c),
                    Finding(
                        "SCOPE.CARRY_INIT", "error", f"carried value {c!r} has no initial definition", prefix + sec.name
                    ),
                )
        local = set()
        for op in sec.ops:
            for name in op.ins:
                if name not in local and name not in visible:
                    why = "is redefined in the loop but not carried" if name in defined else "is never defined"
                    report.add(
                        ("SCOPE.UNCARRIED", prefix + sec.name, name),
                        Finding(
                            "SCOPE.UNCARRIED",
                            "error",
                            f"{op} reads {name!r} from a previous iteration, but it {why}",
                            f"{prefix}{sec.name}/{op}",
                            f"add {name!r} to the loop's carried values or define it earlier in the body",
                        ),
                    )
            local.update(op.outs)
        defined = (defined - redefined) | set(sec.carried)


def verify(sched: Schedule, tile_counts: Sequence[int] = (), target: Optional[TargetDesc] = None) -> Report:
    """Run every check for each tile count; returns a :class:`Report`."""
    target = target or load_target(sched.target)
    counts = [n for n in (tile_counts or sched.meta.get("verify_tiles", (4,))) if sched.valid_tiles(n)]
    report = Report(sched.name, target.key, list(counts))
    if sched.group_sections is not None:  # each group's program is scoped on its own
        for g, secs in enumerate(sched.group_sections):
            _scopes(secs, report, prefix=f"g{g}:")
    elif sched.program is None:
        _scopes(sched.sections, report)
    else:
        for n in counts:  # each tile count is its own straight-line program
            _scopes(sched.program(n), report)
    _sched_groups(_sections(sched, counts), report)
    from .hazards import check as hazard_check  # target-driven checks (targets/SCHEMA.md)

    simulate = hazard_check(sched, target, _sections(sched, counts), report)  # False: counter table rules it out
    for n in counts if simulate else ():
        try:
            _Sim(sched, target, n, report).run()
        except UnknownFact as exc:
            report.add(("TARGET.UNKNOWN", str(exc)), Finding("TARGET.UNKNOWN", "error", str(exc), n_tiles=n))
    from .pins import check as pin_check  # register pins (PIN.*); no-op without pins

    pin_check(sched, target, report)
    return report
