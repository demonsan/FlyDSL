# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-driven hazard checks: every number comes from the target description.

A check runs only when the target describes the facts it needs (see
``targets/SCHEMA.md``); a described-but-null fact is reported as
``TARGET.UNKNOWN`` instead of being guessed.

* Counter table (``counters``): every queue / wait names a counter the target
  has; a queue only carries op kinds its counter counts (``kinds``); a wait
  value other than 0 is invalid on an out-of-order counter (``in_order``,
  ``ooo_kinds``).
* Matrix dependencies (``matrix_hazards``): inside one straight-line section
  (a loop body also wraps into its next iteration), the instructions issued
  between a matrix op and a dependent op -- the sum of ``op.count`` of the ops in
  between, ``s_nop n`` counting ``n + 1``, scheduler pseudo-ops 0 -- must reach
  the target's required count.  Too few is an error where the hardware does not
  interlock, and a stall (info) where it does.  The distance is measured from
  the producer op's *last* instruction to the consumer op's *first*.  An op with
  ``count=0`` does not describe its instructions: it is never a producer or
  consumer, and in between it makes the distance a lower bound (error -> warn).
* LDS bank conflicts (``lds_banks``, ``lds_bank_bytes``, ``lds_lanes_per_cycle``):
  for ``ds_read`` / ``ds_write`` ops that declare ``args["lds_stride_bytes"]``
  (byte stride between consecutive lanes) and ``args["lds_access_bytes"]``.
"""

from math import ceil
from typing import Dict, List, Optional

from .ir import MEMORY_KINDS, Op, Schedule
from .target import TargetDesc, UnknownFact

PSEUDO_KINDS = ("sched_barrier", "sched_group", "anchor")
VALU_KINDS = ("valu", "trans")
WAIT_COUNTERS = ("vm", "lgkm")  # the counters the simulator's wait model drains


def issued(op: Op) -> int:
    """Instructions (wait states) ``op`` contributes to a dependency distance."""
    if op.kind in PSEUDO_KINDS:
        return 0
    if op.kind == "nop":
        return int(op.args.get("n", 0)) + 1
    return op.count


def check(sched: Schedule, target: TargetDesc, sections, report) -> bool:
    """Run every check the target supports.  Returns False when the counter table rules out simulation."""
    ctx = _Ctx(target, report)
    ok = _counters(sched, target, sections, ctx)
    _matrix(sections, target, ctx)
    _lds_banks(sections, target, ctx)
    return ok


class _Ctx:
    def __init__(self, target: TargetDesc, report):
        from .verify import Finding  # verify imports this module

        self.t, self.r, self.Finding = target, report, Finding

    def add(self, key, rule, sev, msg, where="", hint=""):
        self.r.add((rule,) + key, self.Finding(rule, sev, msg, where, hint))

    def unknown(self, what: str, why: str):
        self.add((what,), "TARGET.UNKNOWN", "info", f"{self.t.key}: {what} is unknown (null); {why} not checked")

    def cite(self, name: str, key: str) -> str:
        p = self.t.provenance(name, key)
        return f"{name}.{key}: {p.source} [{p.confidence}]"


# ---------------------------------------------------------------------------
# counters


def _wait_values(op: Op) -> Dict[str, int]:
    if op.kind == "wait":
        return {k: v for k, v in op.args.items() if v is not None}
    if op.kind == "barrier":
        return {k: op.args[k] for k in WAIT_COUNTERS if op.args.get(k) is not None}
    return {}


def _counters(sched: Schedule, target: TargetDesc, sections, ctx: _Ctx) -> bool:
    table = target.get("counters")
    if table is None:
        return True  # the simulator reports the missing table itself
    ok = True
    names = sorted(table)
    for q in sched.queues.values():
        if q.counter not in table:
            ok = False
            ctx.add(
                (q.name,),
                "HW.COUNTER_UNKNOWN",
                "error",
                f"queue {q.name!r} is counted on {q.counter!r}, which {target.key} does not have (counters: {names})",
                f"queue:{q.name}",
            )
    ooo_used = {}  # counter -> op kinds that return out of order on it and occur in the schedule
    for sec in sections:
        for op in sec.ops:
            q = sched.queues.get(op.queue) if op.queue else None
            entry = table.get(q.counter) if q is not None else None
            if entry is None:
                continue
            kinds = entry.get("kinds")
            if kinds is not None and op.kind not in kinds:
                ctx.add(
                    (q.name, op.kind),
                    "HW.COUNTER_MISMATCH",
                    "error",
                    f"{op} ({op.kind}) is on queue {q.name!r} counted on {q.counter!r}, "
                    f"but on {target.key} {q.counter!r} counts {kinds}",
                    f"{sec.name}/{op}",
                    "declare the queue on the counter that counts this op kind",
                )
            if op.kind in entry.get("ooo_kinds", ()):
                ooo_used.setdefault(q.counter, set()).add(op.kind)
    for sec in sections:
        for op in sec.ops:
            for counter, n in _wait_values(op).items():
                entry = table.get(counter)
                if entry is None:
                    ok = ok and counter not in WAIT_COUNTERS
                    ctx.add(
                        (id(op), counter),
                        "HW.COUNTER_UNKNOWN",
                        "error",
                        f"{op} waits on {counter!r}, which {target.key} does not have (counters: {names})",
                        f"{sec.name}/{op}",
                    )
                    continue
                if n == 0:
                    continue
                if entry.get("in_order") is False:
                    ctx.add(
                        (id(op), counter),
                        "HW.WAIT_PARTIAL_OOO",
                        "error",
                        f"{op} waits for {counter}<={n}, but {counter} completes out of order on {target.key}: "
                        "only a wait for 0 identifies which operations finished",
                        f"{sec.name}/{op}",
                        f"wait for {counter}=0",
                    )
                elif counter in ooo_used:
                    ctx.add(
                        (id(op), counter),
                        "HW.WAIT_PARTIAL_OOO",
                        "warn",
                        f"{op} waits for {counter}<={n} while the schedule also counts {sorted(ooo_used[counter])} "
                        f"on {counter}, which complete out of order on {target.key}; the wait is only valid "
                        "if none of those is outstanding here",
                        f"{sec.name}/{op}",
                    )
    return ok


# ---------------------------------------------------------------------------
# matrix dependencies


def _rule(producer: Op, consumer: Op) -> List[str]:
    """Hazard rules that apply between ``producer`` and a later ``consumer``."""
    p_out, c_in, c_out = set(producer.outs), set(consumer.ins), set(consumer.outs)
    rules = []
    if producer.kind == "mfma":
        raw = p_out & c_in
        if consumer.kind == "mfma":
            for name in raw:
                if name in c_out:
                    exact = consumer.mfma == producer.mfma and c_out == p_out
                    rules.append("matrix_to_matrix_srcc_exact" if exact else "matrix_to_matrix_srcc_overlap")
                else:
                    rules.append("matrix_to_matrix_srcab")
        elif consumer.kind in VALU_KINDS:
            if raw or p_out & c_out:
                rules.append("matrix_to_valu")
            if (set(producer.ins) - p_out) & c_out:
                rules.append("matrix_src_to_valu_write")
        elif consumer.kind in MEMORY_KINDS and raw:
            rules.append("matrix_to_mem")
    elif producer.kind in VALU_KINDS and consumer.kind == "mfma" and p_out & c_in:
        rules.append("valu_write_to_matrix_read")
    return sorted(set(rules))


def _required(target: TargetDesc, entry: dict, producer: Op) -> Optional[int]:
    mf = producer.mfma
    if mf is not None and mf in (entry.get("by_op") or {}):
        return entry["by_op"][mf]
    by_passes = entry.get("by_passes") or {}
    if mf is not None and by_passes:
        try:
            passes = target.mfma(mf)["passes"]
        except (UnknownFact, KeyError):
            passes = None
        if str(passes) in by_passes:
            return by_passes[str(passes)]
    return entry.get("wait")


def _matrix(sections, target: TargetDesc, ctx: _Ctx):
    table = target.get("matrix_hazards")
    if table is None:
        return  # the target does not describe matrix hazards
    for sec in sections:
        ops = list(sec.ops)
        # a loop body continues into its next iteration, up to and including the producer itself
        seq = ops + ops if sec.loop is not None else ops
        for j, prod in enumerate(ops):
            if (prod.kind != "mfma" and prod.kind not in VALU_KINDS) or _undescribed(prod):
                continue
            dist, lower_bound = 0, False
            seen = set()
            end = j + len(ops) + 1 if sec.loop is not None else len(ops)
            for k in range(j + 1, end):
                cons = seq[k]
                for rule in _rule(prod, cons):
                    if rule in seen:
                        continue  # only the nearest consumer per rule matters
                    seen.add(rule)
                    if not _undescribed(cons):
                        _judge(ctx, target, table, rule, prod, cons, dist, sec, k >= len(ops), lower_bound)
                dist += issued(cons)
                lower_bound = lower_bound or _undescribed(cons)


def _undescribed(op: Op) -> bool:
    """``count=0`` on a real instruction op: the schedule does not say how many instructions it issues
    (the compiler places them), so it can be neither end of a distance and only bounds one from below."""
    return op.count == 0 and op.kind not in PSEUDO_KINDS and op.kind not in ("nop", "wait", "barrier", "stagger")


def _judge(ctx: _Ctx, target, table, rule, prod: Op, cons: Op, dist: int, sec, wrapped: bool, lower_bound: bool):
    entry = table.get(rule)
    if not isinstance(entry, dict):
        ctx.unknown(f"matrix_hazards.{rule}", f"{prod} -> {cons} distance")
        return
    need = _required(target, entry, prod)
    interlock = entry.get("interlock")
    if need is None or interlock is None:
        what = "wait" if need is None else "interlock"
        ctx.unknown(f"matrix_hazards.{rule}.{what}" + (f" ({prod.mfma})" if prod.mfma else ""), f"{rule} distance")
        return
    if dist >= need:
        return
    where = f"{sec.name}/{prod} -> {'next iteration ' if wrapped else ''}{cons}"
    msg = f"{cons} depends on {prod} ({rule}) after {dist} instruction(s); {target.key} requires {need}"
    cite = ctx.cite("matrix_hazards", rule)
    if interlock:
        ctx.add(
            (id(prod), id(cons), rule),
            "PERF.MATRIX_DEP_STALL",
            "info",
            f"{msg}; the hardware interlocks, so the consumer stalls ({cite})",
            where,
        )
    else:
        if lower_bound:
            msg += " (a lower bound: ops in between have count=0)"
        ctx.add(
            (id(prod), id(cons), rule),
            "HAZARD.MATRIX_DEP",
            "warn" if lower_bound else "error",
            f"{msg}; the hardware does not interlock ({cite})",
            where,
            f"place {need - dist} independent instruction(s) or s_nop wait states in between "
            "(LLVM's hazard recognizer pads intrinsic-emitted code with s_nop; inline asm is not covered)",
        )


# ---------------------------------------------------------------------------
# LDS bank conflicts


def _conflict_degree(stride: int, access: int, lanes: int, banks: int, bank_bytes: int) -> int:
    hits: Dict[int, set] = {}
    for lane in range(lanes):
        first = lane * stride // bank_bytes
        for d in range(max(1, ceil(access / bank_bytes))):
            word = first + d
            hits.setdefault(word % banks, set()).add(word)
    return max(len(w) for w in hits.values())


def _lds_banks(sections, target: TargetDesc, ctx: _Ctx):
    for sec in sections:
        for op in sec.ops:
            if op.kind not in ("ds_read", "ds_read_tr", "ds_write") or "lds_stride_bytes" not in op.args:
                continue
            banks, bank_bytes = target.get("lds_banks"), target.get("lds_bank_bytes")
            lanes_table = target.get("lds_lanes_per_cycle")
            access = int(op.args.get("lds_access_bytes", 4))
            width = f"b{access * 8}"
            lanes = None
            if isinstance(lanes_table, dict):
                lanes = lanes_table.get(width, lanes_table.get("default"))
                if isinstance(lanes, dict):
                    lanes = lanes.get("value")
            if banks is None or bank_bytes is None or lanes is None:
                ctx.unknown(f"LDS lanes/banks for {width}", "LDS bank conflicts")
                continue
            deg = _conflict_degree(int(op.args["lds_stride_bytes"]), access, int(lanes), banks, bank_bytes)
            if deg > 1:
                ctx.add(
                    (id(op),),
                    "PERF.LDS_BANK_CONFLICT",
                    "info",
                    f"{op}: lane stride {op.args['lds_stride_bytes']} B with {access} B accesses maps "
                    f"{lanes} lanes onto {banks} banks {deg}-way on {target.key}",
                    f"{sec.name}/{op}",
                    "pad or swizzle the LDS layout so consecutive lanes hit distinct banks",
                )
