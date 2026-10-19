# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cross-block schedule transformations, proven legal by the verifier.

LLVM schedules one basic block (one scheduling region) at a time: it cannot move
a DMA from the top of loop iteration ``j`` to the bottom of iteration ``j-1``,
nor across an ``s_barrier`` into another phase, nor re-derive the ``s_waitcnt``
values afterwards.  The Schedule IR sees the whole program -- prologue, every
cluster of the loop body, the loop back-edge and the peeled epilogue -- so these
moves are ordinary list edits here:

* :func:`move_op`      move an op to another position or another straight-line
                       section, across barriers and clusters;
* :func:`rotate_up`    move a loop-body op across the back-edge into the previous
                       iteration; the first instance goes to the prologue and the
                       matching op at the head of the epilogue is absorbed, so no
                       runtime guard is needed (works for "loop + peeled
                       epilogue" programs such as the dual-wave MHA);
* :func:`synthesize_waits`  recompute every ``s_waitcnt`` to its loosest safe
                       value after ops moved;
* :func:`hoist_async` / :func:`sink_async`  greedy passes that move each async
                       memory op as early / late as legality allows, using the
                       moves above.

No transformation is trusted on its own: every candidate is accepted only if
:func:`flydsl.sched.verify.verify` reports no errors on all tile counts of the
schedule (RAW/WAR across barriers for every wave group, stale tiles, coverage,
SSA scope, counter encodings).
"""

import copy
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .ir import COMPUTE_KINDS, MEMORY_KINDS, Op, Schedule, Section, SlotUse, TileRef
from .target import TargetDesc, load_target
from .verify import trace, verify

Pos = Tuple[str, int]  # (section name, index)


@dataclass
class Step:
    action: str
    op: str
    detail: str


@dataclass
class TransformLog:
    steps: List[Step] = field(default_factory=list)
    verifies: int = 0

    def add(self, action, op, detail=""):
        self.steps.append(Step(action, str(op), detail))

    def __str__(self):
        lines = [f"{len(self.steps)} change(s), {self.verifies} verifier run(s)"]
        lines += [f"  {s.action:10s} {s.op:44s} {s.detail}" for s in self.steps]
        return "\n".join(lines)


class IllegalTransform(ValueError):
    """The requested edit is structurally impossible (not merely unverified)."""


# ---------------------------------------------------------------------------- helpers
def clone(sched: Schedule) -> Schedule:
    """Deep copy; callables (bases, loop ranges, workload) are shared."""
    return copy.deepcopy(sched)


def _ok(sched: Schedule, tiles: Sequence[int], log: Optional[TransformLog]) -> bool:
    return _check(sched, tiles, log)[0]


def _check(sched: Schedule, tiles: Sequence[int], log: Optional[TransformLog]) -> Tuple[bool, str]:
    """(ok, first error rule + message) -- the reason a move is refused."""
    if log is not None:
        log.verifies += 1
    r = verify(sched, tiles)
    if r.ok:
        return True, ""
    e = r.errors[0]
    return False, f"{e.rule} {e.message}"


def _shift_ref(ref, base_from: str, base_to: str, delta: int):
    if isinstance(ref, TileRef) and ref.base == base_from:
        return TileRef(base_to, ref.off + delta)
    return ref


def rebase(op: Op, base_from: str, base_to: str, delta: int) -> Op:
    """Copy of ``op`` with every tile reference ``base_from+k`` rewritten to ``base_to+k+delta``."""

    def su(u: SlotUse) -> SlotUse:
        return replace(u, tile=_shift_ref(u.tile, base_from, base_to, delta))

    label = op.label
    for v in op.args.values():
        new = _shift_ref(v, base_from, base_to, delta)
        if new is not v and label:
            label = label.replace(str(v), str(new))
    return replace(
        copy.deepcopy(op),
        label=label,
        args={k: _shift_ref(v, base_from, base_to, delta) for k, v in op.args.items()},
        reads=tuple(su(u) for u in op.reads),
        writes=tuple(su(u) for u in op.writes),
        consumes=tuple((t, _shift_ref(r, base_from, base_to, delta)) for t, r in op.consumes),
    )


def _signature(op: Op):
    """Identity of an op modulo its tile references (used to find the epilogue twin)."""
    args = tuple(sorted((k, v) for k, v in op.args.items() if not isinstance(v, TileRef)))
    slots = tuple((u.buffer, u.slot) for u in op.writes + op.reads)
    return (op.kind, op.binding, args, slots, op.queue, op.count, op.outs, op.ins)


def _loop_point(sched: Schedule, sec: Section, tiles: Sequence[int], which: str) -> Tuple[str, int]:
    """(base, c) with ``bases[base] + c`` equal, for every tile count, to the loop's

    * ``"first"``:  first induction value ``lb``;
    * ``"after"``:  ``lb + step * trips`` (the value one past the last iteration; ``lb`` if empty);
    * ``"before"``: ``lb - step``.
    """
    var, rng = sec.loop
    cand: Optional[Dict[str, int]] = None
    for n in tiles:
        b = sched.bases(n)
        r = rng(b)
        val = {"first": r.start, "after": r.start + r.step * len(r), "before": r.start - r.step}[which]
        offs = {k: val - v for k, v in b.items()}
        cand = offs if cand is None else {k: v for k, v in cand.items() if offs.get(k) == v}
    if not cand:
        raise IllegalTransform(f"loop {sec.name}: '{which}' is not a fixed offset of any tile base")
    base = sorted(cand)[0]
    return base, cand[base]


def _same(a: Op, b: Op) -> bool:
    """Same op including its (rebased) tile references."""
    return _signature(a) == _signature(b) and a.args == b.args and a.writes == b.writes and a.reads == b.reads


def _step(sched: Schedule, sec: Section, tiles: Sequence[int]) -> int:
    steps = {sec.loop[1](sched.bases(n)).step for n in tiles}
    if len(steps) != 1:
        raise IllegalTransform(f"loop {sec.name}: step depends on the tile count")
    return steps.pop()


def _sec_index(sched: Schedule, name: str) -> int:
    for i, s in enumerate(sched.sections):
        if s.name == name:
            return i
    raise KeyError(name)


def _tiles(sched: Schedule, tiles: Optional[Sequence[int]]) -> List[int]:
    return [n for n in (tiles or sched.meta.get("verify_tiles", (4,))) if sched.valid_tiles(n)]


# ---------------------------------------------------------------------------- primitive moves
def move_op(sched: Schedule, src: Pos, dst: Pos) -> Schedule:
    """Move one op between positions of straight-line code or within a loop body.

    ``dst`` is an insertion index into the destination section *after* the op has
    been removed from ``src``.  Moving between a loop body and code outside it changes
    how often the op runs, so that is refused here: use :func:`rotate_up`.
    """
    s = clone(sched)
    a, b = s.section(src[0]), s.section(dst[0])
    if (a.loop is not None or b.loop is not None) and a.name != b.name:
        raise IllegalTransform("moving into or out of a loop body changes the trip count of the op; use rotate_up")
    op = a.ops.pop(src[1])
    b.ops.insert(dst[1], op)
    return s


def rotate_up(sched: Schedule, loop: str, index: int, tiles: Optional[Sequence[int]] = None) -> Schedule:
    """Move body op ``index`` of loop ``loop`` across the back-edge into the previous iteration.

    for j: A(j); ...                 A(lb)                          (prologue tail)
                         ==>         for j: ...; A(j+step)          (body tail)
    epilogue: A(end-k); ...          epilogue: ...                  (twin absorbed)

    The op's last instance A(last+step) is exactly the epilogue's peeled twin, which is
    removed; with zero trips the prologue copy *is* that twin.  Values the op defines
    become loop-carried.  Legality is left to the verifier.
    """
    tiles = _tiles(sched, tiles)
    s = clone(sched)
    li = _sec_index(s, loop)
    body = s.sections[li]
    if body.loop is None:
        raise IllegalTransform(f"{loop} is not a loop section")
    if li == 0 or li == len(s.sections) - 1:
        raise IllegalTransform("rotate_up needs straight-line code before and after the loop")
    var = body.loop[0]
    step = _step(s, body, tiles)
    op = body.ops.pop(index)
    _check_rotatable(op)
    pro, epi = s.sections[li - 1], s.sections[li + 1]
    after = rebase(op, var, *_loop_point(s, body, tiles, "after"))
    twin = next((i for i, o in enumerate(epi.ops) if _same(o, after)), None)
    if twin is None:
        raise IllegalTransform(f"no peeled twin {after} of {op} in {epi.name}; a runtime guard would be needed")
    epi.ops.pop(twin)
    body.ops.append(rebase(op, var, var, step))
    pro.ops.append(rebase(op, var, *_loop_point(s, body, tiles, "first")))
    if op.outs:
        body.carried = tuple(body.carried) + tuple(o for o in op.outs if o not in body.carried)
    s.meta = dict(s.meta, xblock=list(s.meta.get("xblock", [])) + [f"rotate_up {op}"])
    return s


def rotate_down(sched: Schedule, loop: str, index: int, tiles: Optional[Sequence[int]] = None) -> Schedule:
    """Inverse of :func:`rotate_up`: move body op ``index`` to the top of the *next* iteration.

    for j: ...; A(j)                 for j: A(j-step); ...           (body head)
    prologue: ...; A(lb-step)  ==>   prologue: ...                   (twin absorbed)
                                     epilogue: A(after-step); ...    (last instance)
    """
    tiles = _tiles(sched, tiles)
    s = clone(sched)
    li = _sec_index(s, loop)
    body = s.sections[li]
    if body.loop is None or li == 0 or li == len(s.sections) - 1:
        raise IllegalTransform("rotate_down needs a loop with straight-line code on both sides")
    var = body.loop[0]
    step = _step(s, body, tiles)
    op = body.ops.pop(index)
    _check_rotatable(op)
    pro, epi = s.sections[li - 1], s.sections[li + 1]
    before = rebase(op, var, *_loop_point(s, body, tiles, "before"))
    twin = next((i for i in reversed(range(len(pro.ops))) if _same(pro.ops[i], before)), None)
    if twin is None:
        raise IllegalTransform(f"no twin {before} of {op} in {pro.name}; a runtime guard would be needed")
    pro.ops.pop(twin)
    body.ops.insert(0, rebase(op, var, var, -step))
    base, c = _loop_point(s, body, tiles, "after")
    epi.ops.insert(0, rebase(op, var, base, c - step))
    if op.outs:
        body.carried = tuple(body.carried) + tuple(o for o in op.outs if o not in body.carried)
    s.meta = dict(s.meta, xblock=list(s.meta.get("xblock", [])) + [f"rotate_down {op}"])
    return s


def _check_rotatable(op: Op):
    if op.kind not in MEMORY_KINDS and op.kind not in ("valu", "trans", "salu", "mfma"):
        raise IllegalTransform(f"{op} ({op.kind}) is a control op; rotate the ops around it instead")


# ---------------------------------------------------------------------------- waits
def _counters(sched: Schedule) -> List[str]:
    """Counters the target declares, except ``exp`` (exports are not scheduled here)."""
    return [c for c in load_target(sched.target).require("counters") if c != "exp"]


def _wait_slots(sched: Schedule) -> List[Tuple[str, int, str]]:
    out = []
    ctrs = _counters(sched)
    for sec in sched.sections:
        for i, op in enumerate(sec.ops):
            for ctr in ctrs:
                if op.kind in ("wait", "barrier") and op.args.get(ctr) is not None:
                    out.append((sec.name, i, ctr))
    return out


def synthesize_waits(
    sched: Schedule,
    tiles: Optional[Sequence[int]] = None,
    target: Optional[TargetDesc] = None,
    log: Optional[TransformLog] = None,
    keep_roles: bool = True,
) -> Schedule:
    """Set every explicit ``s_waitcnt`` field to the loosest value the verifier accepts.

    If the schedule verifies as is, waits are only *loosened* (each field binary-searched
    upward from its current value, in program order), so the author's placement is kept
    and only provably unnecessary waiting is removed.  Otherwise every field is reset to 0
    (safe if anything is) and raised in program order.

    ``keep_roles``: a looser value is rejected if it moves the point where any async op
    completes from a memory cluster into a compute cluster (the stall would then land in
    the MFMA stream -- measured 1-2% slower on the dual-wave MHA, see docs).
    """
    tiles = _tiles(sched, tiles)
    target = target or load_target(sched.target)
    cmax = {k: v["max"] for k, v in target.require("counters").items()}
    s = clone(sched)
    slots = _wait_slots(s)
    before = {(sec, i, c): s.section(sec).ops[i].args[c] for sec, i, c in slots}
    if not _ok(s, tiles, log):
        for sec, i, c in slots:
            s.section(sec).ops[i].args[c] = 0
        if not _ok(s, tiles, log):
            raise IllegalTransform("schedule is not legal even with every wait at 0")
    for sec, i, c in slots:
        op = s.section(sec).ops[i]
        lo, hi = op.args[c], cmax[c]
        base_roles = _drain_roles(s) if keep_roles else None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            op.args[c] = mid
            ok = _ok(s, tiles, log)
            if ok and keep_roles:
                now = _drain_roles(s)
                ok = not any(r == "memory" and now.get(k) == "compute" for k, r in base_roles.items())
            if ok:
                lo = mid
            else:
                hi = mid - 1
        op.args[c] = lo
        if log is not None and lo != before[(sec, i, c)]:
            log.add("wait", f"{sec}[{i}] {op}", f"{c}cnt {before[(sec, i, c)]} -> {lo}")
        _relabel_wait(op)
    return s


def _relabel_wait(op: Op):
    if op.kind != "wait":
        return
    parts = [f"{k}cnt({v})" for k, v in op.args.items() if v is not None]
    op.label = "s_waitcnt " + " ".join(parts)


# ---------------------------------------------------------------------------- analysis
def cluster_roles(sched: Schedule) -> Dict[int, str]:
    """id(op) -> "compute" | "memory" for every op: the role of the barrier-delimited
    cluster it sits in (a cluster with any MFMA is a compute cluster).  In the dual-wave
    design each SIMD pairs one group's compute cluster with the other's memory cluster."""
    roles: Dict[int, str] = {}
    for sec in sched.sections:
        cluster: List[Op] = []
        for op in list(sec.ops) + [None]:
            if op is None or op.kind in ("barrier", "stagger"):
                role = "compute" if any(o.kind == "mfma" for o in cluster) else "memory"
                for o in cluster + ([op] if op is not None else []):
                    roles[id(o)] = role
                cluster = []
            else:
                cluster.append(op)
    return roles


def _role_at(sched: Schedule, section: str, index: int) -> str:
    return cluster_roles(sched)[id(sched.section(section).ops[index])]


def _drain_roles(sched: Schedule) -> Dict[Tuple[str, int], str]:
    """(op label, instance #) -> role of the cluster in which that async instance completes."""
    out: Dict[Tuple[str, int], str] = {}
    seen: Dict[str, int] = {}
    for c in async_cover(sched):
        k = seen.get(c.op, 0)
        seen[c.op] = k + 1
        out[(c.op, k)] = c.drain_role
    return out


@dataclass
class Cover:
    op: str
    section: str
    issue: int
    drained_at: Optional[int]
    mfma_cycles: int  # MFMA work of the issuing wave between issue and drain
    barriers: int
    drain_role: str = ""  # role of the cluster where it completes
    work: int = 0  # compute/memory ops (not control ops) between issue and completion


def async_cover(sched: Schedule, n_tiles: Optional[int] = None, target: Optional[TargetDesc] = None) -> List[Cover]:
    """How much MFMA work hides each async op's latency (single wave, program order).

    An async op is covered by the MFMA cycles its own wave issues between the op and
    the wait (explicit, or the compiler's for tracked queues) that drains it.  Other
    waves' overlap comes on top; this is the part the schedule controls directly.
    """
    target = target or load_target(sched.target)
    n = n_tiles or max(_tiles(sched, None))
    events = trace(sched, n)
    fifo: Dict[str, List[dict]] = {c: [] for c in target.require("counters")}
    produced: Dict[str, dict] = {}
    entries: List[dict] = []
    mfma_acc = barriers = work = 0

    def drain(ctr, keep, pos):  # same retirement rule as the verifier
        q = fifo[ctr]
        total = sum(e["count"] for e in q)
        while q and total - q[0]["count"] >= keep:
            total -= q[0]["count"]
            e = q.pop(0)
            e["done"], e["mf_end"], e["bar_end"], e["w_end"] = pos, mfma_acc, barriers, work

    for i, ev in enumerate(events):
        op = ev.op
        for ctr in fifo:
            if op.kind in ("wait", "barrier") and op.args.get(ctr) is not None:
                drain(ctr, op.args[ctr], i)
        for name in op.ins:
            e = produced.get(name)
            if e is not None and e["done"] is None and sched.queues[e["queue"]].tracked:
                q = fifo[e["ctr"]]
                if e in q:
                    drain(e["ctr"], sum(x["count"] for x in q[q.index(e) + 1 :]), i)
        if op.kind == "mfma":
            mfma_acc += op.count * target.mfma_cycles(op.mfma)
        if op.kind in ("barrier", "stagger"):
            barriers += 1
        if op.kind in COMPUTE_KINDS or op.kind in MEMORY_KINDS:
            work += 1
        if op.queue is not None:
            ctr = sched.queues[op.queue].counter
            e = dict(
                op=str(op),
                where=ev.where.split("/")[0],
                pos=i,
                count=op.count,
                queue=op.queue,
                ctr=ctr,
                done=None,
                mf0=mfma_acc,
                bar0=barriers,
                mf_end=None,
                bar_end=None,
                w0=work,
                w_end=None,
            )
            fifo[ctr].append(e)
            entries.append(e)
            for name in op.outs:
                produced[name] = e
    roles = cluster_roles(sched)
    out = []
    for e in entries:
        done = e["done"] is not None
        role = roles.get(id(events[e["done"]].op), "") if done else ""
        out.append(
            Cover(
                e["op"],
                e["where"],
                e["pos"],
                e["done"],
                (e["mf_end"] - e["mf0"]) if done else -1,
                (e["bar_end"] - e["bar0"]) if done else -1,
                role,
                (e["w_end"] - e["w0"]) if done else -1,
            )
        )
    return out


def cover_summary(sched: Schedule, n_tiles: Optional[int] = None) -> Dict[str, Tuple[int, int]]:
    """Per async op label in the loop: (min MFMA cycles of cover, min barriers crossed) over its instances."""
    loop = next((s.name for s in sched.sections if s.loop is not None), None)
    res: Dict[str, Tuple[int, int]] = {}
    for c in async_cover(sched, n_tiles):
        if loop and not c.section.startswith(loop) or c.drained_at is None:
            continue
        m, b = res.get(c.op, (c.mfma_cycles, c.barriers))
        res[c.op] = (min(m, c.mfma_cycles), min(b, c.barriers))
    return res


# ---------------------------------------------------------------------------- passes
def _movable(op: Op, kinds) -> bool:
    return op.kind in kinds


def _cover_map(sched: Schedule) -> Dict[str, Tuple[int, int]]:
    """label -> (min MFMA cycles, min instruction distance) between issue and drain, over all instances."""
    out: Dict[str, Tuple[int, int]] = {}
    for c in async_cover(sched):
        if c.drained_at is None:
            continue
        v = (c.mfma_cycles, c.work)
        out[c.op] = min(out.get(c.op, v), v)
    return out


def _improves(old: Schedule, new: Schedule, old_label: str, new_label: str) -> Tuple[bool, str]:
    """A hoist step must not shorten the moved op's latency cover (MFMA cycles, then distance)
    nor take MFMA cover away from any other async op."""
    a, b = _cover_map(old), _cover_map(new)
    if new_label in b and old_label in a and b[new_label] < a[old_label]:
        return False, f"cover of {new_label} would drop {a[old_label]} -> {b[new_label]} (an earlier wait drains it)"
    for k, v in a.items():
        if k != old_label and k in b and b[k][0] < v[0]:
            return False, f"cover of {k} would drop {v[0]} -> {b[k][0]} MFMA cycles"
    return True, ""


def hoist_async(
    sched: Schedule,
    kinds: Sequence[str] = ("lds_dma", "vmem_load"),
    sections: Optional[Sequence[str]] = None,
    cross_backedge: bool = True,
    resynthesize_waits: bool = True,
    tiles: Optional[Sequence[int]] = None,
    max_steps: int = 256,
    keep_roles: bool = True,
) -> Tuple[Schedule, TransformLog]:
    """Move every async op of ``kinds`` as early as legality allows, across clusters and
    (``cross_backedge``) into the previous loop iteration.

    The op walks upward one position at a time; each position where it may *land* is
    checked with the verifier and must not shorten any async op's latency cover.  With
    ``keep_roles`` (default) memory ops pass through compute clusters but only land in
    memory clusters, preserving the dual-wave split of MFMA and memory phases.  Waits are
    kept while moving and re-synthesized to their loosest safe values at the end.
    """
    tiles = _tiles(sched, tiles)
    log = TransformLog()
    s = clone(sched)
    if not _ok(s, tiles, log):
        raise IllegalTransform("input schedule does not verify")
    names = list(sections) if sections else [sec.name for sec in s.sections]
    for name in names:
        done = set()  # labels already placed (a rotated op reappears at the end of the body)
        idx = 0
        while idx < len(s.section(name).ops):
            op = s.section(name).ops[idx]
            if not _movable(op, kinds) or str(op) in done:
                idx += 1
                continue
            done.add(str(op))
            ctr = s.queues[op.queue].counter if op.queue else None
            cmax = load_target(s.target).counter_max(ctr) if ctr else 0
            walk, pos, rotated = s, idx, False
            landed, landed_rot, blocker = None, False, ""
            best = _cover_map(s).get(str(op))
            for _ in range(max_steps):
                if pos > 0:
                    crossed = walk.section(name).ops[pos - 1]
                    walk, pos = move_op(walk, (name, pos), (name, pos - 1)), pos - 1
                    w = walk.section(name).ops[pos + 1]
                    if ctr and w.kind in ("wait", "barrier") and w.args.get(ctr) is not None:
                        # Compound move: issuing the op before a wait on its own counter keeps
                        # the wait's meaning only if the wait allows op.count more in flight.
                        if w.args[ctr] + op.count > cmax:
                            blocker = f"{crossed}: {ctr}cnt would exceed {cmax}"
                            break
                        w.args[ctr] += op.count
                        _relabel_wait(w)
                elif cross_backedge and walk.section(name).loop is not None and not rotated:
                    try:
                        walk = rotate_up(walk, name, 0, tiles)
                    except IllegalTransform as exc:
                        blocker = f"back-edge: {exc}"
                        break
                    rotated, pos = True, len(walk.section(name).ops) - 1
                    continue  # body end == next iteration's head: same instant, nothing gained yet
                else:
                    break
                moved_op = walk.section(name).ops[pos]
                if keep_roles and _role_at(walk, name, pos) == "compute":
                    continue  # pass through a compute cluster without landing in it
                ok, why = _check(walk, tiles, log)
                if ok:
                    ok, why = _improves(s, walk, str(op), str(moved_op))
                if not ok:
                    where = "previous iteration: " if rotated else ""
                    blocker = f"{where}{crossed}: {why}"
                    break
                now = _cover_map(walk).get(str(moved_op))
                if best is not None and now is not None and now <= best:
                    continue  # legal but no better: pass through, land only where cover strictly grows
                s, landed, landed_rot, best = walk, pos, rotated, now
                done.add(str(moved_op))
            if landed is not None:
                if landed_rot:
                    log.add("rotate_up", op, "issued one iteration earlier (prologue copy, epilogue twin absorbed)")
                log.add(
                    "hoist", op, f"{name}: index {idx} -> {landed}" + (" (previous iteration)" if landed_rot else "")
                )
            if blocker:
                log.add("stopped", op, blocker)
            idx += 1
    if resynthesize_waits:
        s = synthesize_waits(s, tiles, log=log)
    return s, log


def sink_async(
    sched: Schedule,
    kinds: Sequence[str] = ("lds_dma",),
    sections: Optional[Sequence[str]] = None,
    tiles: Optional[Sequence[int]] = None,
    max_steps: int = 64,
) -> Tuple[Schedule, TransformLog]:
    """Move async ops as late as legality allows (fixed waits).  Produces the naive
    "issue just in time" placement, used to show what the hoist pass recovers."""
    tiles = _tiles(sched, tiles)
    log = TransformLog()
    s = clone(sched)
    names = list(sections) if sections else [sec.name for sec in s.sections]
    for name in names:
        sec = s.section(name)
        for idx in reversed(range(len(sec.ops))):
            op = sec.ops[idx]
            if not _movable(op, kinds):
                continue
            pos, moved = idx, 0
            while moved < max_steps and pos + 1 < len(s.section(name).ops):
                cand = move_op(s, (name, pos), (name, pos + 1))
                if not _ok(cand, tiles, log):
                    break
                s, pos, moved = cand, pos + 1, moved + 1
            if moved:
                log.add("sink", op, f"{name}: index {idx} -> {pos}")
    return s, log


def describe_positions(sched: Schedule, section: str) -> List[str]:
    """Cluster label (C0, C1, ...) of every op index in a section; clusters end at barriers."""
    out, c = [], 0
    for op in sched.section(section).ops:
        out.append(f"C{c}")
        if op.kind in ("barrier", "stagger"):
            c += 1
    return out


def where_is(sched: Schedule, pred: Callable[[Op], bool]) -> List[Pos]:
    return [(sec.name, i) for sec in sched.sections for i, op in enumerate(sec.ops) if pred(op)]
