# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register pins: "place this value in v[a:b]" as Schedule IR data.

A pin is a register-allocation hint at a value's *definition*:

* ``Op.pins = {out: "v[64:79]"}`` -- the value this op defines is hinted to the range;
* ``Section.pins = {carried: "v[192:255]"}`` -- a loop-carried value is hinted to the
  range wherever the loop defines or receives it: the initial value before the loop,
  the iteration argument at the top of the body (the loop-header phi), the new value
  at the end of the body and the loop result.

A spec is ``"v42"``, ``"v[64:79]"``, ``"a[0:3]"`` or ``"s10"``/``"s[4:5]"``, strict by
default; append ``" soft"`` (``"v[64:79] soft"``) for an advisory hint.

Mechanism (ROCm/FlyDSL#1307; required LLVM patches ``schedule-bank-and-gpr.patch`` and
``schedule-gpr-hint-coalescing.patch`` on the pinned LLVM, ``thirdparty/llvm-build-info.json``):
each hint is ``rocdl.sched.{vgpr,agpr,sgpr} %v, regnum [soft]``, an identity op that
lowers to ``llvm.amdgcn.sched.*`` and a pseudo; the pre-RA ``AMDGPUSchedBank`` pass
turns it into a COPY and attaches an allocation hint to both sides, propagated through
COPY / REG_SEQUENCE / INSERT_SUBREG / SUBREG_TO_REG, and the coalescer carries it onto
the register it keeps (``SIRegisterInfo::updateRegAllocHint``, the second patch; without
it a hint on a loop-carried value is lost when the phi is coalesced).  A *strict* hint is
a hard hint (``getRegAllocationHints`` returns true: the allocator considers only that
register for the hinted virtual registers) and is never replaced by a soft one; a *soft*
one only moves the register to the front.  Widths are
32/64/128/256/512 bits (``WIDTHS``); a wider value is split into leaves that take
consecutive sub-ranges, and the backend silently drops a hint of any other width, so the
verifier rejects those.

Nothing is reserved: a pinned value occupies the range while it is live, other values
may use it otherwise.  What actually happened is visible only on the ISA:
:func:`isa_pin_report` lists who writes and reads each range, MFMAs that accumulate in
place, and the register copies into or out of it (``PIN.INDIRECT``).

History: an earlier version pinned through ``llvm.read_register`` /
``llvm.write_register`` with a named-register LLVM branch; it was replaced by this
mechanism (user decision, 2026-10-20).

This module is pure Python (MLIR is imported lazily by the lowering helpers).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import Op, Schedule, Section
from .target import TargetDesc, load_target

# Widths (dwords) the AMDGPU ISel lowers llvm.amdgcn.sched.{vgpr,agpr,sgpr} to a pseudo for
# (B32/B64/B128/B256/B512 in schedule-bank-and-gpr.patch); any other width is dropped silently.
WIDTHS = {"v": (1, 2, 4, 8, 16), "a": (1, 2, 4, 8, 16), "s": (1, 2, 4, 8, 16)}
_FILE_FACT = {"v": "arch_vgpr_max_per_wave", "a": "acc_vgpr_max_per_wave", "s": "sgpr_addressable_per_wave"}
_OPS = {"v": "sched_vgpr", "a": "sched_agpr", "s": "sched_sgpr"}

_REG_RE = re.compile(r"^([vas])(?:(\d+)|\[(\d+):(\d+)\])$")


@dataclass(frozen=True)
class RegRange:
    """``v[lo:hi]`` / ``a[lo:hi]`` / ``s[lo:hi]`` (inclusive), or one register ``v5``."""

    file: str
    lo: int
    hi: int

    @classmethod
    def parse(cls, text: str) -> "RegRange":
        m = _REG_RE.match(str(text).strip())
        if m is None:
            raise ValueError(f"bad register range {text!r}: expected 'v12', 'v[64:79]', 'a[0:15]' or 's[4:5]'")
        f, one, lo, hi = m.groups()
        lo, hi = (int(one), int(one)) if one is not None else (int(lo), int(hi))
        if hi < lo:
            raise ValueError(f"bad register range {text!r}: {hi} < {lo}")
        return cls(f, lo, hi)

    @property
    def dwords(self) -> int:
        return self.hi - self.lo + 1

    def overlaps(self, other: "RegRange") -> bool:
        return self.file == other.file and self.lo <= other.hi and other.lo <= self.hi

    def sub(self, offset: int, dwords: int) -> "RegRange":
        if offset < 0 or offset + dwords > self.dwords:
            raise ValueError(f"{self}: no {dwords} dwords at offset {offset}")
        return RegRange(self.file, self.lo + offset, self.lo + offset + dwords - 1)

    def __str__(self):
        return f"{self.file}{self.lo}" if self.lo == self.hi else f"{self.file}[{self.lo}:{self.hi}]"


@dataclass(frozen=True)
class PinSpec:
    """A register range and whether the hint is strict (default) or soft."""

    reg: RegRange
    strict: bool = True

    @classmethod
    def parse(cls, text) -> "PinSpec":
        if isinstance(text, PinSpec):
            return text
        parts = str(text).split()
        if len(parts) == 2 and parts[1] in ("soft", "strict"):
            return cls(RegRange.parse(parts[0]), parts[1] == "strict")
        if len(parts) != 1:
            raise ValueError(f"bad pin {text!r}: expected '<range>' or '<range> soft'")
        return cls(RegRange.parse(parts[0]))

    def __str__(self):
        return str(self.reg) if self.strict else f"{self.reg} soft"


# ---------------------------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------------------------


@dataclass(eq=False)
class PinDecl:
    value: str
    spec: str  # as written
    scope: str  # "op" | "carried"
    sec: Section = field(repr=False)
    op_index: Optional[int] = None  # op pins: index in the section
    group: Optional[int] = None  # per-group programs: the group whose program declares it

    @property
    def section(self) -> str:
        return self.sec.name

    @property
    def where(self) -> str:
        g = f"g{self.group}:" if self.group is not None else ""
        at = f"/ops[{self.op_index}]" if self.op_index is not None else ""
        return f"{g}{self.section}{at} {self.value}->{self.spec}"


def _programs(sched: Schedule) -> List[Tuple[Optional[int], List[Section]]]:
    groups = getattr(sched, "group_sections", None)
    if groups is not None:
        return [(g, list(secs)) for g, secs in enumerate(groups)]
    if sched.program is not None:
        seen, out = set(), []
        for n in sched.meta.get("verify_tiles", (4,)):
            if sched.valid_tiles(n):
                key = tuple(id(s) for s in sched.program(n))
                if key not in seen:
                    seen.add(key)
                    out.append((None, list(sched.program(n))))
        return out
    return [(None, list(sched.sections))]


def declarations(sched: Schedule) -> List[PinDecl]:
    """Every pin of every program, once per (section, op) object."""
    out, seen = [], set()
    for g, secs in _programs(sched):
        for sec in secs:
            for value, spec in getattr(sec, "pins", {}).items():
                key = (id(sec), None, value)
                if key not in seen:
                    seen.add(key)
                    out.append(PinDecl(value, spec, "carried", sec, None, g))
            for i, op in enumerate(sec.ops):
                for value, spec in getattr(op, "pins", {}).items():
                    key = (id(sec), i, value)
                    if key not in seen:
                        seen.add(key)
                        out.append(PinDecl(value, spec, "op", sec, i, g))
    return out


def has_pins(sched: Schedule) -> bool:
    return any(
        getattr(sec, "pins", None) or any(getattr(op, "pins", None) for op in sec.ops)
        for _, secs in _programs(sched)
        for sec in secs
    )


# ---------------------------------------------------------------------------------------------
# Verifier (PIN.* rules)
# ---------------------------------------------------------------------------------------------


def _positions(secs: Sequence[Section]):
    """Linear position of every op and the (first, last) span of every section."""
    pos, spans, p = {}, {}, 0
    for si, sec in enumerate(secs):
        start = p
        for oi in range(len(sec.ops)):
            pos[(si, oi)] = p
            p += 1
        spans[si] = (start, max(p - 1, start))
    return pos, spans


def _live_until(secs, si, oi, value, pos, spans):
    """Last position a definition of ``value`` at (si, oi) is read before it is redefined.

    A definition outside a loop that the loop reads is live around the whole loop."""
    start = pos[(si, oi)] if oi is not None else spans[si][0]
    end = start
    for sj, sec in enumerate(secs):
        for oj, op in enumerate(sec.ops):
            if pos[(sj, oj)] <= start:
                continue
            if value in op.ins:
                end = max(end, pos[(sj, oj)])
                if sj != si and sec.loop is not None:
                    end = max(end, spans[sj][1])
            if value in op.outs:
                return end
    return end


def _lifetime(secs, si, oi, value, pos, spans):
    """[first, last] position where a pinned definition (op pin) or carried value (oi None) is live."""
    if oi is not None:
        return pos[(si, oi)], _live_until(secs, si, oi, value, pos, spans)
    # carried: from its last definition before the loop to its last use after it
    start = spans[si][0]
    for sj in range(si - 1, -1, -1):
        hit = [oj for oj, op in enumerate(secs[sj].ops) if value in op.outs]
        if hit:
            start = pos[(sj, hit[-1])]
            break
    end = spans[si][1]
    for sj in range(si + 1, len(secs)):
        for oj, op in enumerate(secs[sj].ops):
            if value in op.ins:
                end = pos[(sj, oj)]
            if value in op.outs:
                return start, end
    return start, end


def check(sched: Schedule, target: Optional[TargetDesc] = None, report=None):
    """Add ``PIN.*`` findings for ``sched`` to ``report`` (a ``verify.Report``); returns the report."""
    from .verify import Finding, Report  # lazy: verify calls this module

    target = target or load_target(sched.target)
    if report is None:
        report = Report(sched.name, target.key, [])
    decls = declarations(sched)
    if not decls:
        return report

    def find(rule, sev, msg, where="", hint=""):
        report.add((rule, where, msg), Finding(rule, sev, msg, where, hint))

    dwords_of: Dict[str, set] = {}
    for _, secs in _programs(sched):
        for sec in secs:
            for op in sec.ops:
                for name, d in op.out_dwords.items():
                    dwords_of.setdefault(name, set()).add(d)

    parsed: Dict[PinDecl, PinSpec] = {}
    for d in decls:
        try:
            parsed[d] = PinSpec.parse(d.spec)
        except ValueError as exc:
            find("PIN.SYNTAX", "error", str(exc), d.where)

    carried_specs: Dict[str, set] = {}
    for d, spec in parsed.items():
        reg, sec = spec.reg, d.sec
        if d.scope == "carried":
            if sec.loop is None:
                find("PIN.UNKNOWN_VALUE", "error", f"section {sec.name} is not a loop; it carries nothing", d.where)
            elif d.value not in sec.carried:
                find(
                    "PIN.UNKNOWN_VALUE",
                    "error",
                    f"{d.value!r} is not carried by loop {sec.name} (carried: {list(sec.carried)})",
                    d.where,
                )
            carried_specs.setdefault(d.value, set()).add(str(reg))
            known = dwords_of.get(d.value, set())
        else:
            op = sec.ops[d.op_index]
            if d.value not in op.outs:
                find("PIN.UNKNOWN_VALUE", "error", f"{op} does not define {d.value!r} (outs: {list(op.outs)})", d.where)
            known = {op.out_dwords[d.value]} if d.value in op.out_dwords else set()

        # width
        if not known:
            find(
                "PIN.WIDTH_UNKNOWN",
                "warn",
                f"width of {d.value!r} is not declared (out_dwords); the lowering checks it against the real type",
                d.where,
            )
        for w in sorted(known):
            if w != reg.dwords:
                find("PIN.WIDTH", "error", f"{reg} is {reg.dwords} dwords but {d.value!r} is {w}", d.where)
        if reg.dwords not in WIDTHS[reg.file]:
            find(
                "PIN.SPLIT",
                "info",
                f"{reg} ({reg.dwords} dwords) has no single hint width ({WIDTHS[reg.file]} dwords); the lowering "
                "hints every leaf of the value to consecutive sub-ranges and fails if a leaf has another width "
                "(the backend would drop that hint silently)",
                d.where,
            )

        # register file
        if reg.file == "a" and not target.get("acc_vgpr_max_per_wave"):
            find(
                "PIN.NO_AGPR",
                "error",
                f"{target.key} has no AccVGPR file (acc_vgpr_max_per_wave is 0 or not described); {reg} cannot be "
                "pinned",
                d.where,
            )
        size = target.get(_FILE_FACT[reg.file])
        if size is None:
            if reg.file != "a":
                find(
                    "TARGET.UNKNOWN", "warn", f"{target.key}: {_FILE_FACT[reg.file]} unknown; cannot range-check {reg}"
                )
        elif reg.hi >= size:
            find(
                "PIN.OUT_OF_FILE",
                "error",
                f"{reg} is beyond the {size} {reg.file}-registers a wave can address on {target.key}",
                d.where,
            )
        align = target.get("vgpr_tuple_alignment") if reg.file in "va" else 2
        if reg.dwords > 1 and align and reg.lo % align:
            find(
                "PIN.ALIGN",
                "error" if spec.strict else "warn",
                f"{reg} is not {align}-aligned: no {reg.dwords}-dword register of {target.key} starts at "
                f"{reg.file}{reg.lo}, so the hint has no candidate"
                + (" (a strict hint then leaves the allocator nothing to choose)" if spec.strict else ""),
                d.where,
            )
        if reg.file == "v" and reg.lo == 0:
            find(
                "PIN.ABI",
                "warn",
                f"{reg} includes v0, which holds the work-item id at kernel entry (AMDGPUUsage, initial VGPR state)",
                d.where,
                "pin a range above the registers the kernel ABI initializes",
            )
        if d.scope == "carried":
            find(
                "PIN.CARRIED",
                "info",
                f"{reg} is hinted at the initial value, the loop-header phi (top of the body), the new value at the "
                "end of the body and the loop result; the hint survives coalescing of the phi "
                "(schedule-gpr-hint-coalescing.patch)",
                d.where,
            )

    for value, regs in carried_specs.items():
        if len(regs) > 1:
            find("PIN.MULTI", "error", f"carried value {value!r} pinned to different ranges {sorted(regs)}", value)

    # lifetimes, conflicts and overlap, per program
    by_place = {(id(d.sec), d.op_index, d.value): d for d in parsed}
    for _, secs in _programs(sched):
        pos, spans = _positions(secs)
        carried_pin = {v: parsed[d] for d in parsed if d.scope == "carried" for v in [d.value]}
        live: List[Tuple[int, int, PinDecl, PinSpec]] = []
        for si, sec in enumerate(secs):
            for value in getattr(sec, "pins", {}):
                d = by_place.get((id(sec), None, value))
                if d is not None:
                    live.append((*_lifetime(secs, si, None, value, pos, spans), d, parsed[d]))
            for oi, op in enumerate(sec.ops):
                for value in getattr(op, "pins", {}):
                    d = by_place.get((id(sec), oi, value))
                    if d is None:
                        continue
                    other = carried_pin.get(value)
                    if other is not None and other.reg != parsed[d].reg:
                        find(
                            "PIN.CONFLICT",
                            "error",
                            f"{value!r} is pinned to {parsed[d].reg} here but its loop pins it to {other.reg}",
                            d.where,
                        )
                    live.append((*_lifetime(secs, si, oi, value, pos, spans), d, parsed[d]))
        for i, (s1, e1, d1, p1) in enumerate(live):
            for s2, e2, d2, p2 in live[i + 1 :]:
                if d1.value == d2.value and p1.reg == p2.reg:
                    continue
                if p1.reg.overlaps(p2.reg) and s1 <= e2 and s2 <= e1:
                    both = p1.strict and p2.strict
                    find(
                        "PIN.OVERLAP",
                        "error" if both else "warn",
                        f"{d1.value}->{p1.reg} and {d2.value}->{p2.reg} share registers while both are live"
                        + ("; two strict hints cannot both hold" if both else "; at most one hint can hold"),
                        f"{d1.where} / {d2.where}",
                    )

    # budget at the requested occupancy
    wpe = sched.meta.get("waves_per_eu")
    file_lane = target.get("vgpr_file_per_lane")
    if wpe and file_lane:
        budget = min(target.get("arch_vgpr_max_per_wave") or file_lane, file_lane // int(wpe))
        top = max((p.reg.hi + 1 for p in parsed.values() if p.reg.file == "v"), default=0)
        if top > budget:
            find(
                "PIN.BUDGET",
                "warn",
                f"pins reach v{top - 1}: the kernel needs >= {top} arch VGPRs, above the {budget} that "
                f"waves_per_eu={wpe} allows",
            )

    # strict hints vs loads the compiler does not track
    untracked = {q for q, spec in sched.queues.items() if not spec.tracked}
    async_outs = sorted(
        {o for _, secs in _programs(sched) for s in secs for op in s.ops if op.queue in untracked for o in op.outs}
    )
    if async_outs and any(p.strict for p in parsed.values()):
        find(
            "PIN.ASYNC_SPILL",
            "warn",
            f"strict pins restrict the allocator, and {async_outs} arrive through queues the compiler does not "
            "track (explicit waits): if the kernel spills or copies, that code may read those registers before "
            "their wait",
            hint="check the ISA: no scratch, and pins.pending_lds_uses() empty",
        )

    # interactions with scheduling hints
    for _, secs in _programs(sched):
        for sec in secs:
            pinned_here = bool(getattr(sec, "pins", None)) or any(getattr(op, "pins", None) for op in sec.ops)
            if pinned_here and any(op.kind in ("sched_group", "issue") for op in sec.ops):
                find(
                    "PIN.SCHED",
                    "info",
                    "pins add no instructions, but copies into or out of a pinned range (PIN.INDIRECT) are VALU "
                    "moves that sched_group/issue VALU slots can absorb; check the ISA with isa_pin_report",
                    sec.name,
                )
    return report


def verify_pins(sched: Schedule, target: Optional[TargetDesc] = None):
    """Only the PIN.* rules, as a ``verify.Report``."""
    return check(sched, target)


# ---------------------------------------------------------------------------------------------
# Lowering (inside a @flyc.kernel; MLIR imported lazily)
# ---------------------------------------------------------------------------------------------


def _to_ir(value):
    from flydsl._mlir import ir

    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return value.ir_value()
    return value


def type_dwords(ty) -> int:
    """Per-lane dwords of an MLIR scalar or vector type."""
    from flydsl._mlir import ir

    def cast(cls, t):
        try:
            return cls(t)
        except (ValueError, TypeError):
            return None

    n, elem = 1, ty
    vt = cast(ir.VectorType, ty)
    if vt is not None:
        for d in vt.shape:
            n *= d
        elem = vt.element_type
    it = cast(ir.IntegerType, elem)
    bits = it.width if it is not None else ir.FloatType(elem).width
    total = n * bits
    if total % 32:
        raise ValueError(f"cannot pin {ty}: {total} bits is not a whole number of registers")
    return total // 32


def support_error() -> str:
    """Why pins cannot be lowered in this process ("" when they can)."""
    try:
        from flydsl.expr import rocdl
    except ImportError as exc:
        return f"FlyDSL has no ROCDL backend ({exc})"
    missing = [name for name in _OPS.values() if not hasattr(rocdl, name)]
    if missing:
        return f"flydsl.expr.rocdl lacks {missing} (FlyDSL without ROCm/FlyDSL#1307)"
    return ""


def require_support():
    reason = support_error()
    if reason:
        raise RuntimeError(f"register pins are not available: {reason}")


def _rewrap(original, raw):
    """A DSL numeric (``fx.Float32``, ...) comes back as the same DSL type, anything else raw."""
    if raw is original or not hasattr(original, "ir_value"):
        return raw
    try:
        return type(original)(raw)
    except Exception:  # wrappers that need more than the value (vectors with a shape) stay raw
        return raw


def hint(value, reg: RegRange, strict: bool = True):
    """``rocdl.sched.{vgpr,agpr,sgpr}`` on one leaf value; returns the hinted value."""
    from flydsl.expr import rocdl

    v = _to_ir(value)
    n = type_dwords(v.type)
    if n != reg.dwords:
        raise ValueError(f"pin {reg}: value of type {v.type} is {n} dwords, the range {reg.dwords}")
    if n not in WIDTHS[reg.file]:
        raise ValueError(f"pin {reg}: {n} dwords is no hint width ({WIDTHS[reg.file]}); the backend would drop it")
    out = getattr(rocdl, _OPS[reg.file])(v, reg.lo, soft=not strict)
    return _rewrap(value, out)


def _leaves(value, out):
    if isinstance(value, (list, tuple)):
        return (type(value), [_leaves(v, out) for v in value])
    out.append(value)
    return None


def _rebuild(shape, it):
    if shape is None:
        return next(it)
    kind, parts = shape
    return kind(_rebuild(p, it) for p in parts)


def leaf_dwords(leaf) -> int:
    return type_dwords(_to_ir(leaf).type)


def pin_value(value, spec, *, mark=None, dwords=None):
    """Hint ``value`` (a leaf or nested lists of leaves) to ``spec``: leaves take consecutive sub-ranges.

    ``mark(leaf, sub_range, strict)`` emits one hint (default :func:`hint`, after
    :func:`require_support`) and ``dwords(leaf)`` sizes a leaf (default :func:`leaf_dwords`);
    tests replace both to run without MLIR.
    """
    spec = PinSpec.parse(spec)
    leaves: list = []
    shape = _leaves(value, leaves)
    if mark is None:
        require_support()
        mark = hint
    size = dwords or leaf_dwords
    out, off = [], 0
    for leaf in leaves:
        n = size(leaf)
        out.append(mark(leaf, spec.reg.sub(off, n), spec.strict))
        off += n
    if off != spec.reg.dwords:
        raise ValueError(f"pin {spec.reg}: the value has {off} dwords, the range {spec.reg.dwords}")
    return _rebuild(shape, iter(out))


def _identity(v):
    return v


def carried_pins(sched: Schedule) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, secs in _programs(sched):
        for sec in secs:
            for name, spec in getattr(sec, "pins", {}).items():
                if out.setdefault(name, spec) != spec:
                    raise ValueError(f"{sched.name}: carried {name!r} pinned to both {out[name]} and {spec}")
    return out


def pin_codecs(sched: Schedule, codecs: Mapping, **kw) -> Mapping:
    """``emit_program`` codecs with loop-carried pins applied (``codecs`` unchanged without pins).

    ``pack`` runs on the initial value and on the new value at the end of the body,
    ``unpack`` on the iteration argument at the top of the body and on the loop result:
    all four are hinted (the (packed) leaves are what the loop carries).
    """
    pinned = carried_pins(sched)
    if not pinned:
        return codecs
    out = dict(codecs)
    for name, spec in pinned.items():
        pack, unpack = codecs.get(name, (_identity, _identity))
        out[name] = (
            lambda v, p=pack, s=spec: pin_value(p(v), s, **kw),
            lambda v, u=unpack, s=spec: u(pin_value(v, s, **kw)),
        )
    return out


def pin_inputs(op: Op, env: Mapping, **kw) -> list:
    """Arguments of ``op``'s binding (a hint lives at the definition; reads need nothing)."""
    return [env[n] for n in op.ins]


def pin_outputs(op: Op, env, **kw) -> None:
    """After ``op``'s results are in ``env``: hint its pinned outputs."""
    for out, spec in (getattr(op, "pins", None) or {}).items():
        if out in op.outs:
            env[out] = pin_value(env[out], spec, **kw)


# ---------------------------------------------------------------------------------------------
# ISA placement report (pure text; FLYDSL_DUMP_IR=1 writes *_final_isa.s)
# ---------------------------------------------------------------------------------------------

_REG_TOKEN = re.compile(r"\b([vas])(?:(\d+)|\[(\d+):(\d+)\])")
_COPY_OPS = (
    "v_mov_b32",
    "v_mov_b64",
    "v_pk_mov_b32",
    "v_accvgpr_write_b32",
    "v_accvgpr_read_b32",
    "v_accvgpr_mov_b32",
    "s_mov_b32",
    "s_mov_b64",
)
_NO_DST_PREFIX = ("ds_write", "ds_store", "buffer_store", "global_store", "flat_store", "scratch_store", "s_waitcnt")


def _regs(operand: str) -> List[Tuple[str, int, int]]:
    out = []
    for m in _REG_TOKEN.finditer(operand):
        f, one, lo, hi = m.groups()
        out.append((f, int(one), int(one)) if one is not None else (f, int(lo), int(hi)))
    return out


def _instructions(asm_text: str, kernel: Optional[str] = None):
    """(block, mnemonic, operands, line) of every instruction (of ``kernel``, if given)."""
    block, inside = "entry", kernel is None
    for raw in asm_text.splitlines():
        s = raw.split(";", 1)[0].strip()
        if not s:
            continue
        if s.endswith(":"):  # function or basic-block label (.LBB0_3)
            label = s[:-1]
            if kernel is not None and label == kernel:
                inside = True
            elif kernel is not None and label.startswith(".Lfunc_end"):
                inside = False
            block = label
            continue
        if not inside or s.startswith("."):
            continue
        parts = s.split(None, 1)
        mnem = re.sub(r"_(e32|e64|sdwa|dpp)$", "", parts[0])
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        yield block, mnem, ops, s


@dataclass
class RangeReport:
    reg: str
    writers: Counter = field(default_factory=Counter)  # mnemonic -> instructions defining part of the range
    readers: Counter = field(default_factory=Counter)
    # mnemonic -> instructions reading and writing the range (an MFMA accumulating in place)
    in_place: Counter = field(default_factory=Counter)
    copies_in: List[str] = field(default_factory=list)  # "block: instruction"
    copies_out: List[str] = field(default_factory=list)

    @property
    def placed(self) -> bool:
        return bool(self.writers or self.readers)

    def summary(self) -> dict:
        return {
            "reg": self.reg,
            "placed": self.placed,
            "writers": dict(self.writers),
            "readers": dict(self.readers),
            "in_place": dict(self.in_place),
            "copies_in": len(self.copies_in),
            "copies_out": len(self.copies_out),
            "copy_blocks": dict(Counter(c.split(":", 1)[0] for c in self.copies_in + self.copies_out)),
        }


def isa_pin_report(asm_text: str, regs: Sequence[str], kernel: Optional[str] = None) -> Dict[str, RangeReport]:
    """Who writes / reads each pinned range on the ISA, and the copies into / out of it.

    ``copies_in``: a register copy whose destination is in the range and whose source is
    not (the producer did not write the range directly); ``copies_out``: the reverse (a
    consumer read a copy).  Both are ``PIN.INDIRECT``.  The first operand of an
    instruction is taken as its destination (stores and waits have none).
    """
    ranges = {r: PinSpec.parse(r).reg for r in regs}
    reports = {r: RangeReport(str(ranges[r])) for r in regs}
    for block, mnem, ops, line in _instructions(asm_text, kernel):
        if not ops:
            continue
        has_dst = not mnem.startswith(_NO_DST_PREFIX)
        dst = _regs(ops[0]) if has_dst else []
        srcs = [r for o in (ops[1:] if has_dst else ops) for r in _regs(o)]
        for name, rng in ranges.items():

            def hits(regs, rng=rng):
                return any(f == rng.file and lo <= rng.hi and rng.lo <= hi for f, lo, hi in regs)

            rep = reports[name]
            w, r = hits(dst), hits(srcs)
            if w:
                rep.writers[mnem] += 1
            if r:
                rep.readers[mnem] += 1
            if w and r:
                rep.in_place[mnem] += 1
            if mnem in _COPY_OPS and srcs:  # a copy from a register (an immediate is an initialization)
                if w and not r:
                    rep.copies_in.append(f"{block}: {line}")
                elif r and not w:
                    rep.copies_out.append(f"{block}: {line}")
    return reports


def mfma_accumulators(asm_text: str, kernel: Optional[str] = None) -> Counter:
    """Destination register of every MFMA (``"v[128:131]"`` -> count): where accumulators actually live."""
    out: Counter = Counter()
    for _, mnem, ops, _ in _instructions(asm_text, kernel):
        if mnem.startswith("v_mfma") and ops:
            out[ops[0]] += 1
    return out


_STORE_PREFIX = ("scratch_store", "buffer_store", "global_store", "flat_store", "ds_write")


def pending_lds_uses(asm_text: str) -> List[str]:
    """Instructions that touch the destination of an LDS read before ``s_waitcnt lgkmcnt`` drains it.

    LLVM inserts these waits itself for the loads it emits, not for loads inside inline
    asm (FlyDSL's ``ds_read_b64_tr_b16`` helpers), whose results the schedule drains with
    explicit waits.  A copy or spill store the register allocator places between such a
    load and its wait reads the register before the data arrives -- what strict pins can
    cause by restricting the allocator.  Heuristic: straight-line program order, ``lgkm``
    as one FIFO of DS / scalar loads, control flow ignored.
    """
    pending: List[set] = []
    hits = []

    def regset(operands):
        return {(f, i) for o in operands for f, lo, hi in _regs(o) if f in "va" for i in range(lo, hi + 1)}

    for raw in asm_text.splitlines():
        s = raw.split(";", 1)[0].strip()
        if not s or s.startswith(".") or s.endswith(":"):
            continue
        parts = s.split(None, 1)
        mnem = parts[0]
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        if mnem.startswith("s_waitcnt"):
            m = re.search(r"lgkmcnt\((\d+)\)", s)
            if m:
                n = int(m.group(1))
                pending = pending[len(pending) - n :] if n else []
            continue
        live = set().union(*pending) if pending else set()
        if mnem.startswith(("ds_read", "ds_write", "s_load", "s_buffer_load")):
            srcs = regset(ops[1:] if mnem.startswith("ds_read") else ops)
            if srcs & live:
                hits.append(s)
            pending.append(regset(ops[:1]) if mnem.startswith("ds_read") else set())
            continue
        if ops and regset(ops) & live:
            hits.append(s)
    return hits
