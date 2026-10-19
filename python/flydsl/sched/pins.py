# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register pins: "this value lives in v[a:b]" as Schedule IR data.

A pin names a physical register range for a value:

* ``Op.pins = {out: "v[64:79]"}`` -- this definition is written to the range and
  every later read of the value (until it is redefined) reads the range;
* ``Section.pins = {carried: "v[192:255]"}`` -- a loop-carried value lives in the
  range across the whole loop: it is written before the loop and at the end of
  every iteration, read at the top of every iteration and after the loop.

Semantics follow the AMDGPU named-register branch
(``yuyzhang512/llvm-project`` ``amdgpu-named-reg-pin-std``): a pinned value has
no storage of its own; every read and write becomes ``llvm.read_register`` /
``llvm.write_register`` with the range as an ``MDString`` name.  The branch

* accepts only names whose width equals the value's width in dwords, and only
  widths of a register class (``WIDTHS``);
* rejects ranges beyond the addressable VGPR file of the subtarget;
* reserves a range for the whole function when it is read in a block before any
  write in that block (a live-in read: always the case for loop-carried pins);
  other ranges are protected by liveness only;
* selects the AGPR form of an MFMA when its operands are pinned to ``a[...]``.

The guarantee is placement at the read/write points.  Whether the producer
(an MFMA, a ``ds_read``) writes the range directly or through a ``v_mov`` copy
depends on register coalescing; ``flydsl.compiler.named_reg.isa_pin_report``
counts such copies on the final ISA (``PIN.INDIRECT``).

Lowering is opt-in and needs no compiler rebuild: :func:`pin_value` emits an
inline-asm *marker* (an asm comment with a tied ``"=v,0"`` operand) that the
stock toolchain compiles as a no-op; ``flydsl.compiler.named_reg`` rewrites the
markers in the dumped LLVM IR into the intrinsics, and ``tools/named_reg``
compiles that with the branch's ``llc`` and runs it through the JIT cache.

This module is pure Python (MLIR is imported lazily by the lowering helpers).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import Op, Schedule, Section
from .target import TargetDesc, load_target

BRANCH = "yuyzhang512/llvm-project amdgpu-named-reg-pin-std @583640bc5"
# Widths (dwords) SITargetLowering::getRegisterByName maps to a register on the branch:
# 32-bit, or the VReg_64/96/128/256/512 resp. AReg_64/128/256/512 tuple classes.
WIDTHS = {"v": (1, 2, 3, 4, 8, 16), "a": (1, 2, 4, 8, 16)}
_FILE_FACT = {"v": "arch_vgpr_max_per_wave", "a": "acc_vgpr_max_per_wave"}

MARKER = ";flydsl.pin"  # inline-asm text: "<MARKER>.<mode> <reg>"
MODES = ("read", "write")
LIVE_KEY = "__pins__"  # emit env key: value name -> range of its current pinned definition

_REG_RE = re.compile(r"^([va])(?:(\d+)|\[(\d+):(\d+)\])$")


@dataclass(frozen=True)
class RegRange:
    """``v[lo:hi]`` / ``a[lo:hi]`` (inclusive), or one register ``v5``."""

    file: str
    lo: int
    hi: int

    @classmethod
    def parse(cls, text: str) -> "RegRange":
        m = _REG_RE.match(str(text).strip())
        if m is None:
            raise ValueError(f"bad register range {text!r}: expected 'v12', 'v[64:79]' or 'a[0:15]'")
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


def marker_text(mode: str, reg: RegRange) -> str:
    assert mode in MODES
    return f"{MARKER}.{mode} {reg}"


# ---------------------------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------------------------


@dataclass(eq=False)
class PinDecl:
    value: str
    reg: str  # as written
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
        return f"{g}{self.section}{at} {self.value}->{self.reg}"


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
            for value, reg in getattr(sec, "pins", {}).items():
                key = (id(sec), None, value)
                if key not in seen:
                    seen.add(key)
                    out.append(PinDecl(value, reg, "carried", sec, None, g))
            for i, op in enumerate(sec.ops):
                for value, reg in getattr(op, "pins", {}).items():
                    key = (id(sec), i, value)
                    if key not in seen:
                        seen.add(key)
                        out.append(PinDecl(value, reg, "op", sec, i, g))
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


def _intervals(secs: Sequence[Section]):
    """Linear positions of ``secs`` and the (start, end) span of every loop section."""
    pos, spans, p = {}, {}, 0
    for si, sec in enumerate(secs):
        start = p
        for oi, _ in enumerate(sec.ops):
            pos[(si, oi)] = p
            p += 1
        spans[si] = (start, max(p - 1, start))
    return pos, spans, p


def _op_pin_lifetime(secs, si, oi, value):
    """[first, last] linear position where an op-pinned definition is live, and whether a read is live-in."""
    pos, spans, total = _intervals(secs)
    start = pos[(si, oi)]
    end = start
    live_in = False
    loops_crossed = set()
    # forward walk until the value is redefined
    order = [(sj, oj) for sj, sec in enumerate(secs) for oj in range(len(sec.ops))]
    for sj, oj in order:
        if pos[(sj, oj)] <= start:
            continue
        op = secs[sj].ops[oj]
        if value in op.ins:
            end = pos[(sj, oj)]
            if sj != si:
                live_in = True
                if secs[sj].loop is not None:
                    loops_crossed.add(sj)
        if value in op.outs:
            break
    # a definition outside a loop that is read inside it is live around the whole loop
    for sj in loops_crossed:
        end = max(end, spans[sj][1])
    return start, end, live_in


def check(sched: Schedule, target: Optional[TargetDesc] = None, report=None):
    """Add ``PIN.*`` findings for ``sched`` to ``report`` (a ``verify.Report``); returns the report."""
    from .verify import Finding, Report  # lazy: verify may call this module

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

    parsed: Dict[PinDecl, RegRange] = {}
    for d in decls:
        try:
            parsed[d] = RegRange.parse(d.reg)
        except ValueError as exc:
            find("PIN.SYNTAX", "error", str(exc), d.where)

    carried_regs: Dict[str, set] = {}
    for d, reg in parsed.items():
        sec = d.sec
        # the value exists where the pin says
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
            carried_regs.setdefault(d.value, set()).add(str(reg))
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
                find(
                    "PIN.WIDTH",
                    "error",
                    f"{reg} is {reg.dwords} dwords but {d.value!r} is {w}; the branch rejects a name whose width "
                    "differs from the value's",
                    d.where,
                )
        if reg.dwords not in WIDTHS[reg.file]:
            find(
                "PIN.SPLIT",
                "info",
                f"{reg} ({reg.dwords} dwords) is no single {reg.file}-register class (widths {WIDTHS[reg.file]}); "
                "the lowering pins every leaf of the value to consecutive sub-ranges and fails if a leaf is not one",
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
        if size is None and reg.file == "v":
            find("TARGET.UNKNOWN", "warn", f"{target.key}: {_FILE_FACT['v']} unknown; cannot range-check {reg}")
        elif size is not None and reg.hi >= size:
            find(
                "PIN.OUT_OF_FILE",
                "error",
                f"{reg} is beyond the {size} {reg.file}-registers a wave can address on {target.key}",
                d.where,
            )
        align = target.get("vgpr_tuple_alignment")
        if reg.dwords > 1 and align and reg.lo % align:
            find(
                "PIN.ALIGN",
                "warn",
                f"{reg} is not {align}-aligned; {target.key} needs aligned register tuples, so multi-dword "
                "accesses go through copies",
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
                "PIN.RESERVED",
                "info",
                f"{reg} is read at the top of every iteration before it is written there, so the branch reserves it "
                f"for the whole function ({reg.dwords} registers fewer for everything else)",
                d.where,
            )

    for value, regs in carried_regs.items():
        if len(regs) > 1:
            find("PIN.MULTI", "error", f"carried value {value!r} pinned to different ranges {sorted(regs)}", value)

    # lifetimes and overlap, per program
    by_place = {(id(d.sec), d.op_index, d.value): d for d in parsed}
    for _, secs in _programs(sched):
        live: List[Tuple[int, int, PinDecl, RegRange]] = []
        carried_here = {n for s in secs if s.loop is not None for n in s.carried}
        for si, sec in enumerate(secs):
            for value in getattr(sec, "pins", {}):
                d = by_place.get((id(sec), None, value))
                if d is not None:
                    live.append((-1, 1 << 30, d, parsed[d]))  # reserved: the whole function
            for oi, op in enumerate(sec.ops):
                for value in getattr(op, "pins", {}):
                    d = by_place.get((id(sec), oi, value))
                    if d is None:
                        continue
                    if value in carried_here:
                        find(
                            "PIN.CARRIED_OP_PIN",
                            "error",
                            f"{value!r} is loop-carried; pin it on the loop (Section.pins) so the iter_arg and every "
                            "iteration share the range",
                            d.where,
                        )
                        continue
                    s, e, live_in = _op_pin_lifetime(secs, si, oi, value)
                    if live_in:
                        find(
                            "PIN.RESERVED",
                            "info",
                            f"{parsed[d]} is read in another section than it is written; that read is live-in, so "
                            "the branch reserves the range for the whole function",
                            d.where,
                        )
                        s, e = -1, 1 << 30
                    live.append((s, e, d, parsed[d]))
        for i, (s1, e1, d1, r1) in enumerate(live):
            for s2, e2, d2, r2 in live[i + 1 :]:
                if d1.value == d2.value and r1 == r2:
                    continue
                if r1.overlaps(r2) and s1 <= e2 and s2 <= e1:
                    find(
                        "PIN.OVERLAP",
                        "error",
                        f"{d1.value}->{r1} and {d2.value}->{r2} share registers while both are live",
                        f"{d1.where} / {d2.where}",
                    )

    # budget at the requested occupancy
    wpe = sched.meta.get("waves_per_eu")
    file_lane = target.get("vgpr_file_per_lane")
    if wpe and file_lane:
        budget = min(target.get("arch_vgpr_max_per_wave") or file_lane, file_lane // int(wpe))
        top = max((r.hi + 1 for r in parsed.values() if r.file == "v"), default=0)
        if top > budget:
            find(
                "PIN.BUDGET",
                "warn",
                f"pins reach v{top - 1}: the kernel needs >= {top} arch VGPRs, above the {budget} that "
                f"waves_per_eu={wpe} allows",
            )

    # register pressure vs loads the compiler does not track
    untracked = {q for q, spec in sched.queues.items() if not spec.tracked}
    async_outs = sorted(
        {o for _, secs in _programs(sched) for s in secs for op in s.ops if op.queue in untracked for o in op.outs}
    )
    if async_outs:
        n = sum(r.dwords for r in parsed.values())
        find(
            "PIN.ASYNC_SPILL",
            "warn",
            f"pins take up to {n} registers from the allocator, and {async_outs} arrive through queues the compiler "
            "does not track (explicit waits): if the kernel now spills, spill/copy code may read those registers "
            "before their wait",
            hint="check ScratchSize == 0 and named_reg.pending_lds_uses on the ISA",
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


def marker(value, reg: RegRange, mode: str):
    """One marker on one leaf value; returns the value to use from here on.

    A DSL numeric (``fx.Float32``, ...) comes back as the same DSL type; anything
    else as the raw ``ir.Value``.
    """
    from flydsl._mlir.dialects import llvm

    v = _to_ir(value)
    n = type_dwords(v.type)
    if n != reg.dwords:
        raise ValueError(f"pin {reg}: value of type {v.type} is {n} dwords, the range {reg.dwords}")
    if n not in WIDTHS[reg.file]:
        raise ValueError(f"pin {reg}: {n} dwords is no {reg.file}-register class on the branch ({WIDTHS[reg.file]})")
    out = llvm.inline_asm(v.type, [v], marker_text(mode, reg), "=v,0", has_side_effects=False)
    if v is not value:
        try:
            return type(value)(out)
        except Exception:  # wrappers that need more than the value (vectors with a shape) stay raw
            return out
    return out


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


def pin_value(value, reg, mode: str, *, mark=None, dwords=None):
    """Pin ``value`` (a leaf or nested lists of leaves) to ``reg``: leaves take consecutive sub-ranges.

    ``mark(leaf, sub_range, mode)`` emits one marker (default :func:`marker`) and
    ``dwords(leaf)`` sizes a leaf (default :func:`leaf_dwords`); both are looked up at
    call time so tests can replace them without MLIR.
    """
    reg = reg if isinstance(reg, RegRange) else RegRange.parse(reg)
    leaves: list = []
    shape = _leaves(value, leaves)
    mark = mark or marker
    size = dwords or leaf_dwords
    out, off = [], 0
    for leaf in leaves:
        n = size(leaf)
        out.append(mark(leaf, reg.sub(off, n), mode))
        off += n
    if off != reg.dwords:
        raise ValueError(f"pin {reg}: the value has {off} dwords, the range {reg.dwords}")
    return _rebuild(shape, iter(out))


def _identity(v):
    return v


def carried_pins(sched: Schedule) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, secs in _programs(sched):
        for sec in secs:
            for name, reg in getattr(sec, "pins", {}).items():
                if out.setdefault(name, reg) != reg:
                    raise ValueError(f"{sched.name}: carried {name!r} pinned to both {out[name]} and {reg}")
    return out


def pin_codecs(sched: Schedule, codecs: Mapping, **kw) -> Mapping:
    """``emit_program`` codecs with loop-carried pins applied (``codecs`` unchanged without pins).

    ``pack`` runs on the initial value before the loop and on the new value at the end
    of the body: the pinned (packed) leaves are written there.  ``unpack`` runs at the
    top of the body and on the loop results: the leaves are read there.
    """
    pinned = carried_pins(sched)
    if not pinned:
        return codecs
    out = dict(codecs)
    for name, reg in pinned.items():
        pack, unpack = codecs.get(name, (_identity, _identity))
        out[name] = (
            lambda v, p=pack, r=reg: pin_value(p(v), r, "write", **kw),
            lambda v, u=unpack, r=reg: u(pin_value(v, r, "read", **kw)),
        )
    return out


def pin_inputs(op: Op, env: Mapping, **kw) -> list:
    """Arguments of ``op``'s binding; values with a live op pin are read from their range."""
    live = env.get(LIVE_KEY) or {}
    return [pin_value(env[n], live[n], "read", **kw) if n in live else env[n] for n in op.ins]


def pin_outputs(op: Op, env, **kw) -> None:
    """After ``op``'s results are in ``env``: write pinned outs, end pins of redefined values."""
    live = env.get(LIVE_KEY) or {}
    pins = getattr(op, "pins", None) or {}
    if not pins and not any(o in live for o in op.outs):
        return
    live = dict(live)
    for o in op.outs:
        if o in pins:
            env[o] = pin_value(env[o], pins[o], "write", **kw)
            live[o] = pins[o]
        else:
            live.pop(o, None)
    env[LIVE_KEY] = live
