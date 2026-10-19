# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Named-register pins at the LLVM IR / ISA level (opt-in; needs a pin-capable LLVM).

``flydsl.sched.pins`` lowers a pin to an inline-asm *marker* that any LLVM
compiles as a no-op::

    %7 = call <16 x float> asm ";flydsl.pin.write v[192:207]", "=v,0"(<16 x float> %6)

:func:`rewrite_markers` turns the markers of a module's LLVM IR (the
``NN_llvm_ir.ll`` that ``FLYDSL_DUMP_IR=1`` writes, i.e. the input of
``gpu-module-to-binary``) into the intrinsics of the AMDGPU named-register branch
(``yuyzhang512/llvm-project`` ``amdgpu-named-reg-pin-std``)::

    %pin_7_i = bitcast <16 x float> %6 to <16 x i32>
    call void @llvm.write_register.v16i32(metadata !9, <16 x i32> %pin_7_i)
    %7 = bitcast <16 x float> %6 to <16 x float>
    ...
    %pin_9_r = call <16 x i32> @llvm.read_register.v16i32(metadata !9)
    %9 = bitcast <16 x i32> %pin_9_r to <16 x float>
    !9 = !{!"v[192:207]"}

A write marker's result stays the written value (so a loop's phi of it becomes
dead once every read goes through the register); a read marker ignores its
operand.  ``mode="strip"`` removes the markers without pinning (the same IR
minus the pins, for a fair comparison).

:func:`isa_pin_report` checks the final ISA: which instructions write and read
each pinned range, and how many register copies move values into or out of it
(``PIN.INDIRECT``).

Pure Python (stdlib only) so it can be used on a machine without ROCm;
``tools/named_reg/named_reg.py`` drives ``opt``/``llc``/``ld.lld`` and the JIT
cache injection.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

MARKER = ";flydsl.pin"

_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<res>%[-\w.$]+) = (?:tail |notail |musttail )?call (?P<ty>.+?) asm "
    r'(?:sideeffect )?(?:alignstack )?"' + re.escape(MARKER) + r'\.(?P<mode>read|write) (?P<reg>[^"]+)", '
    r'"=v,0"\((?P<rest>.*)$'
)
_DBG_RE = re.compile(r"!dbg (![0-9]+)")
_MD_ID_RE = re.compile(r"^!(\d+) = ", re.M)
_VEC_RE = re.compile(r"^<(\d+) x (\w+)>$")
_ELEM_BITS = {"i8": 8, "i16": 16, "half": 16, "bfloat": 16, "i32": 32, "float": 32, "i64": 64, "double": 64}


def ir_type_dwords(ty: str) -> int:
    """Dwords of an LLVM IR first-class type (``<16 x float>``, ``i32``, ``<8 x bfloat>``)."""
    ty = ty.strip()
    m = _VEC_RE.match(ty)
    n, elem = (int(m.group(1)), m.group(2)) if m else (1, ty)
    if elem not in _ELEM_BITS:
        raise ValueError(f"cannot pin a value of type {ty}")
    bits = n * _ELEM_BITS[elem]
    if bits % 32:
        raise ValueError(f"cannot pin {ty}: {bits} bits is not a whole number of registers")
    return bits // 32


def _int_type(n: int) -> Tuple[str, str]:
    return ("i32", "i32") if n == 1 else (f"<{n} x i32>", f"v{n}i32")


def _split_operand(rest: str, ty: str) -> Tuple[str, str]:
    """``'<ty> %x) , !dbg !5'`` -> (``'%x'``, ``', !dbg !5'``)."""
    head = ty + " "
    if not rest.startswith(head):
        raise ValueError(f"marker operand does not have the result type {ty}: {rest!r}")
    body = rest[len(head) :]
    depth = 0
    for i, ch in enumerate(body):
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            if depth == 0 and ch == ")":
                return body[:i].strip(), body[i + 1 :]
            depth -= 1
    raise ValueError(f"unterminated marker call: {rest!r}")


def _local(res: str) -> str:
    return re.sub(r"[^\w]", "_", res[1:])


@dataclass
class RewriteResult:
    text: str
    reads: int = 0
    writes: int = 0
    regs: Dict[str, int] = field(default_factory=dict)  # reg -> dwords
    markers: List[Tuple[str, str, str]] = field(default_factory=list)  # (mode, reg, type)


def rewrite_markers(ll_text: str, mode: str = "pin") -> RewriteResult:
    """Rewrite pin markers: ``mode="pin"`` -> named-register intrinsics; ``"strip"`` -> plain values."""
    if mode not in ("pin", "strip"):
        raise ValueError(f"mode must be 'pin' or 'strip', got {mode!r}")
    ids = [int(x) for x in _MD_ID_RE.findall(ll_text)]
    next_id = max(ids, default=-1) + 1
    md: Dict[str, int] = {}
    decls: Dict[str, str] = {}
    out = RewriteResult("")
    lines = []
    for line in ll_text.splitlines():
        m = _CALL_RE.match(line)
        if m is None:
            lines.append(line)
            continue
        ind, res, ty, kind, reg = (
            m.group("indent"),
            m.group("res"),
            m.group("ty").strip(),
            m.group("mode"),
            m.group("reg"),
        )
        operand, tail = _split_operand(m.group("rest"), ty)
        dbg = _DBG_RE.search(tail)
        dbg_s = f", !dbg {dbg.group(1)}" if dbg else ""
        n = ir_type_dwords(ty)
        out.markers.append((kind, reg, ty))
        if out.regs.setdefault(reg, n) != n:
            raise ValueError(f"{reg} used with {out.regs[reg]} and {n} dwords")
        if mode == "strip":
            lines.append(f"{ind}{res} = bitcast {ty} {operand} to {ty}{dbg_s}")
            continue
        it, suf = _int_type(n)
        if reg not in md:
            md[reg] = next_id
            next_id += 1
        mdref = f"!{md[reg]}"
        tmp = _local(res)
        if kind == "write":
            out.writes += 1
            src = operand
            if ty != it:
                lines.append(f"{ind}%pin_{tmp}_i = bitcast {ty} {operand} to {it}{dbg_s}")
                src = f"%pin_{tmp}_i"
            lines.append(f"{ind}call void @llvm.write_register.{suf}(metadata {mdref}, {it} {src}){dbg_s}")
            lines.append(f"{ind}{res} = bitcast {ty} {operand} to {ty}{dbg_s}")
            decls[f"llvm.write_register.{suf}"] = f"declare void @llvm.write_register.{suf}(metadata, {it})"
        else:
            out.reads += 1
            lines.append(f"{ind}%pin_{tmp}_r = call {it} @llvm.read_register.{suf}(metadata {mdref}){dbg_s}")
            lines.append(f"{ind}{res} = bitcast {it} %pin_{tmp}_r to {ty}{dbg_s}")
            decls[f"llvm.read_register.{suf}"] = f"declare {it} @llvm.read_register.{suf}(metadata)"
    for name, decl in decls.items():
        if f"@{name}(" not in ll_text:
            lines.append(decl)
    for reg, i in md.items():
        lines.append(f'!{i} = !{{!"{reg}"}}')
    out.text = "\n".join(lines) + "\n"
    return out


# ---------------------------------------------------------------------------------------------
# ISA placement check
# ---------------------------------------------------------------------------------------------

_REG_TOKEN = re.compile(r"\b([va])(?:(\d+)|\[(\d+):(\d+)\])")
_COPY_OPS = ("v_mov_b32", "v_mov_b64", "v_pk_mov_b32", "v_accvgpr_write_b32", "v_accvgpr_read_b32", "v_accvgpr_mov_b32")
_NO_VDST_PREFIX = ("ds_write", "ds_store", "buffer_store", "global_store", "flat_store", "scratch_store", "s_")


def _parse_range(text: str) -> Tuple[str, int, int]:
    m = re.fullmatch(r"([va])(?:(\d+)|\[(\d+):(\d+)\])", text.strip())
    if m is None:
        raise ValueError(f"bad register range {text!r}")
    f, one, lo, hi = m.groups()
    return (f, int(one), int(one)) if one is not None else (f, int(lo), int(hi))


def _regs(operand: str) -> List[Tuple[str, int, int]]:
    out = []
    for m in _REG_TOKEN.finditer(operand):
        f, one, lo, hi = m.groups()
        out.append((f, int(one), int(one)) if one is not None else (f, int(lo), int(hi)))
    return out


def _hits(regs, rng) -> bool:
    f, lo, hi = rng
    return any(r[0] == f and r[1] <= hi and lo <= r[2] for r in regs)


@dataclass
class RangeReport:
    reg: str
    writers: Counter = field(default_factory=Counter)  # mnemonic -> count (instructions defining part of the range)
    readers: Counter = field(default_factory=Counter)
    copies_in: List[str] = field(default_factory=list)  # "block: instruction"
    copies_out: List[str] = field(default_factory=list)
    # mnemonic -> count of instructions reading and writing the range (an MFMA accumulating in place)
    in_place: Counter = field(default_factory=Counter)

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
    """Who writes / reads each pinned range in the ISA, and the copies into / out of it.

    ``copies_in``: a register copy whose destination is in the range and whose source is
    not (the producer did not write the range directly); ``copies_out``: the reverse
    (a consumer read a copy).  Both are ``PIN.INDIRECT``.  Only the first operand of an
    instruction is taken as its destination (stores and scalar ops have none).
    """
    ranges = {r: _parse_range(r) for r in regs}
    reports = {r: RangeReport(r) for r in regs}
    block, in_kernel = "entry", kernel is None
    for raw in asm_text.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        s = line.strip()
        if s.endswith(":"):  # function or basic-block label (.LBB0_3)
            label = s[:-1]
            if kernel is not None and label == kernel:
                in_kernel = True
            elif kernel is not None and label.startswith(".Lfunc_end"):
                in_kernel = False
            block = label
            continue
        if not in_kernel or s.startswith("."):
            continue
        parts = s.split(None, 1)
        mnem = re.sub(r"_(e32|e64|sdwa|dpp)$", "", parts[0])
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        if not ops:
            continue
        has_dst = not mnem.startswith(_NO_VDST_PREFIX)
        dst = _regs(ops[0]) if has_dst else []
        srcs = [r for o in (ops[1:] if has_dst else ops) for r in _regs(o)]
        for name, rng in ranges.items():
            rep = reports[name]
            w, r = _hits(dst, rng), _hits(srcs, rng)
            if w:
                rep.writers[mnem] += 1
            if r:
                rep.readers[mnem] += 1
            if w and r:
                rep.in_place[mnem] += 1
            if mnem in _COPY_OPS and srcs:  # a copy from a register (an immediate is an initialization)
                if w and not r:
                    rep.copies_in.append(f"{block}: {s}")
                elif r and not w:
                    rep.copies_out.append(f"{block}: {s}")
    return reports


_STORE_PREFIX = ("scratch_store", "buffer_store", "global_store", "flat_store", "ds_write")


def pending_lds_uses(asm_text: str) -> List[str]:
    """Instructions that touch the destination of an LDS read before ``s_waitcnt lgkmcnt`` drains it.

    LLVM inserts these waits itself for the loads it emits; it does not for loads inside
    inline asm (FlyDSL's ``ds_read_b64_tr_b16`` helpers), whose results the schedule
    drains with explicit waits.  A copy or spill store the register allocator places
    between such a load and its wait reads the register before the data arrives.  Pins
    take registers away from the allocator, so this is the failure to look for when a
    pinned kernel spills.  Heuristic: straight-line program order, ``lgkm`` as one FIFO
    of DS / scalar loads, control flow ignored.
    """
    pending: List[set] = []
    hits = []
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
        live = {(f, i) for regs in pending for f, i in regs}

        def regset(operands):
            return {(f, i) for o in operands for f, lo, hi in _regs(o) for i in range(lo, hi + 1)}

        if mnem.startswith(("ds_read", "ds_write", "s_load", "s_buffer_load")):
            srcs = regset(ops[1:] if mnem.startswith("ds_read") else ops)
            if srcs & live:
                hits.append(s)
            pending.append(regset(ops[:1]) if mnem.startswith("ds_read") else set())
            continue
        touched = regset(ops)
        if not mnem.startswith(_STORE_PREFIX) and not ops:
            continue
        if touched & live:
            hits.append(s)
    return hits
