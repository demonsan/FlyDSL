# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ISA conformance: does the final assembly follow the schedule it was emitted from?

``lower.emit(..., tag=<name>)`` gives every MLIR op built for schedule op ``i`` of
section ``s`` the location ``sched:<name>/<s>:<i + 1>``.  Compiled with
``FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`` that location ends up in the DWARF line
table, i.e. in the ``.file`` / ``.loc`` directives of ``*_final_isa.s``
(``FLYDSL_DUMP_IR=1``).  This module reads them back:

* ``parse_isa`` -- instructions of one kernel, each with its class, source
  location and (when tagged) its schedule section / op index;
* ``conform`` -- per schedule op, the instructions it produced, checked against
  the schedule (order across fences, waits, barriers, sched_group interleave,
  ``s_setprio`` placement, instruction counts, unattributed code);
* ``diff_isa`` -- a structural diff of two kernels, aligned per barrier region
  (e.g. a hand-written kernel against the schedule-emitted one).

Attribution model (documented so findings can be argued with): an instruction
belongs to the location of the closest preceding ``.loc``.  LLVM gives
instructions it creates without a location (most ``s_waitcnt`` it inserts, spill
code, ``s_nop`` hazard padding) the location of the previous instruction, and
merged / hoisted instructions may get line 0 or the location of a different
op.  A ``sched:`` location therefore says which schedule op an instruction was
*emitted for or placed next to*, not which op needed it.

Pure Python: no MLIR or ROCm needed.
"""

import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import CONTROL_KINDS, Op, Schedule
from .lower import TAG_SCHEME
from .verify import Finding, Report

# --------------------------------------------------------------------------- ISA model

TRANS_OPS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_")
MAX_WAIT = {"vm": 63, "lgkm": 15, "exp": 7}
# fence ops split a schedule section into regions ("clusters")
FENCE_KINDS = ("barrier", "sched_barrier", "stagger")
# schedule op kind -> instruction class its ``count`` refers to
COUNT_CLASS = {
    "mfma": "mfma",
    "lds_dma": "lds_dma",
    "ds_read": "ds_read",
    "ds_read_tr": "ds_read_tr",
    "ds_write": "ds_write",
    "vmem_load": "vmem_load",
    "vmem_store": "vmem_store",
}
# sched_group class -> instruction classes it can take (as this checker counts them)
GROUP_CLASSES = {
    "mfma": {"mfma"},
    "valu": {"valu"},
    "trans": {"trans"},
    "salu": {"salu"},
    "ds_read": {"ds_read", "ds_read_tr"},
    "ds_write": {"ds_write"},
    "all_ds": {"ds_read", "ds_read_tr", "ds_write"},
    "vmem_read": {"vmem_load", "lds_dma"},
    "vmem_write": {"vmem_store"},
    "all_vmem": {"vmem_load", "vmem_store", "lds_dma"},
    "ldsdma": {"lds_dma"},
}
SHORT = {"mfma": "M", "valu": "V", "trans": "T", "salu": "S", "ds_read": "R", "ds_read_tr": "Rt", "ds_write": "W"}


def classify(mnemonic: str, operands: str = "") -> str:
    m = mnemonic
    if m.startswith(("v_mfma", "v_smfmac")):
        return "mfma"
    if m.startswith(TRANS_OPS):
        return "trans"
    if m.startswith("v_"):
        return "valu"
    if m.startswith("ds_read") or m.startswith("ds_load"):
        return "ds_read_tr" if "_tr" in m else "ds_read"
    if m.startswith(("ds_write", "ds_store")):
        return "ds_write"
    if m.startswith("ds_"):
        return "ds_other"
    if m.startswith(("buffer_load", "global_load", "flat_load")):
        return "lds_dma" if re.search(r"\blds\b", operands) else "vmem_load"
    if m.startswith(("buffer_store", "global_store", "flat_store", "buffer_atomic", "global_atomic")):
        return "vmem_store"
    if m.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if m.startswith("s_waitcnt"):
        return "waitcnt"
    if m == "s_barrier":
        return "barrier"
    if m == "s_setprio":
        return "setprio"
    if m == "s_nop":
        return "nop"
    if m.startswith(("s_cbranch", "s_branch", "s_endpgm", "s_setpc", "s_swappc")):
        return "branch"
    if m.startswith("s_"):
        return "salu"
    return "other"


@dataclass
class Inst:
    idx: int  # position among the kernel's instructions
    line: int  # 1-based line in the ISA file
    text: str
    mnemonic: str
    operands: str
    cls: str
    asm: bool  # inside an inline-asm block (invisible to LLVM's scheduler / waitcnt pass)
    block: str  # enclosing label
    src: Optional[Tuple[str, int, int]] = None  # (file, line, col) from .loc
    section: Optional[str] = None  # schedule section, when src is a sched: tag
    op: Optional[int] = None  # 0-based schedule op index
    tag: Optional[str] = None

    @property
    def where(self) -> str:
        if self.section is not None:
            return f"{self.section}/{self.op}"
        return f"{self.src[0]}:{self.src[1]}" if self.src else "?"


@dataclass
class IsaKernel:
    name: str
    insts: List[Inst]
    meta: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def tagged(self) -> bool:
        return any(i.section is not None for i in self.insts)

    def class_counts(self) -> Counter:
        return Counter(i.cls for i in self.insts)


_FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
_LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?")
_META_RE = re.compile(r"^\s*;\s*([A-Za-z][\w ]*?):\s+(\S+)\s*$")
_HSA_RE = re.compile(r"^\s*\.amdhsa_(\w+)\s+(\S+)")


def _parse_tag(path: str, line: int):
    """``sched:<tag>/<section>`` + line -> (tag, section, op index), or None."""
    if not path.startswith(TAG_SCHEME) or line <= 0:
        return None
    rest = path[len(TAG_SCHEME) :]
    tag, _, section = rest.rpartition("/")
    if not tag:
        tag, section = rest, ""
    return tag, section, line - 1


def kernel_names(text: str) -> List[str]:
    return re.findall(r"^\s*\.type\s+([\w.$]+),\s*@function", text, flags=re.M)


def parse_isa(text: str, kernel: Optional[str] = None) -> IsaKernel:
    """Instructions of ``kernel`` (default: the first function) in an AMDGPU ``.s`` file."""
    names = kernel_names(text)
    if kernel is None:
        if not names:
            raise ValueError("no function (.type <name>,@function) in the ISA text")
        kernel = names[0]
    elif kernel not in names:
        raise ValueError(f"kernel {kernel!r} not in ISA; functions: {names}")
    files: Dict[str, str] = {}
    insts: List[Inst] = []
    meta: Dict[str, str] = {}
    tags: List[str] = []
    state = "before"  # before -> body -> after
    src = None
    block = kernel
    in_asm = False
    desc = None  # kernel whose .amdhsa_kernel descriptor is being read
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        fm = _FILE_RE.match(raw)
        if fm:
            num, a, b = fm.groups()
            path = f"{a}/{b}" if b is not None and a else (b if b is not None else a)
            files[num] = path
            continue
        if state == "before":
            if line == f"{kernel}:":
                state = "body"
            continue
        if state == "after":
            if line.startswith(".amdhsa_kernel "):
                desc = line.split()[-1]
                continue
            if line.startswith(".end_amdhsa_kernel"):
                desc = None
                continue
            hm = _HSA_RE.match(raw)
            if hm:
                if desc == kernel:
                    meta[hm.group(1)] = hm.group(2)
                continue
            mm = _META_RE.match(raw)
            if mm and mm.group(1) not in meta:
                meta[mm.group(1)] = mm.group(2)
            continue
        # body
        if line.startswith(".Lfunc_end") or line.startswith(".size"):
            state = "after"
            continue
        lm = _LOC_RE.match(raw)
        if lm:
            f, ln, col = lm.group(1), int(lm.group(2)), int(lm.group(3) or 0)
            src = (files.get(f, f"file{f}"), ln, col)
            continue
        if line.startswith(";;#ASMSTART"):
            in_asm = True
            continue
        if line.startswith(";;#ASMEND"):
            in_asm = False
            continue
        if re.match(r"^[\w.$]+:$", line):
            # debug-info temp labels and local labels inside inline asm ("1:") do not start a block
            if not in_asm and not line.startswith((".Ltmp", ".Lfunc_begin")):
                block = line[:-1]
            continue
        if not line or line.startswith((";", ".", "//")):
            continue
        body = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not body:
            continue
        mnemonic, _, operands = body.partition(" ")
        inst = Inst(len(insts), n, body, mnemonic, operands.strip(), classify(mnemonic, operands), in_asm, block, src)
        if src is not None:
            t = _parse_tag(src[0], src[1])
            if t is not None:
                inst.tag, inst.section, inst.op = t
                if t[0] not in tags:
                    tags.append(t[0])
        insts.append(inst)
    if state == "before":
        raise ValueError(f"kernel label {kernel}: not found")
    return IsaKernel(kernel, insts, meta, tags)


def read_isa(path: str, kernel: Optional[str] = None) -> IsaKernel:
    with open(path, encoding="utf-8") as f:
        return parse_isa(f.read(), kernel)


def waitcnt_values(inst: Inst) -> Dict[str, int]:
    """Counters of an ``s_waitcnt``; a counter it does not name stays at its maximum (no wait)."""
    out = dict(MAX_WAIT)
    ops = inst.operands
    if inst.mnemonic != "s_waitcnt":  # s_waitcnt_vmcnt null, 0 style (not on gfx9, kept for completeness)
        name = inst.mnemonic[len("s_waitcnt_") :].replace("cnt", "")
        if name in out:
            out[name] = int(ops.split(",")[-1].strip(), 0)
        return out
    for k, v in re.findall(r"(vm|lgkm|exp)cnt\((\d+)\)", ops):
        out[k] = int(v)
    if ops.strip() == "0":
        out = {k: 0 for k in out}
    return out


def _fmt_wait(w: Dict[str, int]) -> str:
    parts = [f"{k}cnt({v})" for k, v in w.items() if v < MAX_WAIT[k]]
    return "s_waitcnt " + (" ".join(parts) if parts else "(none)")


def _rle(classes: Sequence[str], limit: int = 40) -> str:
    runs: List[List] = []
    for c in classes:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    s = " ".join(f"{SHORT.get(c, c)}{n}" for c, n in runs[:limit])
    return s + (f" ... (+{len(runs) - limit} runs)" if len(runs) > limit else "")


# --------------------------------------------------------------------------- conformance


class ConformReport(Report):
    """``verify.Report`` with conformance findings plus per-op and diff summaries."""

    def __init__(self, schedule, target, isa_kernel, **kw):
        super().__init__(schedule, target, [], **kw)
        self.kernel = isa_kernel
        self.summary: List[str] = []

    def to_json(self):
        d = json.loads(super().to_json())
        d.update(kernel=self.kernel, summary=self.summary)
        return json.dumps(d, indent=2)

    def render(self, verbose: bool = True) -> str:
        text = super().render(verbose)
        text = "conform" + text[len("verify") :] if text.startswith("verify") else text
        head, _, rest = text.partition("\n")
        head = head.replace(" tiles=[]", f" kernel={self.kernel}")
        lines = [head] + ([rest] if rest else [])
        if verbose and self.summary:
            lines.append("  summary:")
            lines += [f"    {s}" for s in self.summary]
        return "\n".join(lines)


def _regions(ops: Sequence[Op]) -> List[int]:
    """Region index of every op: code may not move between regions.

    ``sched_barrier(0)`` is a boundary.  A ``barrier`` is a boundary on the side(s)
    its ``fence`` puts a ``sched_barrier(0)``: with ``fence="before"`` later code may
    still move above the ``s_barrier`` itself, so the barrier joins the next region.
    ``stagger`` emits a fenced barrier and counts as ``fence="both"``.
    """
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


def _wait_request(op: Op) -> Optional[Dict[str, int]]:
    if op.kind == "wait":
        req = {k: op.args.get(k) for k in ("vm", "lgkm", "exp")}
    elif op.kind == "barrier" and (op.args.get("vm") is not None or op.args.get("lgkm") is not None):
        req = {"vm": op.args.get("vm"), "lgkm": op.args.get("lgkm"), "exp": None}
    else:
        return None
    return {k: MAX_WAIT[k] if v is None else v for k, v in req.items()}


def _satisfies(actual: Dict[str, int], req: Dict[str, int]) -> bool:
    return all(actual[k] <= req[k] for k in req)


def conform(
    isa: IsaKernel,
    schedule: Optional[Schedule] = None,
    tag: Optional[str] = None,
    target: Optional[str] = None,
) -> ConformReport:
    """Check ``isa`` (a ``parse_isa`` result) against ``schedule``.

    Without a schedule only the attribution summary is produced.  ``tag``
    restricts the check to instructions of one ``emit(tag=...)`` name (default:
    the only tag in the file).
    """
    sname = schedule.name if schedule is not None else "(no schedule)"
    rep = ConformReport(sname, target or (schedule.target if schedule else "?"), isa.name)
    if not isa.tagged:
        rep.add(
            ("NO_TAGS",),
            Finding(
                "CONFORM.NO_TAGS",
                "error",
                "no instruction carries a sched: location",
                hint="emit with tag=<name> and compile with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_DUMP_IR=1",
            ),
        )
        return rep
    if tag is None and len(isa.tags) > 1:
        raise ValueError(f"several schedule tags in the ISA {isa.tags}; pass tag=")
    tag = tag or isa.tags[0]
    mine = [i for i in isa.insts if i.tag == tag]
    by_op: Dict[Tuple[str, int], List[Inst]] = defaultdict(list)
    for i in mine:
        by_op[(i.section, i.op)].append(i)
    sections = []
    for i in mine:
        if i.section not in sections:
            sections.append(i.section)

    _unattributed(rep, isa, tag)
    if schedule is None:
        for sec in sections:
            ops = sorted({i.op for i in mine if i.section == sec})
            rep.summary.append(f"{sec}: {len(ops)} ops with instructions, {sum(i.section == sec for i in mine)} insts")
        return rep

    known = {s.name: s for s in schedule.sections}
    for sec in sections:
        if sec not in known:
            rep.add(
                ("SECTION", sec),
                Finding("CONFORM.UNKNOWN_SECTION", "error", f"ISA has section {sec!r}; schedule has {sorted(known)}"),
            )
    for sec_name in [s for s in sections if s in known]:
        ops = known[sec_name].ops
        regions = _regions(ops)
        sec_insts = [i for i in mine if i.section == sec_name]
        bad = [i for i in sec_insts if i.op >= len(ops)]
        if bad:
            rep.add(
                ("RANGE", sec_name),
                Finding(
                    "CONFORM.UNKNOWN_OP",
                    "error",
                    f"{len(bad)} instruction(s) tagged with op index >= {len(ops)} (first: line {bad[0].line})",
                    where=sec_name,
                ),
            )
            sec_insts = [i for i in sec_insts if i.op < len(ops)]
        _check_order(rep, sec_name, ops, regions, sec_insts)
        _check_waits(rep, sec_name, ops, regions, sec_insts, by_op)
        _check_barriers(rep, sec_name, ops, sec_insts, by_op)
        _check_setprio(rep, sec_name, ops, sec_insts, by_op)
        _check_interleave(rep, sec_name, ops, regions, sec_insts)
        _check_counts(rep, sec_name, ops, by_op)
        rep.summary.append(_section_summary(sec_name, ops, sec_insts))
    return rep


def _unattributed(rep: ConformReport, isa: IsaKernel, tag: str):
    mine = [i for i in isa.insts if i.tag == tag]
    others = [i for i in isa.insts if i.tag != tag]
    if not others:
        return
    span = (mine[0].idx, mine[-1].idx) if mine else (0, -1)
    inside = [i for i in others if span[0] <= i.idx <= span[1]]
    by_src = Counter(i.where for i in others)
    top = ", ".join(f"{s} x{n}" for s, n in by_src.most_common(6))
    cls = Counter(i.cls for i in others)
    rep.add(
        ("UNATTRIBUTED",),
        Finding(
            "CONFORM.UNATTRIBUTED",
            "info",
            f"{len(others)} of {len(isa.insts)} instructions are not attributed to a schedule op "
            f"({len(inside)} inside the scheduled span); classes {dict(cls.most_common())}; top sources: {top}",
            hint="address math and setup emitted outside emit(), hoisted code, or helpers that set their own "
            "location (e.g. @flyc.jit helpers under debug info)",
        ),
    )
    for i in inside:
        if i.cls == "waitcnt":
            rep.add(
                ("INSERTED", None, i.where, i.text),
                Finding(
                    "CONFORM.WAIT_INSERTED",
                    "info",
                    f"'{i.text}' (ISA line {i.line}) inside the scheduled span is not in the schedule; "
                    f"its location is user code {i.where}",
                    where=i.where,
                ),
            )
        if i.cls in ("mfma", "lds_dma", "ds_read", "ds_read_tr", "ds_write", "vmem_load", "vmem_store", "barrier"):
            rep.add(
                ("UNATTR_HEAVY", i.cls, i.where),
                Finding(
                    "CONFORM.UNATTRIBUTED_HEAVY",
                    "warn",
                    f"{i.cls} instruction inside the scheduled span is not attributed to a schedule op "
                    f"(source {i.where}, first at ISA line {i.line}: {i.text})",
                    where=i.where,
                ),
            )


def _label(sec, ops, k):
    return f"{sec}/{k} {ops[k]}"


def _check_order(rep, sec, ops, regions, insts):
    """Instructions issued after code of a later fence region, within one basic block.

    Only straight-line order is compared: code the compiler places in another block
    (out-of-line ``if`` bodies, loop-exit copies) is not "issued after" anything.
    """
    hi = -1
    hi_op = None
    block = None
    for i in insts:
        if i.block != block:
            block, hi, hi_op = i.block, -1, None
        r = regions[i.op]
        if r < hi and i.cls not in ("waitcnt", "nop"):
            rep.add(
                ("ORDER", sec, i.op, hi),
                Finding(
                    "CONFORM.ORDER",
                    "warn",
                    f"{i.cls} of {_label(sec, ops, i.op)} (region {r}) issues after code of "
                    f"{_label(sec, ops, hi_op)} (region {hi}); first at ISA line {i.line}: {i.text}",
                    where=f"{sec}/{i.op}",
                ),
            )
        elif r > hi:
            hi, hi_op = r, i.op


def _check_waits(rep, sec, ops, regions, insts, by_op):
    waits = [i for i in insts if i.cls == "waitcnt" and not i.asm]
    # which schedule wait ops each ISA waitcnt stands for
    explains: Dict[int, List[int]] = defaultdict(list)
    for k, op in enumerate(ops):
        req = _wait_request(op)
        if req is None or (op.kind == "barrier" and op.args.get("asm")):
            continue
        own = [i for i in by_op.get((sec, k), []) if i.cls == "waitcnt"]
        if own:
            for i in own:
                explains[i.idx].append(k)
            continue
        # merged into a neighbouring waitcnt of the same region that is at least as strict?
        cands = [i for i in waits if regions[i.op] == regions[k] and _satisfies(waitcnt_values(i), req)]
        if cands:
            host = min(cands, key=lambda i: abs(i.op - k))
            explains[host.idx].append(k)
            rep.add(
                ("MERGED", sec, k),
                Finding(
                    "CONFORM.WAIT_MERGED",
                    "info",
                    f"{_label(sec, ops, k)} has no s_waitcnt of its own; it is covered by "
                    f"'{host.text}' (ISA line {host.line}, op {host.op})",
                    where=f"{sec}/{k}",
                ),
            )
        else:
            rep.add(
                ("DROPPED", sec, k),
                Finding(
                    "CONFORM.WAIT_DROPPED",
                    "warn",
                    f"{_label(sec, ops, k)}: no s_waitcnt in the ISA satisfies it in its region",
                    where=f"{sec}/{k}",
                    hint="LLVM's waitcnt pass removes waits it believes redundant; it does not see inline-asm "
                    "or LDS-DMA producers",
                ),
            )
    for i in waits:
        got = waitcnt_values(i)
        ks = explains.get(i.idx)
        if not ks:
            rep.add(
                ("INSERTED", sec, i.op, i.text),
                Finding(
                    "CONFORM.WAIT_INSERTED",
                    "info",
                    f"'{i.text}' (ISA line {i.line}) is not in the schedule; placed at {_label(sec, ops, i.op)}",
                    where=f"{sec}/{i.op}",
                ),
            )
            continue
        reqs = [_wait_request(ops[k]) for k in ks]
        want = {c: min(r[c] for r in reqs) for c in MAX_WAIT}
        if got == want:
            continue
        looser = [c for c in MAX_WAIT if got[c] > want[c]]
        stricter = [c for c in MAX_WAIT if got[c] < want[c]]
        names = ", ".join(_label(sec, ops, k) for k in ks)
        if looser:
            rep.add(
                ("LOOSER", sec, i.idx),
                Finding(
                    "CONFORM.WAIT_LOOSENED",
                    "error",
                    f"{names}: schedule asks {_fmt_wait(want)}, ISA line {i.line} has '{i.text}' (looser on {looser})",
                    where=f"{sec}/{ks[0]}",
                ),
            )
        elif stricter:
            rep.add(
                ("STRICTER", sec, i.idx),
                Finding(
                    "CONFORM.WAIT_TIGHTENED",
                    "warn",
                    f"{names}: schedule asks {_fmt_wait(want)}, ISA line {i.line} has '{i.text}' "
                    f"(stricter on {stricter}: the compiler folded in a wait it tracks itself)",
                    where=f"{sec}/{ks[0]}",
                ),
            )


def _check_barriers(rep, sec, ops, insts, by_op):
    want = [k for k, op in enumerate(ops) if op.kind == "barrier"]
    missing = [k for k in want if not any(i.cls == "barrier" for i in by_op.get((sec, k), []))]
    got = sum(i.cls == "barrier" for i in insts)
    on_barrier_ops = sum(i.cls == "barrier" and ops[i.op].kind == "barrier" for i in insts)
    stagger = [k for k, op in enumerate(ops) if op.kind == "stagger"]
    extra = [i for i in insts if i.cls == "barrier" and ops[i.op].kind not in ("barrier", "stagger")]
    if missing or extra or on_barrier_ops != len(want):
        rep.add(
            ("BARRIERS", sec),
            Finding(
                "CONFORM.BARRIER_COUNT",
                "error",
                f"{sec}: schedule has {len(want)} barrier op(s) (+{len(stagger)} stagger), ISA has {got} s_barrier "
                f"({on_barrier_ops} on barrier ops); missing at {[_label(sec, ops, k) for k in missing][:6]}"
                + (f"; extra at {sorted({_label(sec, ops, i.op) for i in extra})[:6]}" if extra else ""),
                where=sec,
            ),
        )


def _check_setprio(rep, sec, ops, insts, by_op):
    for k, op in enumerate(ops):
        if op.kind != "setprio":
            continue
        mine = [i for i in by_op.get((sec, k), []) if i.cls == "setprio"]
        if not mine:
            rep.add(
                ("PRIO_MISSING", sec, k),
                Finding("CONFORM.SETPRIO_MISSING", "error", f"{_label(sec, ops, k)}: no s_setprio", where=f"{sec}/{k}"),
            )
            continue
        for s in mine:
            val = int(s.operands.split()[0], 0)
            if val != op.args["prio"]:
                rep.add(
                    ("PRIO_VAL", sec, k),
                    Finding(
                        "CONFORM.SETPRIO_VALUE",
                        "warn",
                        f"{_label(sec, ops, k)}: ISA line {s.line} has '{s.text}'",
                        where=f"{sec}/{k}",
                    ),
                )
            # compute work of later ops issued before it / of earlier ops after it
            same = [i for i in insts if i.block == s.block and i.cls in ("mfma", "valu", "trans")]
            early = Counter(i.cls for i in same if i.op > k and i.idx < s.idx)
            late = Counter(i.cls for i in same if i.op < k and i.idx > s.idx)
            if early or late:
                sev = "warn" if early.get("mfma") or late.get("mfma") else "info"
                rep.add(
                    ("PRIO_PLACE", sec, k, s.idx),
                    Finding(
                        "CONFORM.SETPRIO_PLACEMENT",
                        sev,
                        f"{_label(sec, ops, k)} at ISA line {s.line}: later ops issue {dict(early)} before it, "
                        f"earlier ops issue {dict(late)} after it",
                        where=f"{sec}/{k}",
                    ),
                )


def _group_block(k, in_region):
    """Basic block a code-less ``sched_group`` op ``k`` lands in.

    ``sched_group_barrier`` acts on its own scheduling region (at most one basic block):
    the block holding most of the code of the nearest earlier op in the region that has
    code (instructions LLVM sank elsewhere do not count), else of the nearest later op.
    """
    before = [i.op for i in in_region if i.op < k]
    after = [i.op for i in in_region if i.op > k]
    if not before and not after:
        return None
    near = max(before) if before else min(after)
    return Counter(i.block for i in in_region if i.op == near).most_common(1)[0][0]


def _check_interleave(rep, sec, ops, regions, insts):
    for r in sorted(set(regions)):
        ks = [k for k, op in enumerate(ops) if regions[k] == r]
        in_region = [i for i in insts if regions[i.op] == r]
        by_block: Dict[str, List[int]] = defaultdict(list)
        for k in ks:
            if ops[k].kind == "sched_group":
                b = _group_block(k, in_region)
                if b is not None:
                    by_block[b].append(k)
        for main, gks in by_block.items():
            _interleave_one(rep, sec, ops, r, gks, main, in_region)


def _interleave_one(rep, sec, ops, r, gks, main, in_region):
    """Requested sched_group slots of one (region, block) against the issued MFMA/VALU/... sequence."""
    request: List[str] = []
    wanted = set()
    for k in gks:
        cls = ops[k].args["cls"]
        wanted |= GROUP_CLASSES.get(cls, {cls})
        request += [cls] * ops[k].args["size"]
    actual = [i.cls for i in in_region if i.block == main and i.cls in wanted and not i.asm]
    # greedy in-order match of the requested slots against the issued sequence; a slot whose
    # class has no instruction left in the block cannot be filled and is skipped
    left = Counter(actual)
    matched = p = 0
    diverged = None
    for s, cls in enumerate(request):
        cs = GROUP_CLASSES.get(cls, {cls})
        if not any(left[c] for c in cs):
            continue
        if p < len(actual) and actual[p] in cs:
            left[actual[p]] -= 1
            matched += 1
            p += 1
        else:
            diverged = s
            break
    need, have = Counter(request), Counter(actual)
    short = {c: (n, sum(have[x] for x in GROUP_CLASSES.get(c, {c}))) for c, n in need.items()}
    short = {c: v for c, v in short.items() if v[1] < v[0]}
    where = f"{sec}/{gks[0]}..{gks[-1]}"
    blocks = sorted({i.block for i in in_region}, key=lambda b: min(i.idx for i in in_region if i.block == b))
    split = f" (region code spans blocks {blocks})" if len(blocks) > 1 else ""
    msg = f"region {r} groups {where} in block {main}{split}: requested {_rle(request)}; issued {_rle(actual)}; "
    if diverged is None:
        msg += f"all {matched} fillable slots in requested order"
    else:
        msg += f"{matched} slots in requested order, then slot {diverged} ({request[diverged]}) diverges"
    if short:
        msg += f"; more slots than instructions (requested, present): {short}"
    sev = "info" if diverged is None else "warn"
    rep.add(("INTERLEAVE", sec, r, main), Finding("CONFORM.INTERLEAVE", sev, msg, where=where))


def _check_counts(rep, sec, ops, by_op):
    for k, op in enumerate(ops):
        if op.kind in CONTROL_KINDS:
            continue
        got = by_op.get((sec, k), [])
        cls = COUNT_CLASS.get(op.kind)
        if not got and op.count:
            rep.add(
                ("OP_MISSING", sec, k),
                Finding(
                    "CONFORM.OP_MISSING",
                    "warn",
                    f"{_label(sec, ops, k)}: no instruction attributed (expected {op.count} {cls or op.kind})",
                    where=f"{sec}/{k}",
                    hint="code merged into another op, CSE'd/hoisted, or built by a helper that sets its own location",
                ),
            )
            continue
        if cls is None or not op.count:
            continue
        n = sum(i.cls == cls for i in got)
        if n != op.count:
            rep.add(
                ("COUNT", sec, k),
                Finding(
                    "CONFORM.COUNT",
                    "warn",
                    f"{_label(sec, ops, k)}: schedule says {op.count} {cls}, ISA has {n}",
                    where=f"{sec}/{k}",
                ),
            )


def _section_summary(sec, ops, insts) -> str:
    c = Counter(i.cls for i in insts)
    with_code = len({i.op for i in insts})
    keys = ("mfma", "valu", "trans", "salu", "ds_read", "ds_read_tr", "lds_dma", "vmem_load", "waitcnt", "barrier")
    return f"{sec}: {len(ops)} ops ({with_code} with code), {len(insts)} insts: " + ", ".join(
        f"{k}={c[k]}" for k in keys if c[k]
    )


def op_table(isa: IsaKernel, schedule: Optional[Schedule] = None, tag: Optional[str] = None) -> List[str]:
    """One line per schedule op: ISA lines and instruction classes attributed to it."""
    tag = tag or (isa.tags[0] if isa.tags else None)
    rows: Dict[Tuple[str, int], List[Inst]] = defaultdict(list)
    for i in isa.insts:
        if i.tag == tag:
            rows[(i.section, i.op)].append(i)
    secs = {s.name: s.ops for s in schedule.sections} if schedule else {}
    out = []
    for (sec, k), insts in sorted(rows.items(), key=lambda kv: (kv[1][0].idx)):
        label = str(secs[sec][k]) if sec in secs and k < len(secs[sec]) else ""
        c = Counter(i.cls for i in insts)
        out.append(
            f"{sec:9s} {k:4d} {label:34.34s} lines {insts[0].line}-{insts[-1].line}  "
            + " ".join(f"{SHORT.get(x, x)}{n}" for x, n in c.most_common())
        )
    return out


# --------------------------------------------------------------------------- ISA diff

DIFF_KEYS = ("mfma", "lds_dma", "ds_read", "ds_read_tr", "ds_write", "vmem_load", "vmem_store", "trans")
DETAIL_KEYS = ("valu", "salu", "smem", "waitcnt", "nop", "setprio", "branch")
# shown even when equal; every other differing .amdhsa_* / "; Key: value" entry is listed too
META_KEYS = ("next_free_vgpr", "next_free_sgpr", "accum_offset", "group_segment_fixed_size",
             "private_segment_fixed_size", "kernarg_size", "NumVgprs", "NumAgprs", "Occupancy", "codeLenInByte")  # fmt: skip


@dataclass
class Region:
    insts: List[Inst]

    @property
    def key(self):
        c = Counter(i.cls for i in self.insts)
        return tuple(c[k] for k in DIFF_KEYS)

    @property
    def label(self) -> str:
        for i in self.insts:
            if i.section is not None:
                return f"{i.section}/{i.op}"
        return self.insts[0].block if self.insts else ""

    @property
    def lines(self) -> str:
        return f"{self.insts[0].line}-{self.insts[-1].line}" if self.insts else "-"


def split_regions(isa: IsaKernel) -> List[Region]:
    """Barrier regions: each ``s_barrier`` closes one (it belongs to the region it ends)."""
    out, cur = [], []
    for i in isa.insts:
        cur.append(i)
        if i.cls == "barrier":
            out.append(Region(cur))
            cur = []
    if cur:
        out.append(Region(cur))
    return out


@dataclass
class IsaDiff:
    a: str
    b: str
    meta: List[Tuple[str, str, str]]
    totals: List[Tuple[str, int, int]]
    rows: List[str]
    findings: List[str]

    def render(self) -> str:
        lines = [f"isa diff  A={self.a}  B={self.b}"]
        changed = [f"{k} {x}->{y}" for k, x, y in self.meta if x != y]
        lines.append("  resources: " + (", ".join(changed) if changed else "identical"))
        if any(x == y for _, x, y in self.meta):
            lines.append("  same: " + ", ".join(f"{k}={x}" for k, x, y in self.meta if x == y))
        lines.append("  instruction totals (A -> B): " + ", ".join(f"{k} {x}->{y}" for k, x, y in self.totals))
        lines += [f"  {f}" for f in self.findings]
        lines.append("  regions (A lines | B lines | difference):")
        lines += [f"    {r}" for r in self.rows]
        return "\n".join(lines)


def _seq_delta(a: List[Inst], b: List[Inst]) -> str:
    """Summary of how two instruction sequences differ (mnemonics only, registers ignored)."""
    ma, mb = [i.mnemonic for i in a], [i.mnemonic for i in b]
    if ma == mb:
        pairs = [(x, y) for x, y in zip(a, b) if x.text != y.text]
        if not pairs:
            return "identical"
        shown = "; ".join(f"'{x.text}' vs '{y.text}'" for x, y in pairs[:3])
        return f"same opcodes, {len(pairs)} operand difference(s): {shown}" + (" ..." if len(pairs) > 3 else "")
    ca, cb = Counter(i.cls for i in a), Counter(i.cls for i in b)
    parts = []
    for k in sorted(set(ca) | set(cb)):
        if ca[k] != cb[k]:
            parts.append(f"{k} {ca[k]}->{cb[k]}")
    wa = [i.text for i in a if i.cls == "waitcnt"]
    wb = [i.text for i in b if i.cls == "waitcnt"]
    if wa != wb:
        parts.append(f"waits A={_short_waits(wa)} B={_short_waits(wb)}")
    if Counter(ma) == Counter(mb):
        sm = difflib.SequenceMatcher(None, ma, mb, autojunk=False)
        moved = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
        parts.append(f"same opcode multiset, order differs ({moved} positions)")
        pa = _rle([i.cls for i in a if i.cls in ("mfma", "valu", "trans")], 12)
        pb = _rle([i.cls for i in b if i.cls in ("mfma", "valu", "trans")], 12)
        if pa != pb:
            parts.append(f"A[{pa}] B[{pb}]")
    return "; ".join(parts) or "opcodes differ"


def _short_waits(ws: List[str]) -> str:
    return "[" + ", ".join(w.replace("s_waitcnt ", "") for w in ws) + "]"


def diff_isa(a: IsaKernel, b: IsaKernel) -> IsaDiff:
    keys = [k for k in META_KEYS if k in a.meta or k in b.meta]
    keys += sorted(k for k in set(a.meta) | set(b.meta) if k not in keys and a.meta.get(k) != b.meta.get(k))
    meta = [(k, a.meta.get(k, "?"), b.meta.get(k, "?")) for k in keys]
    ca, cb = a.class_counts(), b.class_counts()
    totals = [(k, ca[k], cb[k]) for k in sorted(set(ca) | set(cb), key=lambda k: -max(ca[k], cb[k]))]
    totals = [("all", len(a.insts), len(b.insts))] + totals
    ra, rb = split_regions(a), split_regions(b)
    sm = difflib.SequenceMatcher(None, [r.key for r in ra], [r.key for r in rb], autojunk=False)
    rows, findings = [], []
    n_ident = n_diff = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace") and i2 - i1 == j2 - j1:
            for x, y in zip(ra[i1:i2], rb[j1:j2]):
                d = _seq_delta(x.insts, y.insts)
                n_ident += d == "identical"
                if d != "identical":
                    n_diff += 1
                    rows.append(f"A {x.lines:11s} | B {y.lines:11s} [{y.label or x.label}] {d}")
            continue
        for x in ra[i1:i2]:
            rows.append(f"A {x.lines:11s} | B {'-':11s} only in A: {dict(Counter(i.cls for i in x.insts))}")
        for y in rb[j1:j2]:
            rows.append(f"A {'-':11s} | B {y.lines:11s} [{y.label}] only in B: {dict(Counter(i.cls for i in y.insts))}")
    findings.append(
        f"{len(ra)} vs {len(rb)} barrier regions; {n_ident} identical, {n_diff} aligned but different, "
        f"{sum(r.startswith('A -') or '| B -' in r for r in rows)} unaligned"
    )
    return IsaDiff(a.name, b.name, meta, totals, rows, findings)
