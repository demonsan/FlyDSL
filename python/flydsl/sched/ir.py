# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Schedule IR: explicit, analyzable descriptions of a kernel's hot path.

A schedule is *data*.  It says which on-chip buffers exist and how many
slots they have, which async queue (hardware counter) every async op is
counted on, which LDS slot / tile every op reads or writes, and the exact
order of compute, memory, wait, barrier and scheduler-hint ops.

The same op list is consumed by three clients:

* ``verify``  - simulates every wave group over concrete tile counts and
  reports synchronization / data-consistency / hardware-contract findings;
* ``resource``/``cost`` - LDS, VGPR, occupancy and an MFMA/roofline bound;
* ``lower.emit`` - walks the op list inside a ``@flyc.kernel`` and calls the
  kernel's bindings, so the emitted kernel *is* the verified schedule.

This module is pure Python: it must import without MLIR or ROCm.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Op kinds understood by the analyzers.  Compute/memory kinds are lowered
# through kernel bindings; control kinds are lowered by ``lower`` itself.
COMPUTE_KINDS = ("mfma", "valu", "trans", "salu")
MEMORY_KINDS = ("lds_dma", "vmem_load", "vmem_store", "ds_read", "ds_read_tr", "ds_write", "smem")
CONTROL_KINDS = ("wait", "barrier", "sched_barrier", "sched_group", "setprio", "nop", "stagger", "anchor")
ALL_KINDS = COMPUTE_KINDS + MEMORY_KINDS + CONTROL_KINDS


@dataclass(frozen=True)
class TileRef:
    """A tile index ``bases[base] + off``; ``base`` is resolved per section/iteration."""

    base: str
    off: int = 0

    def eval(self, bases: Dict[str, int]) -> int:
        if self.base not in bases:
            raise KeyError(f"tile base {self.base!r} is not bound here (bound: {sorted(bases)})")
        return bases[self.base] + self.off

    def __str__(self):
        return self.base if self.off == 0 else f"{self.base}{self.off:+d}"


def T(base: str, off: int = 0) -> TileRef:
    return TileRef(base, off)


@dataclass(frozen=True)
class SlotUse:
    """One access to one slot of an on-chip buffer, optionally tagged with the tile it holds."""

    buffer: str
    slot: int
    tile: Optional[TileRef] = None

    def __str__(self):
        tag = f"<{self.tile}>" if self.tile is not None else ""
        return f"{self.buffer}[{self.slot}]{tag}"


@dataclass(frozen=True)
class Buffer:
    name: str
    tier: str  # "lds" | "vgpr"
    tensor: str  # logical tensor held, e.g. "K"
    slots: int
    bytes_per_slot: int = 0
    note: str = ""


@dataclass(frozen=True)
class Queue:
    """An async completion queue backed by a hardware counter.

    ``tracked`` means the compiler inserts the wait itself before a consumer
    (plain LLVM loads); untracked queues (LDS DMA, inline-asm ds_read_tr)
    must be drained by explicit ``wait`` ops or the verifier reports it.
    """

    name: str
    counter: str  # "vm" | "lgkm"
    tracked: bool = False


@dataclass
class Op:
    kind: str
    label: str = ""
    binding: Optional[str] = None
    ins: Tuple[str, ...] = ()
    outs: Tuple[str, ...] = ()
    args: Dict[str, object] = field(default_factory=dict)
    count: int = 1  # hardware instructions this op issues (DMA lines, MFMAs, ds_reads ...)
    mfma: Optional[str] = None  # MFMA instruction name (kind == "mfma")
    reads: Tuple[SlotUse, ...] = ()
    writes: Tuple[SlotUse, ...] = ()
    queue: Optional[str] = None
    out_dwords: Dict[str, int] = field(default_factory=dict)  # per-lane VGPR dwords of each out
    consumes: Tuple[Tuple[str, TileRef], ...] = ()  # (tensor, tile) consumed by this compute op
    note: str = ""

    def __str__(self):
        return self.label or f"{self.kind}:{self.binding}"


@dataclass
class Section:
    """A straight-line op list; ``loop`` sections repeat with ``var`` bound per iteration.

    ``carried`` names the values a loop section carries across iterations (its
    ``scf.for`` iter_args).  Any other value the loop redefines is scoped to one
    iteration and must not be read before its definition or after the loop.
    """

    name: str
    ops: List[Op]
    loop: Optional[Tuple[str, Callable[[Dict[str, int]], range]]] = None  # (var, bases -> range)
    carried: Tuple[str, ...] = ()


@dataclass
class Schedule:
    name: str
    target: str
    buffers: Dict[str, Buffer]
    queues: Dict[str, Queue]
    sections: List[Section]
    groups: int = 1  # wave groups that execute the same program (dual-wave = 2)
    stagger: bool = False  # dual-wave phase shift via ``stagger`` ops
    waves_per_group: int = 1
    bases: Callable[[int], Dict[str, int]] = lambda n: {}  # n_tiles -> tile bases ("first", "end", ...)
    # Optional n_tiles -> sections: for programs whose shape depends on the tile count
    # (e.g. first / middle / last tile paths).  ``sections`` then holds the steady state.
    program: Optional[Callable[[int], List["Section"]]] = None
    valid_tiles: Callable[[int], bool] = lambda n: n >= 1
    tensors: Dict[str, str] = field(default_factory=dict)  # tensor -> "every tile consumed once by <binding>"
    meta: Dict[str, object] = field(default_factory=dict)

    def section(self, name: str) -> Section:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(f"{self.name}: no section {name!r}")

    def all_ops(self) -> List[Op]:
        return [op for s in self.sections for op in s.ops]


class Builder:
    """Small helper so schedules read like the expert's description."""

    def __init__(self):
        self.ops: List[Op] = []

    def _add(self, op: Op) -> Op:
        if op.kind not in ALL_KINDS:
            raise ValueError(f"unknown op kind {op.kind!r}")
        self.ops.append(op)
        return op

    # ---- compute / memory -------------------------------------------------
    def op(self, kind, binding, *, ins=(), outs=(), label="", **kw):
        return self._add(Op(kind=kind, binding=binding, ins=tuple(ins), outs=tuple(outs), label=label or binding, **kw))

    # ---- control ------------------------------------------------------------
    def wait(
        self, vm: Optional[int] = None, lgkm: Optional[int] = None, exp: Optional[int] = None, label="", **counters
    ):
        """``s_waitcnt``; ``counters`` names split counters of gfx12+ targets (``ds=0``, ``tensor=1``, ...)."""
        fields = {"vm": vm, "lgkm": lgkm, "exp": exp, **counters}
        parts = [f"{k}cnt({v})" for k, v in fields.items() if v is not None]
        return self._add(Op(kind="wait", label=label or "s_waitcnt " + " ".join(parts), args=fields))

    def barrier(self, label="s_barrier", fence: str = "both", vm=None, lgkm=None, asm: bool = False):
        """Workgroup ``s_barrier``.

        ``fence``: ``sched_barrier(0)`` around it -- "both" | "before" | "after" | "none".
        ``vm``/``lgkm``: an ``s_waitcnt`` issued immediately before the barrier.
        ``asm``: emit wait+barrier as one opaque inline-asm block (invisible to LLVM's
        waitcnt pass), as hand-ported kernels often do.
        """
        assert fence in ("both", "before", "after", "none")
        if vm is not None or lgkm is not None:
            label = (
                f"{label} (s_waitcnt"
                + (f" vmcnt({vm})" if vm is not None else "")
                + (f" lgkmcnt({lgkm})" if lgkm is not None else "")
                + ")"
            )
        return self._add(Op(kind="barrier", label=label, args={"fence": fence, "vm": vm, "lgkm": lgkm, "asm": asm}))

    def sched_barrier(self, mask: int = 0, label="sched_barrier"):
        return self._add(Op(kind="sched_barrier", label=label, args={"mask": mask}))

    def sched_group(self, cls: str, size: int, group: int, label=""):
        return self._add(
            Op(
                kind="sched_group",
                label=label or f"sg[{group}] {size}x{cls}",
                args={"cls": cls, "size": size, "group": group},
            )
        )

    def interleave(self, pairs: int, other_cls: str, other_cnt: int, group: int):
        """``pairs`` x {1 MFMA, ``other_cnt`` x ``other_cls``} sched groups (the expert's co-issue pattern)."""
        for _ in range(pairs):
            self.sched_group("mfma", 1, group)
            self.sched_group(other_cls, other_cnt, group)

    def setprio(self, prio: int):
        return self._add(Op(kind="setprio", label=f"setprio {prio}", args={"prio": prio}))

    def nop(self, n: int, asm: bool = True):
        """``s_nop n``; ``asm`` emits inline asm (opaque to the scheduler) instead of the intrinsic."""
        return self._add(Op(kind="nop", label=f"s_nop {n}", args={"n": n, "asm": asm}))

    def stagger(self, phase: str):
        """Dual-wave phase shift: group 1 runs an extra barrier at "open", group 0 at "close".

        Lowered through the kernel binding ``stagger_<phase>`` (it needs the wave's group id)."""
        assert phase in ("open", "close")
        return self._add(
            Op(kind="stagger", binding=f"stagger_{phase}", label=f"stagger {phase}", args={"phase": phase})
        )

    def anchor(self, binding: str, ins: Sequence[str], outs: Sequence[str]):
        return self._add(Op(kind="anchor", binding=binding, ins=tuple(ins), outs=tuple(outs), label=binding))

    def extend(self, ops: Sequence[Op]):
        for op in ops:
            self._add(op)
