# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Emit a schedule section inside a ``@flyc.kernel`` body.

``emit`` walks the op list in order.  Control ops (waits, barriers,
scheduler hints, priorities) are lowered here, from the target
description, so the kernel author never writes a raw ``s_waitcnt``
bitfield or ``sched_group_barrier`` mask.  Every other op calls the kernel
binding named by ``op.binding`` with its ``ins`` (looked up in ``env``) and
``args`` (tile references resolved by the kernel's ``tile`` callback); the
results are stored back into ``env`` under ``op.outs``.

The primitives object is injectable so the emitted call sequence can be
recorded and compared on a machine without ROCm.
"""

from contextlib import contextmanager
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence

from .ir import Op, Section, TileRef
from .target import TargetDesc, load_target


class RocdlPrimitives:
    """Control-op lowering for CDNA targets through ``flydsl.expr.rocdl``."""

    def __init__(self, target: TargetDesc):
        self.target = target

    @staticmethod
    def _rocdl():
        from flydsl.expr import rocdl  # lazy: keep flydsl.sched importable without MLIR

        return rocdl

    def wait(self, vm=None, lgkm=None, exp=None):
        rocdl = self._rocdl()
        if vm == 0 and lgkm == 0 and exp == 0:
            rocdl.s_waitcnt(0)
        else:
            rocdl.s_waitcnt(vmcnt=vm, lgkmcnt=lgkm, expcnt=exp)

    def barrier(self, fence="both", vm=None, lgkm=None, asm=False):
        rocdl = self._rocdl()
        if fence in ("both", "before"):
            rocdl.sched_barrier(0)
        if asm:
            from flydsl._mlir.dialects import llvm

            cnt = []
            if vm is not None:
                cnt.append(f"vmcnt({vm})")
            if lgkm is not None:
                cnt.append(f"lgkmcnt({lgkm})")
            parts = (["s_waitcnt " + " ".join(cnt)] if cnt else []) + ["s_barrier"]
            llvm.inline_asm(None, [], "\n".join(parts), "", has_side_effects=True)
        else:
            if vm is not None or lgkm is not None:
                self.wait(vm, lgkm, None)
            rocdl.s_barrier()
        if fence in ("both", "after"):
            rocdl.sched_barrier(0)

    def sched_barrier(self, mask=0):
        self._rocdl().sched_barrier(mask)

    def sched_group(self, cls, size, group):
        self._rocdl().sched_group_barrier(self.target.sched_mask(cls), size, group)

    def setprio(self, prio):
        self._rocdl().s_setprio(prio)

    def nop(self, n, asm=True):
        if not isinstance(n, int) or not 0 <= n <= 15:
            raise ValueError("s_nop immediate must be a Python int in [0, 15]")
        if not asm:
            self._rocdl().s_nop(n)
            return
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import llvm

        llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"s_nop {n}", "", has_side_effects=True)


def _resolve(value, tile: Callable[[TileRef], object]):
    return tile(value) if isinstance(value, TileRef) else value


# Location tags.  With ``emit(..., tag=...)`` every MLIR op built for schedule op
# ``i`` of a section carries ``FileLineColLoc("sched:<tag>/<section>", i + 1, 0)``.
# With ``FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`` that location reaches the DWARF line
# table, so ``.file``/``.loc`` in the final ISA name the schedule op of every
# instruction (``flydsl.sched.conform`` reads them back).  Line numbers are 1-based
# because DWARF line 0 means "no source line".
TAG_SCHEME = "sched:"


def tag_file(tag: str, section: Optional[str] = None) -> str:
    """The pseudo source file name ``emit`` uses for ``section`` of schedule ``tag``."""
    return f"{TAG_SCHEME}{tag}/{section}" if section else f"{TAG_SCHEME}{tag}"


@contextmanager
def mlir_op_location(filename: str, line: int):
    """Make ``FileLineColLoc(filename, line, 0)`` the location of every op built inside.

    DSL primitives decorated with ``dsl_loc_tracing`` normally replace the ambient
    location with the user's Python call site; marking the scope as already
    captured (``meta._tls.active_loc``) makes them keep this one instead.
    """
    from flydsl._mlir import ir
    from flydsl.expr import meta

    loc = ir.Location.file(filename, line, 0)
    prev = getattr(meta._tls, "active_loc", None)
    meta._tls.active_loc = loc
    try:
        with loc:
            yield
    finally:
        meta._tls.active_loc = prev


def emit(
    ops: Sequence[Op],
    bindings: Mapping[str, Callable],
    env: MutableMapping[str, object],
    tile: Callable[[TileRef], object] = lambda ref: ref,
    target: Optional[TargetDesc] = None,
    primitives=None,
    trace: Optional[list] = None,
    *,
    tag: Optional[str] = None,
    section: Optional[str] = None,
    location: Callable = mlir_op_location,
):
    """Lower ``ops`` in order; returns ``env`` (updated in place).

    ``tag`` (usually the schedule name) attaches ``sched:<tag>/<section>:<index + 1>``
    as the MLIR location of everything emitted for each op; ``section`` defaults to
    the name of ``ops`` when it is a ``Section``.  ``location(filename, line)`` is the
    context manager that applies it (injectable for tests without MLIR).
    """
    if isinstance(ops, Section):
        section = section or ops.name
        ops = ops.ops
    target = target or load_target("gfx950")
    prim = primitives or RocdlPrimitives(target)
    filename = tag_file(tag, section) if tag is not None else None
    for i, op in enumerate(ops):
        if filename is None:
            _emit_one(op, prim, bindings, env, tile, trace)
        else:
            with location(filename, i + 1):
                _emit_one(op, prim, bindings, env, tile, trace)
    return env


def _emit_one(op: Op, prim, bindings, env, tile, trace):
    if trace is not None:
        trace.append(op)
    k = op.kind
    if k == "wait":
        prim.wait(op.args.get("vm"), op.args.get("lgkm"), op.args.get("exp"))
    elif k == "barrier":
        prim.barrier(op.args.get("fence", "both"), op.args.get("vm"), op.args.get("lgkm"), op.args.get("asm", False))
    elif k == "sched_barrier":
        prim.sched_barrier(op.args.get("mask", 0))
    elif k == "sched_group":
        prim.sched_group(op.args["cls"], op.args["size"], op.args["group"])
    elif k == "setprio":
        prim.setprio(op.args["prio"])
    elif k == "nop":
        prim.nop(op.args["n"], op.args.get("asm", True))
    else:
        _call(op, bindings, env, tile)


def _call(op: Op, bindings: Mapping[str, Callable], env: MutableMapping[str, object], tile):
    if op.binding is None:
        raise ValueError(f"op {op} ({op.kind}) has no binding")
    fn = bindings.get(op.binding)
    if fn is None:
        raise KeyError(f"no binding {op.binding!r} for op {op}; kernel provides: {sorted(bindings)}")
    missing = [n for n in op.ins if n not in env]
    if missing:
        raise KeyError(f"op {op} reads undefined values {missing}")
    # stagger's phase is already in its binding name (stagger_open / stagger_close)
    args = {} if op.kind == "stagger" else {k: _resolve(v, tile) for k, v in op.args.items()}
    result = fn(*[env[n] for n in op.ins], **args)
    if not op.outs:
        return
    if len(op.outs) == 1:
        env[op.outs[0]] = result
        return
    if not isinstance(result, (tuple, list)) or len(result) != len(op.outs):
        raise ValueError(f"binding {op.binding!r} must return {len(op.outs)} values for {op.outs}")
    for name, value in zip(op.outs, result):
        env[name] = value


def bindings_used(ops: Sequence[Op]) -> Dict[str, int]:
    """Bindings a kernel must provide for ``ops`` (name -> use count)."""
    out: Dict[str, int] = {}
    for op in ops:
        if op.kind in ("wait", "barrier", "sched_barrier", "sched_group", "setprio", "nop"):
            continue
        out[op.binding] = out.get(op.binding, 0) + 1
    return out
