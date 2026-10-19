# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Emit a whole schedule -- straight-line sections and ``scf.for`` loops -- in a kernel.

``lower.emit`` lowers one op list.  Cross-block transformations
(:mod:`flydsl.sched.transform`) change which values a loop carries, what the
prologue/epilogue contain, and where ops sit relative to the back-edge, so the
kernel must not hard-code that structure.  :func:`emit_program` builds the loop
from the schedule itself:

* loop-carried values are ``Section.carried``; each value may be an IR value,
  an ``fx`` numeric, or nested lists/tuples of those (flattened into iter_args
  and rebuilt in the body);
* an optional per-value codec ``(pack, unpack)`` reproduces a kernel's own
  packing (e.g. a pair of vectors carried as one wider vector);
* the kernel supplies only the runtime loop bounds and the tile resolver.

The loop is built with the same ``scf_range`` / ``scf_yield_`` helpers the AST
rewriter generates for ``for ... in range(..., init=...)``, so this module can
be called from anywhere inside a ``@flyc.kernel`` body.
"""

from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .ir import Schedule, TileRef
from .lower import emit

Codec = Tuple[Callable[[object], object], Callable[[object], object]]


def _flatten(value, out: List[object]):
    """Append the leaves of ``value``; return its shape (int leaf count tree)."""
    if isinstance(value, (list, tuple)):
        return (type(value), [_flatten(v, out) for v in value])
    out.append(value)
    return None


def _rebuild(shape, it):
    if shape is None:
        return next(it)
    kind, parts = shape
    return kind(_rebuild(p, it) for p in parts)


def flatten(values: Sequence[object]):
    leaves: List[object] = []
    shapes = [_flatten(v, leaves) for v in values]
    return leaves, shapes


def unflatten(leaves: Sequence[object], shapes) -> List[object]:
    it = iter(leaves)
    out = [_rebuild(s, it) for s in shapes]
    rest = list(it)
    if rest:
        raise ValueError(f"{len(rest)} loop values left over after rebuilding the carried state")
    return out


def schedule_fingerprint(sched: Schedule) -> str:
    """Stable hash of everything that changes emitted code (op order, args, slots, waits, carried)."""
    import hashlib

    h = hashlib.sha256()
    for sec in sched.sections:
        h.update(repr((sec.name, sec.loop is not None, tuple(sec.carried))).encode())
        for op in sec.ops:
            args = tuple(sorted((k, str(v)) for k, v in op.args.items()))
            h.update(repr((op.kind, op.binding, op.ins, op.outs, args, op.count)).encode())
    return h.hexdigest()[:16]


def emit_program(
    sched: Schedule,
    bindings: Mapping[str, Callable],
    tile_of: Callable[[Dict[str, object]], Callable[[TileRef], object]],
    loop_bounds: Callable[[str], Tuple[object, object, object]],
    env: Optional[MutableMapping[str, object]] = None,
    codecs: Optional[Mapping[str, Codec]] = None,
    sections: Optional[Sequence[str]] = None,
    loop_impl: Optional[Callable] = None,
    **emit_kw,
) -> MutableMapping[str, object]:
    """Lower every section of ``sched`` in order.

    ``tile_of(extra_bases)`` returns the kernel's TileRef resolver with ``extra_bases``
    (the loop induction variable) bound; ``loop_bounds(section_name)`` returns the
    runtime ``(start, stop, step)`` of a loop section.  ``loop_impl(start, stop, step,
    init, body)`` builds the loop (default: ``scf.for``); ``body(iv, args) -> new args``.
    ``emit_kw`` is forwarded to :func:`flydsl.sched.lower.emit` (``target``, ``primitives``, ...).
    """
    if sched.program is not None:
        raise ValueError(f"{sched.name}: tile-count dependent programs cannot be emitted as one loop nest")
    env = {} if env is None else env
    codecs = codecs or {}
    for sec in sched.sections:
        if sections is not None and sec.name not in sections:
            continue
        kw = dict(emit_kw, section=sec.name) if emit_kw.get("tag") is not None else emit_kw
        if sec.loop is None:
            emit(sec.ops, bindings, env, tile_of({}), **kw)
        else:
            _emit_loop(sec, bindings, env, tile_of, loop_bounds(sec.name), codecs, kw, loop_impl or scf_loop)
    return env


def scf_loop(start, stop, step, init, body):
    """``scf.for`` with iter_args, built with the AST rewriter's own helpers."""
    from flydsl.compiler.ast_rewriter import InsertEmptyYieldForSCFFor, ReplaceYieldWithSCFYield

    results = init
    for iv, args in InsertEmptyYieldForSCFFor.scf_range(start, stop, step, init=init):
        results = ReplaceYieldWithSCFYield.scf_yield_(body(iv, args))
    return list(results) if isinstance(results, (list, tuple)) else [results]


def python_loop(start, stop, step, init, body):
    """Run the loop eagerly in Python (CPU interpretation / tests)."""
    args = list(init)
    for iv in range(start, stop, step):
        args = list(body(iv, args))
    return args


def _emit_loop(sec, bindings, env, tile_of, bounds, codecs, emit_kw, loop_impl):
    var = sec.loop[0]
    names = list(sec.carried)
    missing = [n for n in names if n not in env]
    if missing:
        raise KeyError(f"loop {sec.name} carries {missing} but they are not defined before it")

    def pack(name, value):
        return codecs[name][0](value) if name in codecs else value

    def unpack(name, value):
        return codecs[name][1](value) if name in codecs else value

    leaves, shapes = flatten([pack(n, env[n]) for n in names])

    def body_fn(iv, args):
        body = dict(env)
        for n, v in zip(names, unflatten(list(args), shapes)):
            body[n] = unpack(n, v)
        emit(sec.ops, bindings, body, tile_of({var: iv}), **emit_kw)
        out, _ = flatten([pack(n, body[n]) for n in names])
        if len(out) != len(leaves):
            raise ValueError(f"loop {sec.name}: carried state changed size ({len(leaves)} -> {len(out)} values)")
        return out

    start, stop, step = bounds
    results = loop_impl(start, stop, step, leaves, body_fn)
    for n, v in zip(names, unflatten(list(results), shapes)):
        env[n] = unpack(n, v)
