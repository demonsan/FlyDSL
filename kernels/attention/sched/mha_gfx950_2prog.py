# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 dual-wave MHA as two per-group programs (``Schedule.group_sections``).

The single-program schedule (``mha_gfx950.build_mha_schedule``) runs one program
in both wave groups and creates the one-cluster phase offset at run time: the
``stagger open`` / ``stagger close`` ops lower to a barrier predicated on the
group id (an ``scf.if`` in the prologue, an inline-asm ``s_cbranch`` in the
epilogue), so group 1 crosses one extra barrier at the start and group 0 one at
the end.

Here the offset is code: each group gets its own program, emitted in its own
branch on the wave-uniform group id (as the aiter ``.co`` has two loop copies).

* group 0: the stagger-open barrier is absent, the stagger-close barrier is a
  plain barrier at the end;
* group 1: the stagger-open barrier is a plain barrier in its prologue, the
  stagger-close barrier is absent.

Both groups execute the same number of barriers, and barrier ``k`` pairs group 0
cluster ``c`` with group 1 cluster ``c - 1`` exactly as in the stagger
schedule, so the verifier proves the same LDS protocol (it now checks the
cross-group RAW/WAR with each group's own barrier indices).

Priorities (``prio``):

* ``"toggle"`` keeps the single schedule's ``s_setprio 1`` / ``0`` around the
  P*V clusters in both programs;
* ``"static"`` drops the toggles and gives each group one priority for the
  whole program (``static_prio``, default group 0 -> 0, group 1 -> 1), the
  ``s_setprio 0`` / ``s_setprio 1`` the ``.co``'s two loop copies start with.
  Which group should win issue is an empirical question; time both.
"""

from typing import List, Tuple

from flydsl.sched.ir import Op, Schedule, Section
from kernels.attention.sched.mha_gfx950 import build_mha_schedule

PRIO_MODES = ("toggle", "static")


def _barrier(label: str) -> Op:
    return Op(kind="barrier", label=label, args={"fence": "both", "vm": None, "lgkm": None, "asm": False})


def _group_ops(ops: List[Op], g: int, keep_setprio: bool) -> List[Op]:
    out = []
    for op in ops:
        if op.kind == "stagger":
            phase = op.args["phase"]
            if (phase == "open" and g == 1) or (phase == "close" and g == 0):
                out.append(_barrier(f"s_barrier (group {g} phase offset, {phase})"))
            continue
        if op.kind == "setprio" and not keep_setprio:
            continue
        out.append(op)
    return out


def _group_program(sections: List[Section], g: int, prio: str, static_prio: Tuple[int, int]) -> List[Section]:
    out = []
    for i, sec in enumerate(sections):
        ops = _group_ops(sec.ops, g, keep_setprio=prio == "toggle")
        if prio == "static" and i == 0:
            p = static_prio[g]
            ops = [Op(kind="setprio", label=f"setprio {p} (group {g})", args={"prio": p})] + ops
        out.append(Section(f"g{g}.{sec.name}", ops, loop=sec.loop, carried=sec.carried))
    return out


def build_mha_2prog_schedule(
    head_dim: int = 128,
    causal: bool = True,
    dtype: str = "bf16",
    lazy_rescale: bool = True,
    prio: str = "toggle",
    static_prio: Tuple[int, int] = (0, 1),
) -> Schedule:
    if prio not in PRIO_MODES:
        raise ValueError(f"prio must be one of {PRIO_MODES}, got {prio!r}")
    base = build_mha_schedule(head_dim, causal, dtype, lazy_rescale=lazy_rescale, setprio=True, stagger=True)
    programs = [_group_program(base.sections, g, prio, static_prio) for g in range(base.groups)]
    meta = dict(base.meta)
    meta["expert_intent"] = list(base.meta.get("expert_intent", ())) + [
        "per-group programs: the one-cluster phase offset is a barrier in group 1's prologue and group 0's "
        "epilogue instead of a runtime-predicated stagger barrier",
        f"priorities: {prio}"
        + (f" (group 0 -> {static_prio[0]}, group 1 -> {static_prio[1]})" if prio == "static" else ""),
    ]
    tag = "2prog" if prio == "toggle" else f"2prog_static{static_prio[0]}{static_prio[1]}"
    return Schedule(
        name=f"{base.name}_{tag}",
        target=base.target,
        buffers=base.buffers,
        queues=base.queues,
        sections=programs[0],
        groups=base.groups,
        stagger=False,
        waves_per_group=base.waves_per_group,
        bases=base.bases,
        valid_tiles=base.valid_tiles,
        tensors=base.tensors,
        meta=meta,
        group_sections=programs,
    )
