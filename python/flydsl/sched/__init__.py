# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""flydsl.sched -- an explicit, verifiable Schedule IR for hand-tuned kernels.

The expert's plan -- which buffer holds which tile in which stage, which
async queue each copy lands on, where each wait and barrier sits, how MFMAs
interleave with VALU/TRANS work, which priority a wave runs at -- is written
as data (:mod:`.ir`).  From that data:

* :func:`.verify.verify` simulates every wave group and reports hazards,
  stale tiles, unsatisfiable scheduler hints and hardware-contract issues;
* :func:`.resource.estimate_resources` / :func:`.cost.estimate_cost` give
  LDS / VGPR / occupancy and an MFMA-bound + roofline estimate;
* :func:`.lower.emit` lowers the same op list inside a ``@flyc.kernel``.

Hardware facts come from ``targets/<arch>.json`` with their sources; unknown
facts are ``null`` and are reported as unknown, never guessed.  The package
is pure Python and imports without MLIR.  CLI: ``python -m flydsl.sched``.
"""

from .ir import Buffer, Builder, Op, Queue, Schedule, Section, SlotUse, T, TileRef
from .target import TargetDesc, UnknownFact, load_target
from .verify import Finding, Report, trace, verify

__all__ = [
    "Buffer",
    "Builder",
    "Finding",
    "Op",
    "Queue",
    "Report",
    "Schedule",
    "Section",
    "SlotUse",
    "T",
    "TargetDesc",
    "TileRef",
    "UnknownFact",
    "load_target",
    "trace",
    "verify",
]
