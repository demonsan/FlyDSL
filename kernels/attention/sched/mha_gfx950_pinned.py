# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 dual-wave MHA schedule with register pins on loop-carried values.

Built from the base schedule (``mha_gfx950.build_mha_schedule``): the KV loop
gets ``Section.pins`` and the schedule a new name, nothing else changes.  The
kernel (``flash_attn_sched_gfx950.build_flash_attn_sched_gfx950_module(...,
schedule=...)``) lowers the pins through ``emit_program`` to register-allocation
hints (``rocdl.sched.vgpr``, see ``flydsl.sched.pins``).

Presets (D=128, waves_per_eu=2, 256 arch VGPRs per wave); ``*_soft`` are the same
ranges as advisory hints:

* ``"o"``: the O accumulator (``v_o``, 4 x f32x16 = 64 dwords) in ``v[192:255]``;
* ``"ml"``: the softmax row max / row sum (``m_row``, ``l_row``, 1 dword each) in
  ``v190`` / ``v191``.
"""

import copy
from typing import Dict, Optional

from kernels.attention.sched.mha_gfx950 import build_mha_schedule

PRESETS: Dict[str, Dict[str, str]] = {
    "o": {"v_o": "v[192:255]"},
    "ml": {"m_row": "v190", "l_row": "v191"},
}
PRESETS.update({f"{k}_soft": {v: f"{r} soft" for v, r in t.items()} for k, t in list(PRESETS.items())})


def build_pinned_mha_schedule(
    pins="o", head_dim: int = 128, causal: bool = False, dtype: str = "bf16", base=None, **base_kw
):
    """``pins``: a preset name or ``{carried value: register range}``; ``base``: schedule to pin (default: built)."""
    table: Optional[Dict[str, str]] = PRESETS[pins] if isinstance(pins, str) else dict(pins)
    sched = copy.copy(base if base is not None else build_mha_schedule(head_dim, causal, dtype, **base_kw))
    sections = []
    for sec in sched.sections:
        if sec.loop is not None:
            sec = copy.copy(sec)
            sec.pins = {**getattr(sec, "pins", {}), **table}
        sections.append(sec)
    sched.sections = sections
    tag = pins if isinstance(pins, str) else "custom"
    sched.name = f"{sched.name}_pin_{tag}"
    return sched
