#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Check register pins on a dumped ISA (``FLYDSL_DUMP_IR=1`` writes ``*_final_isa.s``).

    python tools/named_reg/isa_report.py dump/<kernel>/22_final_isa.s v[192:207] v190 [--json]

Prints the kernel's register/scratch use, where MFMA accumulators actually live, per
requested range the instructions that write and read it, MFMAs that accumulate in place
and the register copies into and out of it (``PIN.INDIRECT``), and any instruction that
touches an LDS-read destination before its ``lgkmcnt`` wait (``pins.pending_lds_uses``).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from flydsl.sched import pins

_RES_RE = re.compile(
    r"\.(?:amdhsa_)?(next_free_vgpr|next_free_sgpr|accum_offset|private_segment_fixed_size|vgpr_count|agpr_count|"
    r"sgpr_count|vgpr_spill_count|sgpr_spill_count)[:]? +(\d+)"
)


def report(asm_text: str, regs) -> dict:
    hazards = pins.pending_lds_uses(asm_text)
    return {
        "resources": {k: int(v) for k, v in _RES_RE.findall(asm_text)},
        "mfma_accumulators": dict(pins.mfma_accumulators(asm_text)),
        "pins": {r: rep.summary() for r, rep in pins.isa_pin_report(asm_text, regs).items()},
        "pending_lds_uses": {"count": len(hazards), "first": hazards[:4]},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("asm", type=Path)
    ap.add_argument("regs", nargs="*", help="requested ranges, e.g. v[192:207] a[0:3] v190")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = report(args.asm.read_text(encoding="utf-8"), args.regs)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print("resources:", rep["resources"])
    acc = rep["mfma_accumulators"]
    print("mfma accumulators:", dict(sorted(acc.items(), key=lambda kv: -kv[1])[:8]), "..." if len(acc) > 8 else "")
    for reg, r in rep["pins"].items():
        print(
            f"{reg}: placed={r['placed']} in_place={r['in_place']} copies in/out={r['copies_in']}/{r['copies_out']} "
            f"by block={r['copy_blocks']} writers={r['writers']}"
        )
    pending = rep["pending_lds_uses"]
    if pending["count"]:
        print(f"WARNING: {pending['count']} instruction(s) touch an LDS-read destination before its wait:")
        for line in pending["first"]:
            print("   ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
