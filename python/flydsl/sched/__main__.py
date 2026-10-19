# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Agent-facing CLI: ``python -m flydsl.sched <command> <module:builder> [options]``.

Commands
  check      verifier findings (exit 1 on errors)
  resource   LDS / VGPR / occupancy estimate
  cost       MFMA bound, dual-wave phase table, roofline for --shape
  show       the op list of one section (or every section)
  target     the target description, with sources and confidence
  conform    check a final ISA (*_final_isa.s) against its schedule; --against diffs two ISAs
  calibrate  per-binding instruction counts from a tagged final ISA (--schedule, --out file.json)
  issue      resolved issue programs of a schedule (slots, gaps, findings); --calib for unresolved ones

Examples
  python -m flydsl.sched check  kernels.attention.sched.mha_gfx950:build_mha_schedule --arg head_dim=64
  python -m flydsl.sched cost   kernels.attention.sched.mla_gfx950:build_mla_schedule --shape num_seqs=64,kv_len=8192
  python -m flydsl.sched target gfx950 --json
  python -m flydsl.sched conform emitted_final_isa.s \\
      --schedule kernels.attention.sched.mha_gfx950:build_mha_schedule --arg causal=False --against hand_final_isa.s
"""

import argparse
import importlib
import json
import sys
from dataclasses import asdict

from .cost import estimate_cost
from .resource import estimate_resources
from .target import load_target
from .verify import verify


def _kv(text):
    out = {}
    for item in filter(None, (text or "").split(",")):
        k, v = item.split("=", 1)
        if v.startswith("@"):  # k=@file.json: the JSON file's content (e.g. issue=@calibration.json)
            with open(v[1:], encoding="utf-8") as f:
                out[k] = json.load(f)
            continue
        try:
            out[k] = json.loads(v.lower() if v in ("True", "False") else v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _load(spec, args):
    mod, _, fn = spec.partition(":")
    builder = getattr(importlib.import_module(mod), fn or "build_schedule")
    kw = {}
    for a in args or []:
        kw.update(_kv(a))
    return builder(**kw)


def _conform(a):
    from .conform import conform, diff_isa, op_table, read_isa

    isa = read_isa(a.spec, a.kernel)
    sched = _load(a.schedule, a.arg) if a.schedule else None
    rc = 0
    if isa.tagged or sched is not None:
        r = conform(isa, sched, tag=a.tag)
        print(r.to_json() if a.json else r.render(verbose=a.verbose))
        rc = 0 if r.ok else 1
        if a.ops:
            print("\n".join(op_table(isa, sched, a.tag)))
    if a.against:
        other = read_isa(a.against)
        print(diff_isa(other, isa).render())
    return rc


# ---- issue programs (calibrate / issue) ---------------------------------------------------
def _calibrate(a):
    from .conform import read_isa
    from .issue import calibrate

    if not a.schedule:
        print("calibrate needs --schedule module:builder (the schedule the ISA was emitted from)")
        return 2
    cal = calibrate(read_isa(a.spec, a.kernel), _load(a.schedule, a.arg), a.tag, source=a.spec)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(cal.to_dict(), f, indent=1, sort_keys=True)
    if a.json:
        print(json.dumps(cal.to_dict(), indent=1, sort_keys=True))
        return 0
    for b, c in sorted(cal.by_binding.items()):
        var = f"   (varies: {cal.varies[b]})" if b in cal.varies else ""
        print(f"{b:24s} {' '.join(f'{k}={v}' for k, v in sorted(c.items()))}{var}")
    return 0


def _issue(a):
    from .issue import ISSUE_KIND, Calibration, resolve

    sched = _load(a.spec, a.arg)
    report = sched.meta.get("issue_report")
    if any(op.kind == ISSUE_KIND for op in sched.all_ops()):
        if not a.calib:
            print("schedule has unresolved issue programs; pass --calib calibration.json")
            return 2
        with open(a.calib, encoding="utf-8") as f:
            sched, report = resolve(sched, Calibration.from_dict(json.load(f)))
    if report is None:
        print(f"{sched.name}: no issue programs")
        return 0
    print(report.to_json() if a.json else report.render(verbose=True))
    return 0 if report.ok else 1


# ---- end issue programs -------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m flydsl.sched", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["check", "resource", "cost", "show", "target", "conform", "calibrate", "issue"])
    p.add_argument("--out", help="'calibrate': write the calibration JSON here")
    p.add_argument("--calib", help="'issue': calibration JSON for unresolved issue programs")
    p.add_argument("spec", help="module:builder for a schedule, an arch for 'target', an ISA file for 'conform'")
    p.add_argument("--schedule", help="module:builder of the schedule the ISA was emitted from ('conform')")
    p.add_argument("--against", help="second ISA file to diff against ('conform'; A = --against, B = spec)")
    p.add_argument("--kernel", help="kernel (function) name in the ISA ('conform'; default: the first)")
    p.add_argument("--tag", help="emit(tag=...) name to check ('conform'; default: the only one)")
    p.add_argument("--ops", action="store_true", help="'conform': print the per-op instruction table")
    p.add_argument("--arg", action="append", help="builder kwargs, k=v[,k=v]")
    p.add_argument("--tiles", help="tile counts for 'check', e.g. 4,6,8")
    p.add_argument("--shape", help="workload shape for 'cost', k=v[,k=v]")
    p.add_argument("--section", help="section for 'show'")
    p.add_argument("--verbose", action="store_true", help="include info findings")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    if a.command == "target":
        t = load_target(a.spec)
        facts = {k: asdict(t.fact(k)) for k in t._facts}
        if a.json:
            print(json.dumps({"target": t.key, "sources": t.sources, "facts": facts}, indent=2))
        else:
            for k, f in facts.items():
                print(f"{k:24s} [{f['confidence']:10s}] {json.dumps(f['value'])[:90]}  <- {f['source']}")
        return 0
    if a.command == "conform":
        return _conform(a)
    if a.command == "calibrate":
        return _calibrate(a)
    if a.command == "issue":
        return _issue(a)

    sched = _load(a.spec, a.arg)
    if a.command == "check":
        tiles = [int(x) for x in a.tiles.split(",")] if a.tiles else ()
        r = verify(sched, tiles)
        print(r.to_json() if a.json else r.render(verbose=a.verbose))
        return 0 if r.ok else 1
    if a.command == "resource":
        r = estimate_resources(sched)
        print(json.dumps(asdict(r), indent=2) if a.json else r)
        return 0
    if a.command == "cost":
        r = estimate_cost(sched, shape=_kv(a.shape) if a.shape else None)
        print(json.dumps(asdict(r), indent=2, default=str) if a.json else r)
        return 0
    if a.command == "show":
        secs = [sched.section(a.section)] if a.section else sched.sections
        for sec in secs:
            loop = f"  loop {sec.loop[0]}, carries {list(sec.carried)}" if sec.loop else ""
            print(f"== {sec.name}{loop}")
            for i, op in enumerate(sec.ops):
                extra = []
                if op.queue:
                    extra.append(f"q={op.queue}x{op.count}")
                extra += [f"R {u}" for u in op.reads] + [f"W {u}" for u in op.writes]
                print(f"  {i:4d} {op.kind:13s} {str(op):48s} {' '.join(extra)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
