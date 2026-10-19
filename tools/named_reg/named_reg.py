#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile a FlyDSL kernel with named-register pins through a pin-capable LLVM and run it (opt-in).

Pins (``flydsl.sched.pins``) reach LLVM IR as inline-asm markers that the stock
toolchain compiles as no-ops.  This tool takes the per-kernel LLVM IR FlyDSL dumps
(``NN_llvm_ir.ll``, the input of ``gpu-module-to-binary``), rewrites the markers into
``llvm.read_register`` / ``llvm.write_register`` (``flydsl/compiler/named_reg.py``),
compiles it with the ``opt``/``llc``/``ld.lld`` of an LLVM built from the AMDGPU
named-register branch, and runs the result through a copy of the JIT cache entry
(``tools/inject_asm``), with no FlyDSL rebuild::

    named_reg.py prepare WS -- python bench.py          # dump + cache one kernel (inject_asm prepare)
    named_reg.py build WS plain --llvm $L/bin           # unmodified IR: reproduces FlyDSL's ISA?
    named_reg.py build WS strip --llvm $L/bin --llvm-opt enable-post-misched=false
    named_reg.py build WS pin   --llvm $L/bin --llvm-opt enable-post-misched=false
    named_reg.py run WS pin                             # the prepared command, RUN_ONLY, pinned binary
    named_reg.py run WS strip -- python other_bench.py  # another command hitting the same entry

``build`` writes ``WS/<variant>/{input.ll,<variant>.s,<variant>.hsaco,report.json}`` and a
patched cache ``WS/<variant>/cache``.  The pipeline mirrors gpu-module-to-binary:
``opt -O2`` then ``llc -O2`` for ``amdgcn-amd-amdhsa``, ``ld.lld -shared``; the kernel's
``llvm_options`` compile hints are not in the dump and must be passed with ``--llvm-opt``
(to both ``opt`` and ``llc``).  Kernels that need device libraries (``__ocml_*``) are
refused: they are linked inside gpu-module-to-binary, not present in the dump.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


named_reg = _load("flydsl_named_reg", _REPO / "python" / "flydsl" / "compiler" / "named_reg.py")
inject_asm = _load("inject_asm", _REPO / "tools" / "inject_asm" / "inject_asm.py")

_RES_RE = re.compile(r"; (NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy|NumSgprs|codeLenInByte)[:=] *(\d+)")


def _instrs(asm: str) -> list:
    out = []
    for line in asm.splitlines():
        s = line.split(";", 1)[0].strip()
        if s and not s.startswith(".") and not s.endswith(":"):
            out.append(s)
    return out


def _sh(cmd, what):
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"named_reg: {what} failed ({' '.join(map(str, cmd))}):\n{proc.stderr.strip()}")
    return proc


def _one(ws: Path, pattern: str) -> Path:
    found = sorted((ws / "dump").rglob(pattern))
    if len(found) != 1:
        sys.exit(f"named_reg: expected one {pattern} under {ws / 'dump'}, found {[str(p) for p in found]}")
    return found[0]


def build(ws: Path, variant: str, llvm_bin: Path, mode: str, llvm_opts: list, mcpu: str) -> dict:
    ws = ws.expanduser().resolve()
    out = ws / variant
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ll = _one(ws, "*_llvm_ir.ll")
    text = ll.read_text(encoding="utf-8")
    if "@__ocml_" in text or "@__ockl_" in text:
        sys.exit("named_reg: the kernel calls device-library functions, which the dumped IR does not contain")
    report = {"variant": variant, "mode": mode, "source_ll": str(ll), "llvm_bin": str(llvm_bin), "llvm_opts": llvm_opts}
    regs = []
    if mode != "plain":
        res = named_reg.rewrite_markers(text, mode=mode)
        text = res.text
        regs = sorted(res.regs)
        report.update(markers=len(res.markers), reads=res.reads, writes=res.writes, regs=res.regs)
    (out / "input.ll").write_text(text, encoding="utf-8")

    cl = [f"--{o}" for o in llvm_opts]
    tgt = ["-mtriple=amdgcn-amd-amdhsa", f"-mcpu={mcpu}"]
    _sh([llvm_bin / "opt", "-O2", *tgt, *cl, out / "input.ll", "-o", out / "opt.bc"], "opt")
    asm, obj, hsaco = out / f"{variant}.s", out / f"{variant}.o", out / f"{variant}.hsaco"
    _sh([llvm_bin / "llc", "-O2", *tgt, *cl, out / "opt.bc", "-o", asm], "llc (asm)")
    _sh([llvm_bin / "llc", "-O2", *tgt, *cl, "-filetype=obj", out / "opt.bc", "-o", obj], "llc (obj)")
    _sh([llvm_bin / "ld.lld", "-shared", obj, "-o", hsaco], "ld.lld")

    asm_text = asm.read_text(encoding="utf-8")
    report["resources"] = {k: int(v) for k, v in _RES_RE.findall(asm_text)}
    ref = _one(ws, "*_final_isa.s")
    a, b = _instrs(ref.read_text(encoding="utf-8")), _instrs(asm_text)
    same = sum(x == y for x, y in zip(a, b))
    report["vs_flydsl_isa"] = {"flydsl_instrs": len(a), "instrs": len(b), "same_at_position": same}
    hazards = named_reg.pending_lds_uses(asm_text)
    report["pending_lds_uses"] = {"count": len(hazards), "first": hazards[:4]}
    if hazards:
        print(
            f"named_reg: WARNING {variant}: {len(hazards)} instruction(s) touch an LDS-read destination before its "
            f"lgkmcnt wait (ScratchSize {report['resources'].get('ScratchSize')}); results are likely wrong",
            file=sys.stderr,
        )
    if regs and mode == "pin":
        report["pins"] = {r: rep.summary() for r, rep in named_reg.isa_pin_report(asm_text, regs).items()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _patch_cache(ws, out, hsaco.read_bytes())
    return report


def _patch_cache(ws: Path, out: Path, code_object: bytes):
    src = ws / "cache"
    dst = out / "cache"
    shutil.copytree(src, dst)
    entries = inject_asm.load_entries(dst)
    if len(entries) != 1:
        sys.exit(f"named_reg: expected one cache entry under {dst}, found {len(entries)}")
    path, artifact, _ = entries[0]
    from flydsl.compiler.jit_function import JitCacheManager

    artifact._ir_text, _, _ = inject_asm.replace_binary(artifact._ir_text, None, code_object=code_object)
    JitCacheManager._write_cache_file(path, artifact)


def run(ws: Path, variant: str, cmd: list) -> int:
    ws = ws.expanduser().resolve()
    config = json.loads((ws / "prepare.json").read_text())
    if cmd:
        config = dict(config, cmd=cmd)
    cache = ws / "cache" if variant == "flydsl" else ws / variant / "cache"
    return inject_asm._run_command(
        config, drop=("COMPILE_ONLY",), FLYDSL_RUNTIME_CACHE_DIR=str(cache), FLYDSL_RUNTIME_RUN_ONLY="1"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    cmd = []
    if "--" in argv:
        cut = argv.index("--")
        argv, cmd = argv[:cut], argv[cut + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="dump and cache the command into a fresh workspace (inject_asm prepare)")
    p.add_argument("workspace", type=Path)
    b = sub.add_parser("build", help="rewrite + compile the dumped LLVM IR and patch a copy of the cache")
    b.add_argument("workspace", type=Path)
    b.add_argument("variant", help="output name; also selects the rewrite unless --mode is given")
    b.add_argument("--llvm", type=Path, required=True, help="bin directory with opt, llc, ld.lld")
    b.add_argument("--mode", choices=("plain", "strip", "pin"), help="default: the variant name if it is one")
    b.add_argument("--llvm-opt", action="append", default=[], help="cl::opt for opt and llc, e.g. lsr-drop-solution=1")
    b.add_argument("--mcpu", default=os.environ.get("FLYDSL_GPU_ARCH", "gfx950"))
    r = sub.add_parser("run", help="run the prepared command (or the one after --) on a variant, RUN_ONLY")
    r.add_argument("workspace", type=Path)
    r.add_argument("variant", help="a built variant, or 'flydsl' for the compiler's own binary")
    args = ap.parse_args(argv)
    if args.command == "prepare":
        if not cmd:
            ap.error("prepare takes the command after --")
        inject_asm.prepare(args.workspace, cmd)
        return 0
    if args.command == "build":
        if cmd:
            ap.error("build takes no command")
        mode = args.mode or (args.variant if args.variant in ("plain", "strip", "pin") else None)
        if mode is None:
            ap.error("--mode is required for a variant not named plain/strip/pin")
        rep = build(args.workspace, args.variant, args.llvm.expanduser().resolve(), mode, args.llvm_opt, args.mcpu)
        print(json.dumps(rep, indent=2))
        return 0
    return run(args.workspace, args.variant, cmd)


if __name__ == "__main__":
    sys.exit(main())
