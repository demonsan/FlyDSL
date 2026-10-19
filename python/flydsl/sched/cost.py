# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""First-order performance model: MFMA bound, cluster balance, roofline.

What it computes from documented facts only:

* MFMA cycles per cluster / per loop iteration / per KV tile, from the
  instruction pass counts in the target description;
* for a dual-wave schedule, which clusters share a SIMD in each phase and
  whether the matrix core is double-booked or idle in that phase;
* for a concrete shape (via ``meta["workload"]``), the MFMA-bound time and
  the HBM time with and without cache reuse.

What it does not compute: VALU/LDS/DMA/barrier time.  Their latencies are
``null`` in the target until calibrated, so the model says "unknown" rather
than inventing a number.  Treat the result as a lower bound used to rank
candidates; the GPU measurement is the ground truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir import Op, Schedule
from .target import TargetDesc, load_target


@dataclass
class Cluster:
    index: int
    mfma: int = 0
    mfma_cycles: int = 0
    dma: int = 0
    ds_read: int = 0
    label: str = ""

    @property
    def role(self):
        if self.mfma_cycles and (self.dma or self.ds_read):
            return "mixed"
        return "compute" if self.mfma_cycles else ("memory" if (self.dma or self.ds_read) else "sync")


@dataclass
class CostReport:
    schedule: str
    target: str
    loop_clusters: List[Cluster]
    mfma_cycles_per_iter_per_wave: int
    tiles_per_iter: int
    phases: List[dict] = field(default_factory=list)
    shape: Optional[dict] = None
    estimate: Optional[dict] = None
    unknown: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    # per-group programs: group -> {"clusters", "mfma_cycles_per_iter", "loop_offset"}; ``loop_clusters`` and
    # ``mfma_cycles_per_iter_per_wave`` then describe the busiest group (it bounds the wave pair)
    groups: Dict[int, dict] = field(default_factory=dict)

    @property
    def mfma_cycles_per_tile_per_wave(self):
        return self.mfma_cycles_per_iter_per_wave / self.tiles_per_iter

    def __str__(self):
        lines = [f"cost {self.schedule} @{self.target}"]
        lines.append(
            f"  loop: {len(self.loop_clusters)} clusters, {self.tiles_per_iter} tile(s)/iter, "
            f"{self.mfma_cycles_per_iter_per_wave} MFMA cyc/iter/wave "
            f"({self.mfma_cycles_per_tile_per_wave:.0f} cyc/tile/wave)"
        )
        for c in self.loop_clusters:
            lines.append(
                f"    C{c.index:<2d} {c.role:8s} mfma {c.mfma:3d} ({c.mfma_cycles:5d} cyc)  "
                f"dma {c.dma:2d}  ds_read {c.ds_read:3d}  {c.label}"
            )
        for g, info in sorted(self.groups.items()):
            lines.append(
                f"  group {g}: {len(info['clusters'])} clusters, {info['mfma_cycles_per_iter']} MFMA cyc/iter/wave, "
                f"loop starts after {info['loop_offset']} barrier(s): "
                + " ".join(f"C{c.index}:{c.role[0]}" for c in info["clusters"])
            )
        if self.phases:
            lines.append("  dual-wave phases (group A cluster | group B cluster on the same SIMD):")
            for p in self.phases:
                flag = {"ok": "", "contended": "  <-- matrix core double-booked", "idle": "  <-- matrix core idle"}[
                    p["status"]
                ]
                lines.append(f"    phase {p['phase']}: C{p['a']} | C{p['b']}  mfma {p['mfma_cycles']:5d} cyc{flag}")
        if self.estimate:
            e = self.estimate
            lines.append(f"  shape {self.shape}:")
            lines.append(
                f"    MFMA-bound   {e['t_mfma_us']:.1f} us  ({e['wg_rounds']} WG round(s) of {e['wg_slots']}; "
                f"{e['tflops_at_mfma_bound']:.0f} TFLOP/s issued, {e['useful_tflops']:.0f} useful, "
                f"{100 * e['useful_fraction']:.0f}% of issued MFMA work is useful)"
            )
            lines.append(f"    HBM unique   {e['t_hbm_unique_us']:.1f} us  ({e['hbm_bytes_unique'] / 1e6:.1f} MB)")
            lines.append(
                f"    HBM no-reuse {e['t_hbm_noreuse_us']:.1f} us  ({e['hbm_bytes_noreuse'] / 1e6:.1f} MB)  "
                "<- every WG streams K/V from HBM (upper bound; L2/MALL normally absorbs most)"
            )
            lines.append(f"    bound: {e['bound']}")
        for h in self.hints:
            lines.append(f"  hint: {h}")
        if self.unknown:
            lines.append(f"  not modelled (null in target): {', '.join(self.unknown)}")
        return "\n".join(lines)


def _mfma_rate_hints(ops: List[Op], target: TargetDesc) -> List[str]:
    """Point out MFMAs that run below the best rate the target offers for the same operand type."""
    table = target.require("mfma")
    cyc = target.require("mfma_cycles_per_pass")

    def rate_of(e):  # None when the entry lacks passes or flops (unknown stays unknown)
        return e["flops"] / (e["passes"] * cyc) if e.get("flops") and e.get("passes") else None

    best: Dict[str, tuple] = {}
    for name, e in table.items():
        rate = rate_of(e)
        if rate is not None and rate > best.get(e.get("ab"), (0, ""))[0]:
            best[e.get("ab")] = (rate, name)
    out = []
    for name in sorted({op.mfma for op in ops if op.kind == "mfma"}):
        e = table.get(name)
        rate = rate_of(e) if e else None
        if rate is None:
            out.append(f"{name}: rate unknown on {target.key} (passes/flops not described)")
            continue
        top, top_name = best.get(e.get("ab"), (rate, name))
        if top > rate:
            out.append(
                f"{name} runs at {rate:.0f} flop/clk/SIMD; {top_name} reaches {top:.0f} for {e.get('ab')} operands "
                f"({top / rate:.1f}x) -- {table[top_name].get('note', 'different operand layout')}"
            )
    return out


def _clusters(ops: List[Op], target: TargetDesc) -> List[Cluster]:
    out, cur = [], Cluster(0)
    for op in ops:
        if op.kind in ("barrier", "stagger"):
            if cur.mfma or cur.dma or cur.ds_read or out:
                out.append(cur)
            cur = Cluster(len(out))
            continue
        if op.kind == "mfma":
            cur.mfma += op.count
            cur.mfma_cycles += op.count * target.mfma_cycles(op.mfma)
            cur.label = cur.label or str(op)
        elif op.kind == "lds_dma":
            cur.dma += op.count
            cur.label = cur.label or str(op)
        elif op.kind in ("ds_read", "ds_read_tr"):
            cur.ds_read += op.count
    if cur.mfma or cur.dma or cur.ds_read:
        out.append(cur)
    return out


def _phase(k: int, a: Cluster, b: Cluster) -> dict:
    both = bool(a.mfma_cycles) and bool(b.mfma_cycles)
    none = not a.mfma_cycles and not b.mfma_cycles
    return {
        "phase": k,
        "a": a.index,
        "b": b.index,
        "mfma_cycles": a.mfma_cycles + b.mfma_cycles,
        "status": "contended" if both else ("idle" if none else "ok"),
    }


def _barriers(ops: List[Op], sched: Schedule, g: int) -> int:
    """Barriers group ``g`` executes in ``ops`` (stagger ops count as in the verifier)."""
    n = 0
    for op in ops:
        if op.kind == "barrier":
            n += 1
        elif op.kind == "stagger":
            phase = op.args["phase"]
            n += not sched.stagger or (phase == "open" and g == 1) or (phase == "close" and g == 0)
    return n


def _group_loops(sched: Schedule, target: TargetDesc, rep: CostReport):
    """Per-group programs: clusters per group, and the dual-wave pairing of their merged barrier phases.

    Group ``g``'s loop cluster ``k`` runs in barrier phase ``offset_g + it * n + k`` (``offset_g``: barriers
    before its loop, ``n``: barriers per iteration), so in group 0's slot ``k`` group 1 runs its cluster
    ``(k + offset_0 - offset_1) mod n``.  The pairing is only static when both loops cross the same number
    of barriers per iteration.
    """
    for g in range(sched.groups):
        secs = sched.sections_of(g)
        li = next((i for i, s in enumerate(secs) if s.loop is not None), None)
        loop = secs[li] if li is not None else secs[0]
        before = [op for s in secs[: li or 0] for op in s.ops]
        clusters = _clusters(loop.ops, target)
        rep.groups[g] = {
            "clusters": clusters,
            "mfma_cycles_per_iter": sum(c.mfma_cycles for c in clusters),
            "loop_offset": _barriers(before, sched, g),
            "barriers_per_iter": _barriers(loop.ops, sched, g),
            "ops": loop.ops,
        }
    busiest = max(rep.groups, key=lambda g: rep.groups[g]["mfma_cycles_per_iter"])
    rep.loop_clusters = rep.groups[busiest]["clusters"]
    rep.mfma_cycles_per_iter_per_wave = rep.groups[busiest]["mfma_cycles_per_iter"]
    if sched.groups != 2:
        return
    a, b = rep.groups[0], rep.groups[1]
    n = a["barriers_per_iter"]
    if n != b["barriers_per_iter"] or n != len(a["clusters"]) or n != len(b["clusters"]):
        rep.hints.append(
            f"group loops cross {a['barriers_per_iter']} vs {b['barriers_per_iter']} barriers per iteration "
            f"({len(a['clusters'])} vs {len(b['clusters'])} clusters): no static dual-wave pairing"
        )
        return
    shift = a["loop_offset"] - b["loop_offset"]
    for k in range(n):
        rep.phases.append(_phase(k, a["clusters"][k], b["clusters"][(k + shift) % n]))


def estimate_cost(sched: Schedule, shape: Optional[dict] = None, target: Optional[TargetDesc] = None) -> CostReport:
    target = target or load_target(sched.target)
    loop = next((s for s in sched.sections if s.loop is not None), sched.sections[0])
    clusters = _clusters(loop.ops, target)
    per_iter = sum(c.mfma_cycles for c in clusters)
    tiles_per_iter = int(sched.meta.get("loop_tiles_per_iter", 1))
    rep = CostReport(sched.name, target.key, clusters, per_iter, tiles_per_iter)
    rep.unknown = sorted(k for k, v in (target.fact("latency_cycles").value or {}).items() if v is None)
    if sched.group_sections is not None:
        _group_loops(sched, target, rep)
        rep.hints = _mfma_rate_hints([op for info in rep.groups.values() for op in info["ops"]], target) + rep.hints
        for info in rep.groups.values():
            del info["ops"]
    else:
        rep.hints = _mfma_rate_hints(loop.ops, target)

    if sched.group_sections is None and sched.groups == 2 and sched.stagger and clusters:
        n = len(clusters)
        for k in range(n):
            rep.phases.append(_phase(k, clusters[k], clusters[(k - 1) % n]))

    workload = sched.meta.get("workload")
    if shape is not None and workload is not None:
        w = workload(**shape)
        rep.shape = shape
        clk = target.require("peak_sclk_mhz") * 1e6
        cus = target.require("num_cus")
        simds = target.require("simds_per_cu")
        waves = int(sched.meta["waves"])
        wg_per_cu = max(1, target.require("lds_bytes_per_cu") // max(1, w.get("lds_bytes", 1)))
        wg_per_cu = min(wg_per_cu, int(sched.meta.get("max_wg_per_cu", wg_per_cu)))
        slots = cus * wg_per_cu
        tiles = sorted(w["tiles_per_wg"], reverse=True)
        rounds = -(-len(tiles) // slots)
        # Each round is as long as its slowest WG; the busiest SIMD carries waves/simds waves.
        cyc_tile_simd = rep.mfma_cycles_per_tile_per_wave * (waves / simds) * wg_per_cu
        t_mfma = sum(tiles[r * slots] for r in range(rounds)) * cyc_tile_simd / clk
        bw = target.require("hbm_bytes_per_s")
        est = {
            "wg_slots": slots,
            "wg_rounds": rounds,
            "t_mfma_us": t_mfma * 1e6,
            "flops": w["flops"],
            "tflops_at_mfma_bound": w["flops"] / t_mfma / 1e12,
            "useful_tflops": w.get("useful_flops", w["flops"]) / t_mfma / 1e12,
            "useful_fraction": w.get("useful_flops", w["flops"]) / w["flops"],
            "hbm_bytes_unique": w["hbm_bytes_unique"],
            "hbm_bytes_noreuse": w["hbm_bytes_noreuse"],
            "t_hbm_unique_us": w["hbm_bytes_unique"] / bw * 1e6,
            "t_hbm_noreuse_us": w["hbm_bytes_noreuse"] / bw * 1e6,
        }
        if est["t_mfma_us"] >= est["t_hbm_noreuse_us"]:
            est["bound"] = "MFMA (even without cache reuse)"
        elif est["t_mfma_us"] >= est["t_hbm_unique_us"]:
            est["bound"] = "MFMA if K/V is reused from L2/MALL across q-blocks, else HBM"
        else:
            est["bound"] = "HBM"
        rep.estimate = est
    return rep
