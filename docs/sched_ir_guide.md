# Schedule IR (`flydsl.sched`)

A hand-tuned kernel is mostly a *schedule*: which buffer holds which tile in
which pipeline stage, which counter each async copy lands on, where every wait
and barrier sits, how MFMAs interleave with VALU/TRANS work, and which wave
runs at which priority. In a plain FlyDSL kernel that plan is spread across
`sched_barrier`, raw `s_waitcnt` bitfields and loop structure, where neither a
reviewer nor an agent can check it.

`flydsl.sched` makes the schedule data. The same op list is

* **verified** before compilation (`verify`) by simulating every wave group,
* **costed** (`estimate_resources`, `estimate_cost`) from documented hardware facts,
* **emitted** inside a `@flyc.kernel` (`lower.emit`), so the kernel *is* the
  verified schedule.

The package is pure Python and imports without MLIR or ROCm.

## Pieces

| Module | Role |
|---|---|
| `ir.py` | `Buffer` (LDS slots), `Queue` (async counter, compiler-tracked or not), `Op`, `Section` (straight-line or loop with carried values), `Schedule`, `Builder` |
| `targets/<arch>.json` | Hardware facts, each with `source` and `confidence`; unknown facts are `null` |
| `verify.py` | Hazard / data-consistency / scope / hardware-contract checks |
| `resource.py` | LDS bytes, live-VGPR lower bound, budget at `waves_per_eu` |
| `cost.py` | MFMA cycles per cluster, dual-wave phase pairing, MFMA-bound and HBM time for a shape, faster-instruction hints |
| `lower.py` | `emit(ops, bindings, env, tile)` — control ops lowered from the target; everything else calls a kernel binding |
| `__main__.py` | `python -m flydsl.sched check|resource|cost|show|target` |

## Writing a schedule

```python
from flydsl.sched.ir import Buffer, Builder, Queue, Schedule, Section, SlotUse, T

b = Builder()
b.op("lds_dma", "load_v", args={"tile": T("j", -2), "buf": 1}, count=NDMA, queue="dma",
     writes=(SlotUse("lV", 1, T("j", -2)),))          # V tile j-2 -> lV[1], on vmcnt
b.op("ds_read", "read_k", outs=("v_k",), args={"buf": 1}, count=2 * KS, queue="ds",
     reads=(SlotUse("lK", 1, T("j", -2)),))           # expects K tile j-2 in lK[1]
b.wait(lgkm=0); b.wait(vm=2 * NDMA); b.barrier()
b.op("mfma", "qk", ins=("v_k", "q"), outs=("v_s1",), count=2 * KS, mfma="mfma_f32_32x32x16_bf16",
     consumes=(("K", T("j", -2)),))
b.interleave(6, "trans", 3, group=1)                  # 6 x {1 MFMA, 3 TRANS} sched_group_barrier
```

`args` are passed to the binding; `TileRef`s are resolved by the kernel
(`tile` callback) at emit time and evaluated numerically by the verifier.

## What the verifier models

* Each wave group runs the same trace; with `stagger`, group 1 executes one
  extra barrier at `stagger open` and group 0 one at `stagger close`.
* Position *a* happens-before *b* for every wave iff some barrier every group
  crosses lies between them (`max_g bidx_g(a) < min_g bidx_g(b)`).
* Async ops enter a FIFO per counter and complete at the `wait` that drains
  them; compiler-tracked queues are also drained before their first consumer.

Rules: `SYNC.RAW_LDS`, `SYNC.WAR_LDS`, `SYNC.REG_NOT_READY`, `SYNC.UNINIT_READ`,
`SYNC.UNDRAINED`, `DATA.STALE_TILE`, `SEM.COVERAGE`, `SCOPE.UNDEFINED`,
`SCOPE.UNCARRIED`, `SCOPE.CARRY_INIT`, `HW.UNSUPPORTED`, `HW.WAITCNT_RANGE`,
`HW.SETPRIO_RANGE`, `SCHED.GROUP_UNSATISFIABLE`, `PERF.COUNTER_SATURATED`.

Not modelled: VGPR-level MFMA/VALU data hazards, register banks, LDS bank
conflicts, scalar-memory ordering, and anything inside an op marked opaque.

## Samples (gfx950)

| Schedule | Kernel | Emitted from the IR |
|---|---|---|
| `kernels/attention/sched/mha_gfx950.py` | `kernels/attention/flash_attn_sched_gfx950.py` (dense subset of `flash_attn_gfx950.py`) | whole kernel body (prologue, 8-cluster dual-wave loop, epilogue) |
| `kernels/attention/sched/mla_gfx950.py` | `kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py` | GEMM1 (with interleaved prefetch) and the GEMM2 core on gfx950; the work-item program is modelled for verification |

```
python -m flydsl.sched check kernels.attention.sched.mha_gfx950:build_mha_schedule --arg head_dim=64
python -m flydsl.sched cost  kernels.attention.sched.mha_gfx950:build_mha_schedule --shape batch=4,heads=64,seq_q=8192
python -m flydsl.sched cost  kernels.attention.sched.mla_gfx950:build_mla_schedule --shape num_seqs=64,kv_len=8192
```

Findings on the existing hand schedules:

* MHA D=64: the last epilogue cluster requests 9 MFMA slots from
  `sched_group_barrier` with 8 MFMAs in the region (`SCHED.GROUP_UNSATISFIABLE`).
* MLA: `s_setprio 15/14` keep only `SIMM16[1:0]` (effective 3/2), and the fp8
  MFMA in use (`16x16x32_fp8_fp8`, 1024 flop/clk/SIMD) runs at half the rate of
  `16x16x128_f8f6f4` on gfx950; the cost model puts the 64x8K decode shape
  MFMA-bound.

Tests: `tests/unit/test_sched_verify.py`, `tests/unit/test_sched_emit.py`
(no GPU), `tests/kernels/test_flash_attn_sched_gfx950.py` (gfx950).

### GPU check (MI355X, one GPU, 2026-10-18)

Environment: `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4`,
the `flydsl==0.4.0` wheel's compiled `_mlir` with this branch's Python sources on
top (the branch changes no C++). Timings are CUDA-event medians of 50 runs,
taken once with no clock control. Read them as a sanity check, not a benchmark.

* `test_flash_attn_sched_gfx950.py`: 16/16 pass. The emitted kernel matches SDPA
  and is bit-identical to `flash_attn_gfx950.py`.
* MLA decode (bs/ctx = 1/63, 1/128, 4/2048, 33/2333, 32/8192): split logits and
  LSE are bit-identical to the pre-change kernel. After `mla_reduce_v1` the
  accuracy against an fp32 reference is unchanged (cosine 0.994 to 0.99995).
* MHA and MLA, hand-written vs emitted: the final ISA is the same. MHA has the same
  2867 instructions in the same order (only two kernarg offsets differ, see "ISA
  conformance"). MLA's ISA is textually identical. Early single-pass timings that
  showed 1-8% gaps were measurement order effects: the first kernel timed at a
  large shape runs slow after a short warm-up.
* Calibration gap: the cost model's MFMA bound for 4x8192x64x128 is about 2.5
  PF of issued MFMA work. The measurement is 1.18 PF, so about 53% of the time
  is not in the model (softmax VALU, LDS, barriers, latencies that are `null`
  in the target).

### Comparison with aiter's assembly kernels (MI355X, 2026-10-19)

The same inputs and the same split metadata go to every implementation; aiter
generates the metadata in its process and FlyDSL reads the same arrays. Timing is
profiler device time per call: 50 warm-up calls, then 5 rounds of 30 calls with the
implementations alternated. Processes ran aiter, FlyDSL, aiter, FlyDSL. Clocks were
not pinned. Values are the per-pass medians (pass 1 / pass 2); aiter at commit
`8ddfc7510`.

| Kernel | Shape | aiter `.co` | aiter `hk_mla_decode_fwd` | FlyDSL hand-written | FlyDSL schedule-emitted |
|---|---|---|---|---|---|
| MHA bf16 D=128 (`fwd_hd128_bf16.co`) | 4x8192x64, non-causal | 7563 / 7457 us | - | 7424 / 7442 | 7431 / 7442 |
| MHA bf16 D=128 (`fwd_hd128_bf16_causal.co`) | 16x8192x64, causal | 15692 / 15594 | - | 15525 / 15505 | 15516 / 15501 |
| MLA decode fp8, 128 heads (`mla_a8w8_qh128_m32x4_n16x2_msk0_ps.co`) | 33 seqs x 2333 | 38.8 / 38.4 | 34.2 / 33.9 | 34.8 / 34.8 | 34.8 / 34.7 |
| same | 32 x 8192 | 81.6 / 81.1 | 74.4 / 73.8 | 73.3 / 75.4 | 73.1 / 73.9 |
| same | 128 x 8192 | 268.5 / 267.6 | 241.9 / 245.8 | 266.3 / 267.5 | 266.5 / 268.8 |

The `.co` MHA tile is 256x64 (`fmha_fwd.csv`), the same as the dual-wave kernel's
BLOCK_M/BLOCK_N. For MLA, the "hand-written" column is the m16x8 kernel before
this branch. Two notes:

* Accuracy: after `mla_reduce_v1`, FlyDSL m16x8 is much less accurate than the
  `.co` and hk kernels at short contexts (1x63: max error 0.89, cosine 0.994, vs
  0.018 / 0.99986). The kernel behaved the same before this branch, and the
  results agree from about 32x8192 up.
* aiter's `get_mla_metadata_v1` does not return the same split on repeated calls,
  and the `.co` kernel faulted when given metadata saved in another process. Build
  the metadata in the process that launches it.

## Target packs and hazards

### Packs

The public descriptions in `python/flydsl/sched/targets/` hold only facts from public
documents. To add facts that cannot be published, write a *pack*: a JSON file with the
same schema, kept outside the repository. Point `FLYDSL_SCHED_TARGET_PATH` at the
directory (or several directories, separated by `os.pathsep`):

```bash
export FLYDSL_SCHED_TARGET_PATH=/path/to/private/targets
python -c "from flydsl.sched import load_target; t = load_target('gfx950'); print(t.layers)"
```

`load_target("<arch>[@<sku>]")` reads the following files in order:

1. The public `<arch>.json`.
2. `<arch>.json` from each pack directory.
3. `<arch>@<sku>.json` from each pack directory. When no sku is requested, the sku the
   arch-level layers declare is used.

Facts merge per name, and dict-valued facts merge per key. The later file wins. Arch
matching stays exact, and an arch that exists only in a pack is loadable once its
directory is on the path. The cache key includes the search path.

Every fact records where it came from:

* `t.provenance(name[, key])` returns a `Fact` with `value`, `source`, `confidence`,
  `note`, and `origin`. The origin is the file path, or `builtin:<file>` for the public
  description.
* `t.facts_with_confidence("measured", ...)` lists facts and dict keys by grade.

The schema, including the field names the analyzers read, is in
`python/flydsl/sched/targets/SCHEMA.md`.

### Hazard checks (`hazards.py`)

`verify()` runs these checks after its scope checks. Each one runs only when the target
describes the facts it needs. A fact that is described but `null` gives a
`TARGET.UNKNOWN` info finding, and the check is skipped rather than guessed.

| Rule | Severity | Facts | What it checks |
|------|----------|-------|----------------|
| `HAZARD.MATRIX_DEP` | error | `matrix_hazards` (`interlock: false`) | Inside a section, and around a loop body into its next iteration, the distance from a matrix op to a dependent op is below the required count. The dependent op either reads the result (RAW, as A/B or C, from VALU, or from memory), overwrites the result (WAW), or overwrites a source (WAR). The distance is the sum of `op.count`, with `s_nop n` = n+1 and scheduler pseudo-ops = 0. It also covers a VALU write followed by a matrix read. |
| `PERF.MATRIX_DEP_STALL` | info | `matrix_hazards` (`interlock: true`) | The same distance check, but the hardware stalls instead of computing wrong data. |
| `HW.COUNTER_UNKNOWN` | error | `counters` | A queue or wait names a counter the target does not have, such as `vm` on a split-counter arch. The trace simulation is then skipped. |
| `HW.COUNTER_MISMATCH` | error | `counters[c].kinds` | A queue carries an op kind that its counter does not count. |
| `HW.WAIT_PARTIAL_OOO` | error / warn | `counters[c].in_order`, `ooo_kinds` | A wait value other than 0 on an out-of-order counter (error). Or a wait value other than 0 while the schedule also puts out-of-order kinds on that counter (warn). |
| `PERF.LDS_BANK_CONFLICT` | info | `lds_banks`, `lds_bank_bytes`, `lds_lanes_per_cycle` | Conflict degree for `ds_*` ops that declare `args["lds_stride_bytes"]` and `args["lds_access_bytes"]`. |

Limitations:

* An op with `count=0` does not say how many instructions it issues; the compiler
  places them. The matrix check therefore never uses such an op as a producer or a
  consumer. When one sits in between, the distance is only a lower bound, and an
  error becomes a warn.
* LLVM pads intrinsic-emitted code with `s_nop` for these hazards. A
  `HAZARD.MATRIX_DEP` finding therefore means wait states that are either missing
  (inline asm) or added by the compiler; it does not always mean wrong results.
* The trace simulator still drains only `vm` / `lgkm` waits. Split counters (gfx12) are
  validated by the table checks, but their waits are not simulated.

## Cross-block scheduling

LLVM schedules one region at a time. It does not move a DMA from the head of loop
iteration `j` into iteration `j-1`, across an `s_barrier` into another phase, or out
of the peeled epilogue, and it does not re-derive hand-written `s_waitcnt` values
afterwards. The Schedule IR sees the whole program, so these moves are list edits.
The verifier accepts a move only if it still verifies on every tile count, for every
wave group.

| API (`flydsl.sched.transform`) | Effect |
|---|---|
| `move_op(s, src, dst)` | Moves an op across barriers/clusters, or between straight-line sections. Moving an op into or out of a loop body is refused because it would change the op's trip count. |
| `rotate_up(s, loop, i)` / `rotate_down` | Moves a body op across the back-edge. The first (or last) instance goes to the prologue (or epilogue). The matching op peeled at the head of the epilogue (or tail of the prologue) is absorbed, so no runtime guard is needed for zero-trip loops. Tile references are rebased (`j-2` becomes `j` or `first+1`). Values the op defines become loop-carried. |
| `synthesize_waits(s)` | Loosens every `vmcnt`/`lgkmcnt` to the largest value the verifier accepts. If the schedule is illegal, it rebuilds them from 0. With `keep_roles`, no async op may start completing inside a compute cluster. |
| `hoist_async(s)` | Greedy pass: each async memory op walks upward, crossing barriers and the back-edge. It only lands in memory clusters (dual-wave roles). Crossing a wait on its own counter is a compound move that raises the wait by `op.count`. It lands only where its latency cover strictly grows. Waits are re-synthesized at the end. Every refusal is logged with the verifier rule that blocked it. |
| `async_cover`, `cover_summary`, `cluster_roles` | The objective: MFMA cycles and work ops between an async op and the wait that drains it, plus the cluster role where it completes. |
| `flydsl.sched.program.emit_program` | Emits any (transformed) schedule. Loop-carried state comes from `Section.carried` (nested values are flattened; optional codecs). The loop is built with the AST rewriter's own `scf_range`/`scf_yield_`. `python_loop` runs the same emission eagerly on CPU. |

The MHA kernel builds its whole body with `emit_program`, so
`build_flash_attn_sched_gfx950_module(..., schedule=hoist_async(s)[0])` runs a
transformed schedule without kernel edits. `schedule_fingerprint` keeps JIT cache
entries apart.

Results on the dual-wave MHA:

* `hoist_async(hand)` leaves the loop unchanged. Every loop DMA stops on
  `SYNC.WAR_LDS` against the other group's read of the same slot. With two LDS
  slots per tensor, the hand prefetch distance is the maximum; going further needs
  a third slot. (In the prologue it issues two DMAs ahead of `scale_q`, which is legal
  but gives no measurable gain.)
* Perturbed schedule (K[j+1] rotated into the next iteration, K[j] moved into C4,
  waits re-synthesized): `hoist_async` rotates K back across the back-edge,
  returns K[j] to C2, and restores `vmcnt(4)`. The loop and epilogue come back
  identical to the hand schedule.
* GPU (MI355X GPU 1, bf16, CUDA-event median of 50, two repetitions each, unpinned
  clocks). All variants are bit-identical to the hand
  kernel. B=16 S=8192 H=64 D=128 causal: hand 15405/15441 us, perturbed 15960/16006
  us (+3.6%), recovered 15421/15713 us. B=4 non-causal: 7413/7407 vs 7438/7437 vs
  7419/7409 us. Run-to-run noise on the causal shape is about 2%.
* Calibration lesson: the first objective (MFMA cover only) produced a recovered
  schedule that issued DMAs inside compute clusters and loosened `lgkmcnt(0)` so
  completions landed in the MFMA stream. It measured 6-8% *slower* than hand. The
  role constraint (`keep_roles`) comes from that measurement.

Limits: tile-count-dependent programs (`Schedule.program`, e.g. the MLA work item)
are verified but not transformed or emitted by `emit_program`. Moves are greedy and
do not consider adding LDS slots. The cover objective does not model issue-port or
LDS-bandwidth contention.

## ISA conformance

`emit(..., tag=<name>)` makes every MLIR op built for op `i` of section `s` carry
the location `sched:<name>/<s>:<i+1>`. It uses a `FileLineColLoc`, entered the same
way `dsl_loc_tracing` enters a captured location, so DSL primitives and
`@flyc.jit` helpers keep it. With `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` the location
reaches the DWARF line table. In `*_final_isa.s` (`FLYDSL_DUMP_IR=1`) it appears as
`.file N "sched:<name>" "<s>"` and `.loc N <i+1> 0`. `emit_program` passes
`section=` itself, and `flash_attn_sched_gfx950.py` tags with the schedule name.
Without debug info the tags are stripped. Debug info does not change the
generated code (checked below).

```
python -m flydsl.sched conform emitted_final_isa.s \
    --schedule kernels.attention.sched.mha_gfx950:build_mha_schedule --arg causal=False \
    [--against hand_final_isa.s] [--ops] [--verbose] [--json]
```

`conform.py` assigns each instruction to the closest preceding `.loc`, which gives
the instructions of every schedule op. It reports `verify.Finding`s:

| Rule | Meaning |
|---|---|
| `CONFORM.ORDER` | Within one basic block, code of an op issues after code of a later fence region. `sched_barrier(0)`, fenced barriers and `stagger` are the fences. |
| `CONFORM.WAIT_MERGED` / `WAIT_DROPPED` | A schedule wait has no `s_waitcnt` of its own, and a neighbouring wait in its region covers it (merged) or nothing does (dropped). |
| `CONFORM.WAIT_TIGHTENED` / `WAIT_LOOSENED` | An `s_waitcnt` is stricter or looser than the waits it stands for. |
| `CONFORM.WAIT_INSERTED` | An `s_waitcnt` the schedule does not have, together with the op it sits at. |
| `CONFORM.BARRIER_COUNT` | The number of `s_barrier`s on barrier ops does not match the schedule. |
| `CONFORM.INTERLEAVE` | The `sched_group` slots requested in a region, compared with the MFMA/VALU/TRANS sequence issued in the basic block they act on. Slots that have no instruction left are listed rather than counted as a divergence. |
| `CONFORM.SETPRIO_*` | An `s_setprio` is missing, has a different value, or has compute work from the other side of it moved across. |
| `CONFORM.COUNT` / `OP_MISSING` | The instruction count of an op differs from `Op.count`, or an op has no code at all. |
| `CONFORM.UNATTRIBUTED` | Instructions with no schedule op (setup, address math, merged line-0 code), summarized by source. |

`diff_isa` aligns two kernels by `s_barrier` regions (difflib over the per-region
count of MFMA / DMA / LDS / TRANS) and lists, per region, the class-count, wait
and order differences, plus the `.amdhsa_*` resource differences.

Limits: attribution follows the line table. Instructions LLVM creates without a
location take the location of the previous instruction, so an inserted `s_waitcnt`
is reported at the op it precedes. Merged instructions get line 0 (unattributed).
Instruction classes are coarse (TRANS = `v_exp/log/rcp/rsq/sqrt/sin/cos`).
Inline-asm instructions are excluded from interleave matching because
`sched_group_barrier` does not see them. LLVM's own mask semantics, for example
whether VALU covers TRANS, are not modelled.

### Real run (MI355X GPU 2, 2026-10-19)

Environment as in *GPU check*, plus this branch. Shape: B=4 S=8192 H=64 D=128,
bf16, non-causal; `flash_attn_gfx950.py` (hand) vs `flash_attn_sched_gfx950.py`
(emitted).

* Debug info and the `_flydsl_loc` change do not change the code. The instructions
  of both kernels are identical with and without `FLYDSL_DEBUG_ENABLE_DEBUG_INFO`,
  and identical before and after the change. Outputs are bit-identical between
  hand and emitted, and between debug and non-debug builds.
* Attribution: 2597 of the 2867 instructions map to a schedule op. The other 270
  are kernel setup and address math, `scf.for` bookkeeping, and line-0 merged code.
* Conformance: no errors. Every barrier is placed. All 10 schedule `vmcnt` waits
  merge into the preceding `lgkmcnt(0)` wait as `s_waitcnt vmcnt(N) lgkmcnt(0)`.
  Inside the scheduled code the compiler adds one `s_waitcnt`: an `lgkmcnt(0)` for
  a buffer-descriptor scalar load sunk into the epilogue
  (`WAIT_INSERTED`, attributed to user code).
  `ORDER`: 119 `scale_q` VALU sink below the prologue QK^T MFMAs.
  `INTERLEAVE`: in C1/C5 (and E1/E5/E9), 18 TRANS and 50 VALU slots are requested,
  but the cluster issues 16 TRANS and 29 VALU. The issued pattern is
  `M1 T3 x5, M1 T1, M1 V8 x3, ...`.
  In C3/C7 the `lazy_rescale_o` branch splits the cluster into three blocks. The
  first `{1 MFMA, 6 VALU}` groups act on a block that holds only the 4 MFMAs of
  PV step 0. The rescale body (84 VALU) is out of line, and the remaining groups
  act on the block after the branch.
* Hand vs emitted: the same 2867 instructions in the same order. 25 of 26 barrier
  regions are identical. The only differences are two kernel-argument load offsets
  (`0x240`/`0x250` vs `0xc0`/`0xd0`) and `kernarg_size` 608 vs 212 bytes. The hand
  kernel takes the paged/bias/ALiBi/sink arguments too. VGPR/SGPR/AGPR, LDS and
  scratch are equal.
* Timing (six alternating rounds of 50 launches, CUDA-event medians, unpinned
  clocks): hand 7475-7499 us, emitted 7451-7468 us, about 0.4% apart. The ISA
  contains no instruction-level cause for this gap. The only differences are the
  kernel-argument block size and layout. Whether that layout causes the gap was
  not tested.

## Per-group programs

By default every wave group runs the same program, and the dual-wave phase offset
is the run-time `stagger` barrier. A schedule can instead give each group its own
program, which shares the buffers and queues:

```python
Schedule(..., groups=2, sections=prog0, group_sections=[prog0, prog1])
```

`sections` is then group 0's program, so single-program tools see group 0's code.
`sections_of(g)` returns group `g`'s program, and `all_sections()` / `all_ops()`
cover every group, so the kernel's binding check sees everything.

**Verifier.** Each group's trace is simulated with its own barrier indices. The traces
are merged by barrier phase, so ops of different groups between the same two barriers
count as concurrent, and only a barrier that both groups cross orders them.

* Wait counters, registers, scope and tile coverage are tracked per group.
* LDS slots are shared, and a slot holds a tile only once every group that writes it
  has written its part. Cross-group RAW/WAR use each group's own barrier index.
* Findings carry a `g<k>:` location prefix.
* `SYNC.BARRIER_MISMATCH` (error): the groups execute different numbers of
  `s_barrier` for some tile count, so one group waits forever. This check also covers
  single programs, for example an unpaired `stagger open`.

**Emission.** `emit_program(..., group_id=v)` emits each group's program in its own
branch of an `scf.if` chain on the wave-uniform group id `v`, so the code is
duplicated like the two loop copies of the aiter `.co`. Each branch starts from a
copy of `env`. `flash_attn_sched_gfx950.py` passes `ctx.stagger_i32`
(`readfirstlane(wave) // 4`).

**MHA variant** (`kernels/attention/sched/mha_gfx950_2prog.py`): two programs built
from the single-program schedule. The stagger becomes code: group 1 has a plain
barrier in its prologue and group 0 one at the end. Barrier `k` therefore pairs group
0's cluster `c` with group 1's cluster `c-1`, exactly as with the stagger. The
`prio` argument chooses how `s_setprio` is used:

* `toggle`: the single schedule's per-cluster `s_setprio 1`/`0`.
* `static`: one priority per group for the whole program, as the `.co` does. Group 0
  and group 1 get `static_prio`, default `(0, 1)`.

GPU check (MI355X GPU 2, 2026-10-19). Container `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4`,
the `flydsl==0.4.0` wheel's `_mlir` with this branch's Python, clocks not pinned.

* Correctness: 16 configurations, the same set as `test_flash_attn_sched_gfx950.py`.
  Every variant matches SDPA (atol/rtol 2e-2) and is bit-identical to the hand kernel.
  The variants are `toggle`, `static (0,1)` and `static (1,0)`.
* Timing: bf16, 7 interleaved rounds (the order rotates every round), 50 launches per
  round, CUDA-event median per round; the table gives the median over rounds, with the
  per-round range in brackets:

  | shape (B,S,H,D) | hand | single program | 2prog toggle | 2prog static (0,1) | 2prog static (1,0) |
  |---|---|---|---|---|---|
  | 4,8192,64,128 | 7465 [7432-7487] | 7423 [7410-7463] | 7423 [7394-7460] | 7353 [7310-7382] | 7402 [7363-7432] |
  | 16,8192,64,128 causal | 15299 [15276-15425] | 15250 [15238-15412] | 15346 [15331-15449] | 15175 [15167-15296] | 15287 [15260-15406] |
  | 1,8192,32,64 | 687 [685-693] | 642 [640-643] | 642 [638-643] | 643 [641-643] | 649 [647-650] |

  `toggle` matches the single program, as expected: it is the same code with the
  runtime-predicated stagger barriers made static. `static (0,1)` is about 0.9% faster
  at D=128 non-causal, where its round range does not overlap the single program's,
  and about 0.5% faster causal, where the ranges overlap at the top. At D=64 it makes
  no difference. One session, no pinned clocks: treat these as indications.

## Register pins

`flydsl.sched.pins` turns "place this value in `v[a:b]`" into schedule data. A pin
is a **register-allocation hint at the value's definition**. The mechanism is
ROCm/FlyDSL#1307 (`rocdl.sched.{vgpr,agpr,sgpr}`). It needs two required LLVM patches on
the pinned LLVM `1df311a6` (`thirdparty/llvm-build-info.json`):
`thirdparty/llvm-extensions/schedule-bank-and-gpr.patch` (the PR's) and
`thirdparty/llvm-extensions/schedule-gpr-hint-coalescing.patch` (our fix on top of it). It replaced an
earlier version built on `llvm.read_register` / `llvm.write_register` and a named-register
LLVM branch (user decision, 2026-10-20).

```python
loop = Section("k", body.ops, loop=("k", rng), carried=("acc",))
loop.pins = {"acc": "v[128:131]"}                 # strict hint
b.op("vmem_load", "load_x", outs=("x",), out_dwords={"x": 4}, pins={"x": "v[64:67] soft"})
```

* Spec: `"v42"`, `"v[64:79]"`, `"a[0:3]"`, `"s[4:5]"`. Strict by default; a trailing
  `" soft"` makes it advisory.
* `Op.pins`: the op's output is hinted where the op defines it.
* `Section.pins` (loop-carried values) are hinted at four places: the initial value, the
  iteration argument at the top of the body (the loop-header phi), the new value at the end
  of the body, and the loop result. `emit_program` applies these through its codecs.
* `kernels/attention/sched/mha_gfx950_pinned.py` builds pinned variants of the MHA
  schedule. Presets: `o` (`v_o` in `v[192:255]`), `ml` (`m_row`/`l_row` in `v190`/`v191`),
  and `*_soft`.

**How the hint works.** `rocdl.sched.vgpr %v, regnum [soft]` is an identity op. It lowers
to `llvm.amdgcn.sched.vgpr` and a pseudo. The pre-RA `AMDGPUSchedBank` pass replaces the
pseudo with a COPY and puts the hint on both sides of it. The hint is propagated through
COPY / REG_SEQUENCE / INSERT_SUBREG / SUBREG_TO_REG. When the coalescer merges a hinted
register into another, `schedule-gpr-hint-coalescing.patch` (`SIRegisterInfo::updateRegAllocHint`)
moves the hint onto the register it keeps. Without that, a hint on a loop-carried value is
lost when its phi is coalesced.
* A strict hint is a hard hint: `getRegAllocationHints` returns only that register. A soft
  hint never replaces a strict one.
* A soft hint moves the register to the front of the allocation order.
* Widths are 32/64/128/256/512 bits. Wider values are split into leaves on consecutive
  sub-ranges. The backend silently drops a hint of any other width, so the verifier
  rejects those widths.
* Nothing is reserved: other values may use the range while the pinned value is dead.

**Verifier** (`pins.check`, run by `verify`):
* `PIN.SYNTAX`, `PIN.UNKNOWN_VALUE`.
* `PIN.WIDTH`: the range width must equal `out_dwords`. `PIN.WIDTH_UNKNOWN`.
* `PIN.SPLIT`: no single hint width fits, so the value is split into leaves.
* `PIN.OUT_OF_FILE`: checked against `arch_vgpr_max_per_wave`, `acc_vgpr_max_per_wave`
  and `sgpr_addressable_per_wave`.
* `PIN.NO_AGPR`.
* `PIN.ALIGN`: a tuple that does not start on an even register has no candidate. Error
  for strict, warning for soft.
* `PIN.ABI`: the range includes `v0`.
* `PIN.CARRIED` (info), `PIN.MULTI`.
* `PIN.CONFLICT`: an op pin disagrees with its loop's pin.
* `PIN.OVERLAP`: two live pins share registers. Error if both are strict.
* `PIN.BUDGET`, `PIN.ASYNC_SPILL` (strict pins only), `PIN.SCHED`.

**Checking placement.** `pins.isa_pin_report(asm, ranges)` reports, per requested range:
writers, readers, MFMAs accumulating in place, and register copies into or out of it
(`PIN.INDIRECT`). `pins.mfma_accumulators(asm)` shows where accumulators actually live.
`pins.pending_lds_uses(asm)` flags copies or spills that touch an inline-asm LDS read before
its wait. `tools/named_reg/isa_report.py <final_isa.s> <ranges>` prints all three for a
`FLYDSL_DUMP_IR=1` dump.

### Results (MI355X GPU 3, 2026-10-20)

Setup:
* Clean build from scratch with `scripts/build_llvm.sh` (`amd-minimal`, fresh clone of
  `1df311a6` plus the four required patches) and `scripts/build.sh`. ROCm 7.2.4 container.
* lit:
  * `sched-gpr-hint-coalescing.ll` passes, and fails with the PR patch alone.
  * `sched-gpr-hints.mir`, `sched-bank.ll`, `sched-bank-hints.mir` pass.
  * `sched-gpr.ll` fails at its gfx1250 `v[512:527]` case. That failure is in the PR patch
    alone too, so it was there before our patch.
* pytest: `test_schedule_gpr/bank`, `test_sched_*` and `test_llvm_extension_*` pass (209);
  `test_flash_attn_sched_gfx950.py` passes (46).
* Every variant is correct (fp32 reference / SDPA).
* With no pins, the patches change nothing: the unpinned MHA ISA (D=128 and D=64) is
  identical to that of the same LLVM hash built without either patch.
* Timing: bf16; median of 30 launches per process; 5 interleaved rounds. The table gives
  the median of the round medians and the round range.

| kernel | pin | where the value lives (ISA) | us |
|---|---|---|---|
| demo 16x16x32, 4096 blocks x 512 steps | none | acc in `v[0:3]` | 754.2 [750.9-762.4] |
| | `v[128:131]` | acc in `v[128:131]`, in place, 0 copies; 132 VGPRs | 829.9 [826.9-856.2] |
| | `v[8:11]` | acc in `v[8:11]`, in place, 0 copies | 754.3 |
| | `a[0:3]` strict | 3 of 5 MFMAs in `a[0:3]`, 4+4 accvgpr copies | 754.4 |
| | `a[0:3]` soft | all 5 MFMAs in `a[0:3]`, in place, 0 copies | 753.4 |
| demo 32x32x16, same | none | acc in `v[2:17]` | 774.1 [766.1-777.0] |
| | `v[128:143]` | acc in `v[128:143]`, in place, 0 copies; 144 VGPRs | 877.3 [851.5-880.0] |
| | `a[0:15]` | acc in `a[0:15]` (AGPR-form MFMA), in place | 774.2 |
| MHA D=128 B4 S8192 H64 | none | | 7435.8 [7432.1-7440.2] |
| | `ml` (`v190`/`v191`) | row max / row sum in `v190`/`v191`, 2 copies | 7435.0 [7427.8-7437.4] |
| | `o` (`v[192:255]`) strict / soft | the 48 PV MFMAs accumulate in place in `v[192:255]`; no copies in the loop: the 56 copies in (48 out) are the shared zero initializer broadcast from `v[240:255]` to the other leaves before the loop and on the zero-trip path; 256 VGPRs, 6 spilled (28 B scratch, no LDS-read hazards) | 7446.4 [7441.2-7451.3] / 7447.1 (+0.14%) |
| MHA D=64 B4 S8192 H32 | none | | 2372.5 [2371.8-2378.2] |
| | `v_o` → `v[224:255]` strict / soft | the 24 PV MFMAs accumulate in place in `v[224:255]`; copies (24 in, 16 out) only outside the loop; 256 VGPRs | 2374.8 [2368.7-2375.8] (+0.1%) |
| | `v_o` → `v[32:63]` | PV accumulators in `v[32:63]`; 164 VGPRs | 2372.1 |

* **Hints now hold on loop-carried values**, strict or soft. On the MHA O accumulator the
  cost is at most +0.14%, even though the kernel's VGPR count rises to 256. The register
  count does not limit this kernel. The copies are outside the loop. The remaining
  cost is at D=128, from the 6 spilled VGPRs and the copies.
* **Demo cost.** Low registers and AGPRs cost nothing. Pinning to `v128` costs 9-11% at 4096
  blocks. The section below explains why.

**Why high pins cost on this demo** (MI355X GPU 3, 2026-10-20, clocks not pinned):
* **The code is the same.** The hot loop is identical to the unpinned one apart from register
  numbers (29 instructions). What changes is the VGPR count: 132 (16x16) / 144 (32x32)
  instead of 32 / 40. With a 512-entry unified file in granules of 8, that allows 3 waves per
  SIMD instead of 8.
* **Occupancy-only control.** An unpinned kernel with an empty inline asm that clobbers
  `v131` / `v143` has the same loop and the same VGPR count. It reproduces the pinned time at
  every grid size (within 0.6%). So the cost comes from occupancy alone.
* **Grid sweep.** One wave per block, 1024 SIMDs; 3 rounds, 20 launches each. Time ratio of
  the high pin to the unpinned kernel:

  | waves per SIMD (blocks / 1024) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 16x16, 512 steps | 1.00 | 1.00 | 1.00-1.01 | 1.09-1.10 | | 1.00 | | | | |
  | 16x16, 128 steps | | | | 1.06 | 1.02 | 0.98 | 1.04 | 1.01 | 0.99 | 1.02 |
  | 32x32, 512 steps | 1.02 | 1.00 | 1.00 | 1.10-1.11 | | 1.00 | | | | |
  | 32x32, 128 steps | | | | 1.06 | 1.02 | 0.99 | 1.05 | 1.02 | 0.99 | 1.01 |

* **Interpretation.** The sawtooth is residency quantization: whenever the waves per SIMD
  divide by 3, the gap is zero. In between, the kernel runs one more round with too few
  waves to saturate memory. At 4 waves per SIMD the last round has a single wave per SIMD,
  and one wave per SIMD reaches only 4.8 TB/s against 5.6-5.8 TB/s. Three resident waves per
  SIMD sustain the same bandwidth as eight: at 6 and 12 waves per SIMD the pinned kernel is
  equal or 1-2% faster. So "fewer loads in flight per SIMD" is not the cause here.
* **The default grid.** The demo's default of 4096 blocks happens to hit the worst case.
* **Consequence.** A pin above the occupancy budget is free only when the grid fills whole
  rounds at the reduced occupancy. The `PIN.BUDGET` warning flags pins above the budget.
* **AGPR pins.** A strict `a[0:3]` holds only partly (copies), while soft holds fully; this
  is the allocator's choice between the AGPR and VGPR MFMA forms.
* **Shared initial values.** Two leaves initialized by the same constant are CSE'd into one
  value before the loop. Their two hints then land on one virtual register. With the
  coalescing fix every leaf still gets its range, through copies before the loop.
* **With the PR patch alone** (measured the same day, before the fix):
  * In the one-block demo the hints held.
  * In the MHA no loop-carried pin got its register (`o`, `ml`, D=64 `v_o`). The
    coalescer merged the hinted registers into the phi-elimination registers and dropped
    the target hint. `llc -debug-only=amdgpu-sched-bank,regalloc` shows `reg 240 strict`
    set on `%2733`, and greedy then allocating the merged `%5308` with only a copy hint.
  * `schedule-gpr-hint-coalescing.patch` fixes this, and its lit test reproduces the bug.
* **Comparison with the named-register path (2026-10-19).**
  * What it did: it forced placement through reserved registers. 4-dword accumulators
    accumulated in place. 16-dword accumulators needed copies around every MFMA (+20% in
    the demo; hints: 0 copies). MHA `ml` cost +0.25% (hints: no measurable cost). D=64
    `v_o` cost +6.4% (hints: +0.1%).
  * That +6.4% was put down to occupancy and copies. The 256-VGPR result above leaves the
    copies around the loop as the likely cause.

## Issue programs

A `sched_group` interleave is a hint. LLVM fills what it can and schedules the rest
itself. An issue program (`flydsl.sched.issue`) states the exact class order of one
fence region, and its slot counts are checked against the instructions the region
really produces:

```python
b.issue(group=1, gap=(3, 6))                      # template: MFMA spine, TRANS+VALU fillers
b.issue("explicit", slots=[("mfma", 1), ("trans", 3), ...])
sched, report = resolve(sched, calibration)       # -> exact sched_group_barrier chains
```

**Calibration.** `python -m flydsl.sched calibrate <final_isa.s> --schedule m:b --out c.json`
counts, per op and per binding, the instructions each op produced. It reads them through
the `emit(tag=...)` locations. Counts are *scheduler units*. After scheduling, LLVM unpacks
`v_pk_{add,mul,fma}_f32` that sit in an MFMA's shadow into two VOP3 instructions. The
group sizes count the packed op once, so the checker counts such a pair as one unit
(`conform.mark_unpacked`). The raw count is kept as `raw:<class>` and used to balance gaps
in issued instructions. Exact per-op counts (`by_op`) apply only to sections whose binding
sequence matches the calibrated one; otherwise per-binding counts apply.

**Template lowering.** The fillers are the region's own instructions, taken in op order.
They are split into gaps of equal raw size after each spine instruction. When that would
exceed `gap[1]`, the tail after the last spine instruction takes the rest. The result
becomes a chain of `sched_group_barrier(mask, n, group)` calls, one per run of one class.
Every groupable instruction then has exactly one slot, so LLVM's group solver has no
slack.

**Checks** (`ISSUE.*`, reported as `verify.Finding`s):
- `UNCALIBRATED`: an op has no counts.
- `CONTROL_FLOW`: an op branches. A chain acts inside one basic block.
- `GAP`: the gap bounds are infeasible.
- `DEPENDS`: a filler reads a spine op that issues after it, or a spine op reads a filler
  that issues after it (transitive, through `ins`/`outs`).
- `COUNT`: explicit slots do not match the counts.
- `ASM`: inline-asm instructions in the region. No mask selects them, so they float.

`python -m flydsl.sched issue m:b --arg issue=@c.json` prints the plans.

`conform` reports `CONFORM.ISSUE` per program:
- the requested and issued sequences, in units, with the first divergence;
- the raw gaps between spine instructions, the back-to-back MFMAs, and the gaps outside
  the bounds.

**Why chains, not one instruction per fence.** The other option is emitting each slot
between `sched_barrier(0)` fences from bindings that are streamed one instruction at a
time. That needs every helper rewritten as a generator, and it removes LLVM's freedom
inside a slot. The chains needed no helper changes. On the real ISA they reach the
requested order exactly or to within one unit. Two other things did not work:
- LLVM's exact solver (`amdgpu-igrouplp-exact-solver`) did not finish compiling the
  kernel in 20 minutes.
- Counting raw instructions instead of units made the groups run dry before the last
  MFMAs, which left MFMAs back-to-back.

**What the MHA needed** (`build_mha_schedule(..., p_cast="cvt", issue="default")`, D=128):
- `p_cast="cvt"`: the inline-asm `v_cvt_pk_bf16_f32` of `cast_p` cannot be grouped.
  `fptrunc` makes LLVM select the same instruction (bit-identical) inside the chain.
- The PV clusters (C3/C7, E3/E7) are two issue regions each:
  - PV step 0 with `reduce_max`;
  - then the rescale, fenced off on its own. In the lazy variant it is a wave-uniform
    branch, and an issue region must be one basic block;
  - then steps 1-3 with `sub_m`/`exp2`.
- `rescale="deferred"` is a branchless alternative. The lazy decision is a select, and
  `O *= corr` moves to the next memory cluster under a uniform branch. Because the old
  O is scaled after the PV that reads it, P needs no rescale. The output is
  bit-identical whenever no lane rescales.
- `CALIBRATION_D128` in `kernels/attention/sched/mha_gfx950.py` is the measured table.

### Results (MI355X GPU 1, 2026-10-19)

Environment as in *GPU check*. Shape: B=4 S=8192 H=64 D=128, bf16, non-causal.

- **Conformance.** On the final ISA, every issue region has no back-to-back MFMA, with
  3-6 instructions between consecutive MFMAs. In the loop the only larger gaps are
  across the lazy branch (10, twice) and across the memory clusters. Before:
  9 back-to-back pairs and gaps up to 44 inside compute clusters.
- **Correctness.** The outputs are bit-identical to `flash_attn_gfx950.py`.
  `tests/kernels/test_flash_attn_sched_gfx950.py` covers 3 shapes x causal/non-causal
  x 3 variants; the full file passes (34 tests).
- **Timing.** Six variants, 10 alternating rounds of 50 launches each, CUDA-event
  median per round, unpinned clocks. A second build of the current kernel measured
  +0.08%. Spread across rounds was 1.0-1.6%.

| variant | median us | vs current emitted |
|---|---|---|
| hand `flash_attn_gfx950.py` | 7474.8 | +0.69% |
| current emitted | 7423.8 | 0 |
| issue programs in QK clusters only (`issue_split=False`) | 7395.2 | -0.39% |
| issue programs in all compute clusters (lazy, split PV) | 7589.8 | +2.24% |
| all clusters, deferred rescale | 7646.0 | +2.99% |

The QK-cluster programs were -0.39 to -0.67% in four separate runs. Exact even spacing in
the PV clusters is slower than LLVM's hint-driven order there, in both rescale variants.
Without issue programs, the deferred rescale alone was +6.6 to +7.3%. Its hints were
written for the lazy cluster shape. The ISA does not show which part of the PV order
matters. Possible factors are the two extra fences around the rescale and the even
placement of the `reduce_max` chain. Neither was measured separately.
