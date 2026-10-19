# Target description schema (`flydsl.sched.target/v1`)

A target description is a JSON file `<arch>.json`. The public descriptions in this
directory hold facts from public documents only. Private *packs* with the same schema
can extend or override them. Packs are found through `FLYDSL_SCHED_TARGET_PATH`, a list
of directories separated by `os.pathsep`, and are never shipped with FlyDSL.

## File

| Field      | Meaning |
|------------|---------|
| `schema`   | `"flydsl.sched.target/v1"` |
| `arch`     | Exact gfx name (`gfx950`, `gfx1250`). Packs apply only to the same arch. |
| `sku`      | Optional product key (e.g. `mi355x`). A pack named `<arch>@<sku>.json` applies only when the requested sku matches. |
| `stepping` | Optional silicon stepping the facts were described for (`A0`, `B0`). |
| `doc`      | Free text. |
| `sources`  | Optional map from source id to a description. |
| `facts`    | Map from fact name to fact entry (below). |

## Fact entry

```json
{"value": <any or null>, "source": "<source id + section/page>", "confidence": "<grade>", "note": "<text>"}
```

* `value: null` means *unknown*. Analyzers report a finding (or skip a check) instead of
  guessing.
* `confidence` is the source's own grade, for example `documented`, `image-verified`,
  `source-reported`, `measured`, `inferred`, or `unknown`.
* `note` records qualifiers and any contradictions that are still unresolved.

### Dict-valued facts

When `value` is an object, a pack merges into it **per key**: keys the pack does not
mention keep their earlier value and provenance. A per-key entry can carry its own
`source` / `confidence` / `note`. Otherwise it inherits the ones of the enclosing fact.
`TargetDesc.provenance(name, key)` reports which file each part came from.

### Load order

1. `targets/<arch>.json`: the public description. It may be absent if a pack exists.
2. For each directory in `FLYDSL_SCHED_TARGET_PATH`, in order: `<arch>.json`, then
   `<arch>@<sku>.json`. When the same fact or key appears more than once, the later
   file wins.

## Fact names used by the analyzers

| Fact | Value | Used by |
|------|-------|---------|
| `wave_size`, `simds_per_cu`, `max_waves_per_simd`, `num_cus` | int | resource / cost |
| `vgpr_file_per_lane`, `arch_vgpr_max_per_wave`, `acc_vgpr_max_per_wave`, `vgpr_alloc_granule`, `sgpr_max_per_wave` | int | resource |
| `lds_bytes_per_cu`, `lds_banks`, `lds_bank_bytes` | int | resource, hazards (bank conflicts) |
| `lds_lanes_per_cycle` | `{instruction or "default": lanes serviced per LDS cycle}` | hazards (bank conflicts) |
| `counters` | `{counter: counter entry}` (below) | verify, hazards |
| `mfma` | `{instruction: {passes, flops, ab, ...}}` | resource / cost / hazards |
| `mfma_cycles_per_pass` | int | cost |
| `matrix_hazards` | `{rule: hazard entry}` (below) | hazards |
| `latency_cycles` | `{instruction: cycles or null}` | cost |
| `sched_group_mask` | `{class: mask}` | lower |
| `supported_ops` | list of op kinds | verify |

Packs may carry more facts than this table lists, such as issue rules, register banking or
VALU wait states. Facts that no analyzer reads are still loaded and can be inspected with
`TargetDesc.facts_with_confidence()`.

### Counter entry

| Key | Meaning |
|-----|---------|
| `max` | Largest encodable wait value. |
| `in_order` | `true` when every operation on the counter returns in issue order. A wait value other than 0 is valid only on an in-order counter. |
| `ooo_kinds` | Op kinds that return out of order even on an otherwise in-order counter. While any of them is outstanding, a wait value other than 0 is invalid. |
| `kinds` | Op kinds (`ir.MEMORY_KINDS`) that this counter counts. When it is present, a queue that carries other kinds is an error. |
| `tracks` | Free-text description. |

### Matrix hazard entry (`matrix_hazards`)

Rule keys:

| Rule | Producer -> consumer |
|------|----------------------|
| `valu_write_to_matrix_read` | VALU/trans writes a VGPR -> a matrix op reads it |
| `matrix_to_matrix_srcc_exact` | A matrix result -> the next matrix op accumulates into the same C/D (same instruction, exact overlap) |
| `matrix_to_matrix_srcc_overlap` | A matrix result -> a matrix op reads it as C under any other overlap |
| `matrix_to_matrix_srcab` | A matrix result -> a matrix op reads it as A/B |
| `matrix_to_valu` | A matrix result -> VALU/trans reads it (RAW) or writes it (WAW) |
| `matrix_to_mem` | A matrix result -> VMEM/DS/FLAT reads it (for example as store data) |
| `matrix_src_to_valu_write` | A matrix op reads A/B/C -> VALU overwrites that VGPR (WAR) |

Fields:

| Key | Meaning |
|-----|---------|
| `wait` | Required independent instructions (wait states) between producer and consumer, or `null`. |
| `by_passes` | Optional `{"<passes>": wait}`, chosen by the producer's `mfma[...].passes`. |
| `by_op` | Optional `{"<instruction>": wait}`, which takes precedence. |
| `interlock` | `true` if the hardware stalls the consumer by itself, so a short distance only costs time. `false` if software must cover the distance, so a short distance is a correctness error. `null` if this is unknown. |
| `source` / `confidence` / `note` | As for facts. |

When the needed `wait` or `interlock` is `null`, the hazard check reports it as unknown and
does not guess.
