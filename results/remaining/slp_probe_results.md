# Segmented Linear Timestamp/Duration Probe

This is the offline validation for adding an integer segmented-linear residual
encoding for `ts` and `dur` columns. It does not change the PADOC artifact
format yet. The detailed per-column TSV is:

`results/remaining/slp_probe/slp_probe_columns.tsv`

The generated aggregate table is:

`results/remaining/slp_probe/slp_probe_summary.md`

## Method

The probe uses a fixed-point integer predictor:

```text
pred(i) = base + round(slope_q * (i - start) / 2^32)
value(i) = pred(i) + residual(i)
```

No floating point coefficients are stored. Each segment keeps `end`, `base`,
and `slope_q`. Residuals are tested as `i8` and `i16`. Segments are built by a
greedy interval-feasibility scan: while extending a segment, the encoder keeps
the integer slope interval for which every residual stays within the chosen
bound (`127` for `i8`, `32767` for `i16`). When the interval becomes empty, the
segment ends. This is near-linear in the sampled column length and avoids an
expensive full dynamic program during the first validation pass.

The acceptance criterion for a future in-format implementation should be:

```text
best_slp_ratio_vs_i64 > 2.0
best_slp_mem_bytes < current_i32_mem_bytes
```

Columns that do not satisfy both conditions should fall back to the existing
`I32` representation. This matters because a few irregular timestamp columns
produce many small segments and are worse than `I32`.

## Probe Settings

| Setting | Value |
|---|---:|
| datasets | `leworldmodel_full`, `qwen3_full`, `unifolm_full`, `llama_full` |
| columns per dataset | top 128 longest `ts`/`dur` columns |
| min column length | 1,024 |
| max sampled values per column | 1,000,000 |
| max segment length | 65,536 |
| fixed-point scale | `Q = 32` |
| zstd level for file-size proxy | 3 |

## Aggregate Result

| Dataset | Columns | Sampled values | Best SLP vs int64 memory | Best SLP vs int32 memory | Hybrid vs int64 memory | Hybrid vs int32 memory | SLP accepted cols | Fallback cols | Encode time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 128 | 3,906,828 | 7.06x | 3.53x | 7.06x | 3.53x | 128 | 0 | 0.153 s |
| `qwen3_full` | 128 | 38,097,496 | 4.05x | 2.03x | 4.08x | 2.04x | 118 | 10 | 1.620 s |
| `unifolm_full` | 128 | 58,928,428 | 5.48x | 2.74x | 5.87x | 2.93x | 116 | 12 | 2.333 s |
| `llama_full` | 128 | 93,249,920 | 3.97x | 1.98x | 4.04x | 2.02x | 126 | 2 | 4.087 s |

`Best SLP` means always choosing the smaller of `i8` and `i16` SLP for each
column. `Hybrid` means using SLP only when it beats both the `>2x vs int64`
threshold and the current `I32` memory estimate; otherwise the column remains
`I32`. Hybrid is the recommended implementation policy.

All 512 probed columns passed decode verification for both `i8` and `i16`.

## `ts` vs `dur`

| Dataset | Column | Columns | Best SLP vs int64 | Best SLP vs int32 | Beat int32 cols |
|---|---|---:|---:|---:|---:|
| `leworldmodel_full` | `ts` | 65 | 6.40x | 3.20x | 65 |
| `leworldmodel_full` | `dur` | 63 | 7.89x | 3.95x | 63 |
| `qwen3_full` | `ts` | 66 | 3.14x | 1.57x | 56 |
| `qwen3_full` | `dur` | 62 | 6.12x | 3.06x | 62 |
| `unifolm_full` | `ts` | 65 | 4.29x | 2.14x | 53 |
| `unifolm_full` | `dur` | 63 | 7.76x | 3.88x | 63 |
| `llama_full` | `ts` | 66 | 3.04x | 1.52x | 64 |
| `llama_full` | `dur` | 62 | 5.95x | 2.98x | 62 |

The weak cases are irregular `ts` columns. `dur` columns are consistently
strong, and all probed `dur` columns beat the `I32` memory estimate.

## File-Size Proxy

The TSV also records msgpack and zstd sizes for raw `i64`, raw `i32`, `SLP i8`,
and `SLP i16` projections. The current requirement is primarily in-memory
compression. File size is still recorded because the final artifact is zstd
wrapped and can react differently from memory layout.

For the aggregate best-SLP proxy, zstd bytes are usually lower than raw `i64`
bytes, but this should not be over-interpreted as final artifact size because
the probe encodes each sampled column independently and does not include the
full template/tree artifact context.

## Implementation Implication

The data supports implementing SLP as an optional `NumColumn` variant, but only
with a per-column fallback. A safe production path is:

1. Keep `Constant` detection first.
2. Try SLP `i8` and SLP `i16` on `ts` and `dur`.
3. Estimate in-memory bytes including segment metadata.
4. Use SLP only if it beats `I32` and exceeds 2x vs `I64`.
5. Otherwise keep current `I32`.

The full experiment after integration must rerun both memory and on-disk
breakdowns. The relevant existing breakdowns are `inspect_artifact`'s
in-memory accounting and `--on-disk` region projection; both should gain
separate rows for SLP timestamp/duration columns.
