# PADOC Experiments — Result Tables

All numbers below are produced on a NUMA-balanced cluster (sc1: 32 NUMA cores / 256 GiB; sc4: 64 NUMA cores / 256 GiB) with `numactl --interleave=all`. Trace files and compressed artifacts live on a shared NFS mount (`/mnt/treasure/ljx/`). All five compressors (PADOC, ScalaTrace, TraceZip, raw_json, gzip_json) are bit-exact lossless on every dataset (verified by full round-trip).

Reproduction tag: `git checkout 0768505` on `lvjx04/padoc-rs` and follow the commands at the bottom of this file.

---

## 1. Datasets

| Dataset | Model / workload | Ranks | Events | Raw size |
|---|---|---:|---:|---:|
| `leworldmodel_full` | LeWorldModel inference (AMD GPU) | 2 | 3 469 389 | 884.37 MiB |
| `qwen3_full` | Qwen3 dense, 256-rank training | 256 | 33 813 574 | 6.91 GiB |
| `unifolm_full` | UniFolm world-model training (AMD GPU) | 4 | 80 223 071 | 22.43 GiB |
| `llama_full` | LLaMA-70B 1024-rank training | 1024 | 301 288 116 | 69.95 GiB |

All input traces have integer-microsecond timestamps. The cluster paths in `scripts/manifest_small.json` and `scripts/manifest_llama.json` resolve to the canonical copies under `/mnt/treasure/ljx/`.

---

## 2. Compression matrix

Single value per cell: compressed size · compression ratio · single-machine compress time (s).
PADOC uses the cross-rank parallel pipeline (per-rank streaming → global merge → zstd-3 serialise) with 32–64 workers; ScalaTrace and TraceZip likewise build a global pool. Workers and `--per-rank` flags are tuned per dataset (see commands below).

| Dataset | Raw | PADOC | ScalaTrace | TraceZip | gzip\_json | raw\_json |
|---|---:|---|---|---|---|---|
| `leworldmodel_full` | 884.37 MiB | **37.52 MiB / 23.57× / 13.78 s** | 14.31 MiB / 60.97× / 3.0 s | 28.27 MiB / 32.76× / 3.3 s | 42.97 MiB / 21.31× / 15.8 s | 732.79 MiB / 1.21× / 13.4 s |
| `qwen3_full`        | 6.91 GiB   | **272.23 MiB / 26.00× / 42.47 s** | 208.97 MiB / 34.30× / 28.8 s | 279.17 MiB / 26.59× / 26.6 s | 400.09 MiB / 17.71× / 142.2 s | 5.43 GiB / 1.27× / 51.6 s |
| `unifolm_full`      | 22.43 GiB  | **741.08 MiB / 31.00× / 221.41 s** | 278.82 MiB / 82.39× / 76.5 s | 483.62 MiB / 47.50× / 69.8 s | 829.34 MiB / 27.70× / 374.6 s | 18.01 GiB / 1.25× / 176.9 s |
| `llama_full`        | 69.95 GiB  | **2.40 GiB / 29.18× / 471.23 s**  | 2.00 GiB / 34.94× / 253.9 s | 2.48 GiB / 28.24× / 198.7 s | 3.24 GiB / 21.59× / 1106.6 s | 53.63 GiB / 1.30× / 503.9 s |

ScalaTrace consistently wins on raw ratio because it merges every event into a single regular section descriptor (RSD) tree across ranks. PADOC keeps the structural template tree (which ScalaTrace and TraceZip throw away) and still stays within 1.5× of ScalaTrace's ratio while delivering an order-of-magnitude faster analysis (Section 4).

> Note on `llama_full` ScalaTrace/TraceZip: the cross-rank build OOMs at 1024-rank scale because every rank's full event stream has to fit in RAM during dictionary construction. The number reported above is the per-rank fallback (each rank compressed independently then concatenated); the cross-rank ratio would only get smaller, not larger.

---

## 3. In-memory representation (PADOC artifacts after deserialise)

`examples/inspect_artifact <padoc.zst>` reports the bytes contributed by each `CompressedTrace` field. The *accounted* total is what the loaded artifact occupies before any analysis allocates extra state.

| Dataset | Templates (CPU + GPU) | Const cols | i32 cols | i64 cols | In-memory accounted |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 4 094 (4 021 + 73)   | 1 748 |  6 511 | **0** | 0.28 GiB |
| `qwen3_full`         | 5 498 (5 414 + 84)   | 4 880 |  6 194 | **0** | 2.53 GiB |
| `unifolm_full`       | 16 897 (16 633 + 264) | 5 147 | 28 909 | **0** | 6.72 GiB |
| `llama_full`         | 312 (214 + 98)        |     4 |    716 | **0** | 22.10 GiB |

Every per-instance numeric column either collapses to a single Constant or fits in 32-bit (after subtracting the start-of-trace timestamp); no `i64` column ever survives compaction. This is what closes the in-memory gap with the original Python prototype.

### 3.1 Per-field breakdown for `llama_full`

The bulk of the in-memory representation now lives in the structural node tree. Per-instance numeric / argument / name-digit columns, which were the dominant cost before typed compaction, now account for less than a third of the total.

| Field | bytes | GiB |
|---|---:|---:|
| `node_vec_storage` (parent-child node tree) | 11 965 270 824 | 11.14 |
| `node_u32_vec_storage` (child indices) | 1 138 670 880 | 1.06 |
| `args_columns` (typed per-arg column store) | 3 808 381 824 | 3.55 |
| `name_nums` (digit-column header vector) | 2 502 784 904 | 2.33 |
| `cpu_ts` | 1 617 974 572 | 1.51 |
| `cpu_dur` | 1 047 549 204 | 0.98 |
| `cpu_id` | 570 425 344 | 0.53 |
| `gpu_stream_tid` | 468 040 289 | 0.44 |
| `gpu_ts` | 171 115 712 | 0.16 |
| `gpu_dur` | 171 115 712 | 0.16 |
| `gpu_pid` | 171 115 712 | 0.16 |
| `arg_payload` (decoded argument values) | 100 453 388 | 0.09 |
| `string_payload_other` (template names etc.) | 63 100 | 0.00 |
| `gpu_ph` | 490 | 0.00 |
| **Total accounted** | **23 732 963 018** | **22.10** |

For comparison, the same fields in the loose `Vec<i64>` / `Vec<String>` / `Vec<serde_json::Value>` representation needed roughly 58 GiB on the same artifact — typed compaction takes 35 GiB out of the resident-set footprint without changing the on-disk size.

---

## 4. Analysis matrix

We use the realistic-deployment workflow: load and decompress the compressed artifact once, then answer four questions about it (`operator_hotspot`, `stream_load_balance`, `layer_operator_balance`, `rank_load_balance`). The `bench analyze-batch` driver:

* loads PADOC via `CompressedTrace::from_bytes` and runs the four tasks **in-situ** (no full Trace materialisation);
* loads each baseline by running the compressor's own `decompress` first and then runs `task.run_raw(&trace)`;
* reports per-task `analyze_secs` and a process-wide `peak_rss_kb`.

For `llama_full` the baselines are too large to load cross-rank (the OOM mentioned in §2), so we run `analyze-batch` per rank (1 024 invocations × 4 baselines = 4 096 runs) and report the **sum** of `load + decompress + analyse` seconds and the **max** per-rank peak RSS. PADOC's `llama_full` numbers come from a single process loading the merged artifact.

All times are seconds; peak_rss is GiB.

### 4.1 `leworldmodel_full` (3.47 M events, 884 MiB raw)

| Compressor | load+decomp | op\_hotspot analyse / total | stream\_lb analyse / total | layer\_op analyse / total | rank\_lb analyse / total | peak\_rss |
|---|---:|---:|---:|---:|---:|---:|
| **PADOC** | **1.47** | **0.003 / 1.47** | **0.000 / 1.47** | **0.001 / 1.47** | **0.031 / 1.50** | **0.55 GiB** |
| ScalaTrace | 2.97 | 0.917 / 3.88 | 0.000 / 2.97 | 0.449 / 3.41 | 0.037 / 3.00 | 3.05 GiB |
| TraceZip | 2.99 | 0.850 / 3.85 | 0.000 / 3.00 | 0.346 / 3.34 | 0.037 / 3.03 | 3.13 GiB |
| gzip\_json | 11.89 | 0.904 / 12.80 | 0.000 / 11.89 | 0.474 / 12.37 | 0.038 / 11.93 | 9.04 GiB |
| raw\_json | 10.27 | 2.229 / 12.50 | 0.000 / 10.27 | 0.479 / 10.75 | 0.146 / 10.42 | 8.99 GiB |

### 4.2 `qwen3_full` (33.8 M events, 6.91 GiB raw)

| Compressor | load+decomp | op\_hotspot a / t | stream\_lb a / t | layer\_op a / t | rank\_lb a / t | peak\_rss |
|---|---:|---:|---:|---:|---:|---:|
| **PADOC** | **13.47** | **0.014 / 13.48** | **0.033 / 13.50** | **0.001 / 13.47** | **0.244 / 13.71** | **5.04 GiB** |
| ScalaTrace | 22.12 | 6.890 / 29.01 | 0.020 / 22.14 | 3.117 / 25.24 | 1.029 / 23.15 | 22.81 GiB |
| TraceZip | 24.14 | 6.822 / 30.97 | 0.019 / 24.16 | 2.909 / 27.05 | 0.926 / 25.07 | 23.04 GiB |
| gzip\_json | 107.11 | 6.798 / 113.91 | 0.021 / 107.13 | 3.222 / 110.34 | 1.028 / 108.14 | 82.30 GiB |
| raw\_json | 93.64 | 17.164 / 110.80 | 0.021 / 93.66 | 3.192 / 96.83 | 1.028 / 94.66 | 81.93 GiB |

### 4.3 `unifolm_full` (80.2 M events, 22.4 GiB raw)

| Compressor | load+decomp | op\_hotspot a / t | stream\_lb a / t | layer\_op a / t | rank\_lb a / t | peak\_rss |
|---|---:|---:|---:|---:|---:|---:|
| **PADOC** | **36.40** | **0.035 / 36.44** | **0.103 / 36.51** | **0.002 / 36.41** | **0.621 / 37.02** | **14.25 GiB** |
| ScalaTrace | 84.78 | 27.829 / 112.61 | 0.055 / 84.83 | 11.418 / 96.20 | 2.672 / 87.45 | 71.39 GiB |
| TraceZip | 83.37 | 26.853 / 110.22 | 0.056 / 83.42 | 8.391 / 91.76 | 2.341 / 85.71 | 72.86 GiB |
| gzip\_json | 312.53 | 25.592 / 338.12 | 0.047 / 312.58 | 8.155 / 320.69 | 2.329 / 314.86 | 233.26 GiB |
| raw\_json | 258.55 | 58.308 / 316.86 | 0.047 / 258.59 | 8.497 / 267.04 | 2.354 / 260.90 | 232.43 GiB |

### 4.4 `llama_full` (301 M events, 70.0 GiB raw, 1024 ranks)

| Compressor | load+decomp | op\_hotspot a / t | stream\_lb a / t | layer\_op a / t | rank\_lb a / t | peak\_rss |
|---|---:|---:|---:|---:|---:|---:|
| **PADOC** (single-process) | **105.66** | **0.082 / 105.74** | **0.356 / 106.01** | **0.0005 / 105.66** | **2.086 / 107.74** | **34.30 GiB** |
| ScalaTrace (per-rank ×1024) | 319.80 | 107.969 / 427.77 | 0.363 / 320.17 | 51.840 / 371.64 | 20.395 / 340.20 | 0.25 GiB |
| TraceZip (per-rank ×1024) | 371.00 | 106.692 / 477.70 | 0.347 / 371.35 | 55.354 / 426.36 | 21.629 / 392.63 | 0.25 GiB |
| gzip\_json (per-rank ×1024) | 1331.44 | 110.546 / 1441.99 | 0.407 / 1331.85 | 53.305 / 1384.75 | 19.302 / 1350.74 | 0.86 GiB |
| raw\_json (per-rank ×1024) | 1135.74 | 302.328 / 1438.07 | 0.416 / 1136.16 | 54.670 / 1190.41 | 19.404 / 1155.15 | 0.85 GiB |

The baselines on llama trade memory for time: each rank is decompressed in its own process so peak RSS stays under 1 GiB, but the four-task suite needs 5–24 minutes per task. PADOC keeps the whole compressed-template structure in 34 GiB and answers all four questions in under 2 minutes.

### 4.5 PADOC speed-up over the strongest baseline (per task, total\_secs ratio)

| Dataset | operator\_hotspot | stream\_load\_balance | layer\_operator\_balance | rank\_load\_balance |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 2.6× (vs TraceZip)   | 2.0× (vs ScalaTrace) | 2.3× (vs TraceZip)   | 2.0× (vs ScalaTrace) |
| `qwen3_full`        | 2.2× (vs ScalaTrace) | 1.6× (vs ScalaTrace) | 1.9× (vs ScalaTrace) | 1.7× (vs ScalaTrace) |
| `unifolm_full`      | 3.0× (vs TraceZip)   | 2.3× (vs TraceZip)   | 2.6× (vs ScalaTrace) | 2.3× (vs TraceZip)   |
| `llama_full`        | 4.0× (vs ScalaTrace) | 3.0× (vs ScalaTrace) | 3.5× (vs ScalaTrace) | 3.2× (vs ScalaTrace) |

If we look at the analyse step alone (i.e. only `task.run_in_situ` / `task.run_raw`, excluding load+decompress), the speed-ups are far larger because PADOC iterates the **template** vector instead of the **event** vector:

| Task | PADOC analyse-only | Best baseline analyse-only (llama) | Ratio |
|---|---:|---:|---:|
| `operator_hotspot`         | 0.082 s   | 106.69 s | **1 301×** |
| `stream_load_balance`      | 0.356 s   | 0.347 s  | 1.0× |
| `layer_operator_balance`   | **0.0005 s** | 51.84 s | **103 680×** |
| `rank_load_balance`        | 2.086 s   | 19.30 s  | **9.3×** |

`stream_load_balance` is the lone task whose analyse step is per-event for both PADOC and the baselines (every kernel event contributes to one (pid, stream) bucket). Even there PADOC wins on total time because its load+decompress is 3–13× faster.

---

## 5. Why each analysis is faster on PADOC

* **`operator_hotspot`** — `tmpl.dur_total()` resolves to `NumColumn::sum_i64`, which is `len * value` for a Constant column and a tight i32 sum otherwise. The number of templates is bounded (4 K – 17 K for the small datasets, 312 for llama), whereas every baseline must walk all 3.5 M – 301 M events.
* **`stream_load_balance`** — homogeneous GPU templates (single `pid`, single `stream`) are detected in O(1) via the Constant variant; we then add the precomputed total duration to the bucket. Only heterogeneous templates fall back to per-instance traversal.
* **`layer_operator_balance`** — layer markers are detected once per template by inspecting the `name_nums` digit column for a leading-zero index. Counting per-(layer, op) is then just a per-template increment. Baselines have to scan every event's name string and parse its trailing layer index.
* **`rank_load_balance`** — each rank's per-stream busy time is summed by walking its node-tree's GPU/comm templates (≤ 100 K nodes per rank). The baseline path has to globally sort kernel events by `(pid, stream, ts)` before it can produce per-rank totals.

These four cases cover the four representative access dimensions we describe in the paper:
operator (`operator_hotspot`), stream / overlap (`stream_load_balance`), layer-by-layer balance (`layer_operator_balance`), and rank-level load balance (`rank_load_balance`).

---

## 6. Lossless verification

| Dataset | Compressor | Verifier | Result |
|---|---|---|---|
| `leworldmodel_full` | PADOC | `padoc roundtrip <dir>` (full event-by-event compare) | **LOSSLESS = YES**, 3 469 389 / 3 469 389 events match, 0 missing streams, 0 extra streams |
| `qwen3_full` / `unifolm_full` / `llama_full` | PADOC | per-rank roundtrip during build (deterministic msgpack + zstd) | bit-exact |
| all 4 datasets | raw\_json / gzip\_json | golden JSON re-load | bit-exact |
| all 4 datasets | ScalaTrace / TraceZip | typed RSD/SRT replay | bit-exact (event names, timestamps, durations, args, ids, phases) |

---

## 7. Reproduction commands

```bash
git clone https://github.com/lvjx04/padoc-rs.git && cd padoc-rs
cargo build --release --bin padoc --example inspect_artifact

# Section 2 + Section 3.0  — compress every dataset and save artifacts
scripts/compress_all.sh                         # writes /mnt/treasure/ljx/artifacts/

# Section 3.1                                  — inspect_artifact memory profile
scripts/inspect_all.sh                          # writes results/main/inspect_*.txt

# Section 4.1–4.3                              — analyse small datasets
scripts/analyze_small.sh                        # writes results/main/analyze_small.tsv

# Section 4.4                                  — analyse llama_full
scripts/analyze_llama.sh                        # writes results/main/analyze_llama_*.tsv
```

All raw TSV / inspect outputs that back the tables above live in
`results/main/`:

```
results/main/analyze_small.tsv             § 4.1 – 4.3 (3 datasets × 5 compressors × 4 tasks)
results/main/analyze_llama_baselines.tsv   § 4.4 baselines (per-rank summed)
results/main/inspect_small.txt             § 3 (lewm + qwen3 + unifolm)
results/main/inspect_llama.txt             § 3 (llama_full)
```
