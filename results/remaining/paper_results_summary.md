# PADOC Paper Experiment Summary

This file is the consolidated result sheet for writing the paper. It combines the original main experiment tables in `EXPERIMENTS.md` with the completed remaining experiments under `results/remaining/`.

Current status: the core experiment backlog is complete for the available datasets. The remaining non-core items are new MoE/ViT traces, optional multi-threaded analysis, and optional straggler-style analyses.

## 1. Main Claims Supported By The Data

1. PADOC compresses large AI profiler traces to roughly 24-31x while preserving a queryable structural template representation.
2. PADOC is not always the smallest byte stream compared with ScalaTrace, but it keeps enough structure to answer analysis tasks without materializing the full raw trace.
3. In-situ analysis is faster than reconstruct-then-scan baselines on the historical baseline-comparison tasks; the final core suite now emphasizes model/rank/structure-aware access patterns.
4. The new layer-aware GPU tasks validate the structural claim directly: with CPU-GPU kernel links present, GPU kernels can be attributed to CPU model/repeated scopes; with `no_kernel_links`, the layer-aware result rows disappear.
5. The representation scales to 301M events / 1024 ranks (`llama_full`) in a single merged artifact and answers the current five core PADOC tasks in 110-169 seconds total per task, dominated by artifact load/deserialization plus interval merging for layer-aware overlap.
6. Parallel compression has a clear saturation point: fastest measured `llama_full` compression is 32 workers, after which 64 workers regresses.
7. The ablation data supports the co-design story: smaller on-disk variants can be slower and use more memory, so the right claim is not "every PADOC feature minimizes compressed bytes"; the supported claim is "PADOC preserves analysis-ready structure while maintaining competitive compression."

## 2. Datasets

| Dataset | Workload | Ranks / GPUs | Events | Raw size |
|---|---|---:|---:|---:|
| `leworldmodel_full` | LeWorldModel inference | 2 | 3,469,389 | 884.37 MiB |
| `qwen3_full` | Qwen3 dense training | 256 | 33,813,574 | 6.91 GiB |
| `unifolm_full` | UniFolm world-model training | 4 | 80,223,071 | 22.43 GiB |
| `llama_full` | LLaMA-70B training | 1024 | 301,288,116 | 69.95 GiB |

## 3. Compression Results

Use this as the main PADOC compression table. The `best workers` column comes from the completed v6 thread sweep in `compress_scalability_full.md`.

| Dataset | PADOC artifact | Ratio | Best workers | Best PADOC compress time | Throughput |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.52 MiB | 23.57x | 2 | 13.687 s | 64.6 MB/s |
| `qwen3_full` | 272.23 MiB | 26.00x | 16 | 38.413 s | 184.3 MB/s |
| `unifolm_full` | 741.08 MiB | 31.00x | 16 | 199.686 s | 115.0 MB/s |
| `llama_full` | 2.40 GiB | 29.18x | 32 | 357.691 s | 200.2 MB/s |

Baseline compression comparison from `EXPERIMENTS.md`:

| Dataset | PADOC ratio | ScalaTrace ratio | TraceZip ratio | gzip_json ratio | raw_json ratio |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 23.57x | 60.97x | 32.76x | 21.31x | 1.21x |
| `qwen3_full` | 26.00x | 34.30x | 26.59x | 17.71x | 1.27x |
| `unifolm_full` | 31.00x | 82.39x | 47.50x | 27.70x | 1.25x |
| `llama_full` | 29.18x | 34.94x | 28.24x | 21.59x | 1.30x |

Interpretation for the paper:

PADOC is competitive with trace-compression baselines on size and beats JSON/gzip-style storage, but the key advantage is not pure byte ratio. ScalaTrace can be smaller because it discards or regularizes structure more aggressively; PADOC keeps structural templates, typed columns, rank trees, and CPU-GPU provenance links so analyses can run in situ.

## 4. Analysis Results

### 4.1 Core Analysis Tasks

The current core analysis set is:

| Task | Purpose | Structural dependency |
|---|---|---|
| `operator_hotspot` | Top operator/kernel templates by total duration | template columns |
| `rank_load_balance` | GPU compute/communication balance across ranks | rank-rooted node tree |
| `layer_kernel_hotspot` | Hot GPU kernels inside each model/repeated scope | CPU tree + CPU-GPU kernel links |
| `layer_compute_comm_overlap` | Per-layer/scope compute vs communication overlap | CPU tree + CPU-GPU kernel links |
| `layer_rank_balance` | Per-layer/scope load imbalance across ranks | CPU tree + CPU-GPU kernel links |

`stream_load_balance`, `layer_operator_balance`, and global `compute_comm_overlap` are no longer core experiments. They are retained as historical/background results only: stream balance is not a model-balance question, `layer_operator_balance` is mainly name-pattern based, and global overlap is superseded by the layer-aware overlap task.

Source: `results/remaining/core_layer_analysis.tsv`. Columns combine read + compressed-deserialize as the load cost.

| Dataset | Read + deserialize | Max analyze time | Slowest core task | Total time range | Peak RSS |
|---|---:|---:|---|---:|---:|
| `leworldmodel_full` | 1.460 s | 0.446 s | `layer_rank_balance` | 1.462-1.905 s | 0.55 GiB |
| `qwen3_full` | 11.659 s | 5.948 s | `layer_compute_comm_overlap` | 11.672-17.607 s | 5.04 GiB |
| `unifolm_full` | 37.138 s | 8.605 s | `layer_compute_comm_overlap` | 37.173-45.743 s | 14.24 GiB |
| `llama_full` | 109.766 s | 59.063 s | `layer_compute_comm_overlap` | 109.858-168.829 s | 34.32 GiB |

The layer-aware tasks are intentionally heavier than template-only hotspot analysis because they walk the structural tree and attribute GPU kernels back to CPU model/repeated scopes. Even then, the largest `llama_full` artifact remains analyzable as one merged 1024-rank trace; total time is still dominated by loading/deserializing the 2.40 GiB PADOC artifact plus the interval merge for layer-aware overlap.

### 4.2 Kernel-Link Ablation For Layer-Aware Queries

Source: `results/remaining/core_kernel_link_coverage.tsv`. This is the key ablation for the structure-aware analysis claim. With normal PADOC, layer-aware tasks can attribute GPU kernels to CPU model/repeated scopes. With `padoc_no_kernel_links`, the same tasks return zero attributed GPU refs and zero result rows because the CPU-GPU provenance edge is missing.

| Dataset | Default attributed GPU refs | Default coverage | `no_kernel_links` attributed refs | Result |
|---|---:|---:|---:|---|
| `leworldmodel_full` | 4,637 / 29,589 | 15.67% | 0 / 29,589 | layer-aware rows disappear |
| `qwen3_full` | 1,592,830 / 1,806,096 | 88.19% | 0 / 1,806,096 | layer-aware rows disappear |
| `unifolm_full` | 449,519 / 7,953,432 | 5.65% | 0 / 7,953,432 | layer-aware rows disappear |

Use `qwen3_full` as the main ablation example in the paper because its profiler scopes expose a high-coverage repeated model structure. `leworldmodel_full` and `unifolm_full` still validate the mechanism, but their traces contain more initialization / utility / framework-level GPU work that does not sit under clean repeated model scopes, so the attributable fraction is lower.

Timing source: `results/remaining/core_kernel_link_ablation.tsv`. The no-link timing should not be interpreted as a useful fast path: the query still walks CPU structure to search for layer scopes, but it cannot produce layer-aware GPU rows without kernel links. The semantic ablation is therefore coverage/row-count, not analysis-time reduction.

### 4.3 Historical Cross-Compressor Speedups

The original baseline comparison covered four earlier tasks: `operator_hotspot`, `stream_load_balance`, `layer_operator_balance`, and `rank_load_balance`. Keep this table only as background evidence that PADOC's template/tree representation is faster than reconstruct-then-scan baselines; do not present the non-core tasks as the final analysis suite.

| Dataset | operator_hotspot total speedup | stream_load_balance total speedup | layer_operator_balance total speedup | rank_load_balance total speedup |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 2.6x | 2.0x | 2.3x | 2.0x |
| `qwen3_full` | 2.2x | 1.6x | 1.9x | 1.7x |
| `unifolm_full` | 3.0x | 2.3x | 2.6x | 2.3x |
| `llama_full` | 4.0x | 3.0x | 3.5x | 3.2x |

For `llama_full`, historical analyze-only speedups against the best baseline were:

| Task | PADOC analyze-only | Best baseline analyze-only | Speedup |
|---|---:|---:|---:|
| `operator_hotspot` | 0.082 s | 106.69 s | 1,301x |
| `stream_load_balance` | 0.356 s | 0.347 s | 1.0x |
| `layer_operator_balance` | 0.0005 s | 51.84 s | 103,680x |
| `rank_load_balance` | 2.086 s | 19.30 s | 9.3x |

Do not claim cross-compressor speedups for the three new layer-aware GPU tasks unless those baselines are rerun with equivalent CPU-GPU attribution logic. The completed core evidence is PADOC in-situ performance plus the `no_kernel_links` semantic ablation.

## 5. On-Disk Storage Breakdown

Source: `on_disk_breakdown.txt`, generated by `inspect_artifact --on-disk`.

| Dataset | Artifact | Dominant on-disk regions |
|---|---:|---|
| `leworldmodel_full` | 37.52 MiB | args columns 20.50 MB, node instance refs 13.57 MB, rank node tree 12.95 MB, timestamp columns 4.71 MB |
| `qwen3_full` | 272.23 MiB | timestamp columns 125.50 MB, node instance refs 100.64 MB, rank node tree 100.15 MB, args columns 44.78 MB |
| `unifolm_full` | 741.08 MiB | args columns 352.40 MB, node instance refs 264.15 MB, rank node tree 258.25 MB, timestamp columns 144.34 MB |
| `llama_full` | 2.40 GiB | timestamp columns 1.07 GB, rank node tree 946.11 MB, node instance refs 925.15 MB, args columns 295.25 MB |

For `llama_full`, the main on-disk contributors are:

| Region | zstd bytes | Share of artifact |
|---|---:|---:|
| `ts_columns` | 1,073,867,885 | 41.7% |
| `rank_node_tree` | 946,110,875 | 36.8% |
| `node_instance_refs` | 925,145,515 | 35.9% |
| `args_columns` | 295,247,430 | 11.5% |
| `ids_pids_phases_streams` | 153,426,779 | 6.0% |
| `dur_columns` | 101,498,433 | 3.9% |
| `name_nums` | 2,715,745 | 0.1% |

The shares do not sum to 100% because each region is encoded independently for attribution; it is a contribution profile, not a partition of the exact final zstd stream. The historical region name `node_soft_links` is better interpreted as node instance/reference arrays, not only CPU-GPU kernel links.

## 6. In-Memory Representation

The main in-memory table from `EXPERIMENTS.md`:

| Dataset | Templates | Constant cols | i32 cols | i64 cols | Accounted in-memory |
|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 4,094 | 1,748 | 6,511 | 0 | 0.28 GiB |
| `qwen3_full` | 5,498 | 4,880 | 6,194 | 0 | 2.53 GiB |
| `unifolm_full` | 16,897 | 5,147 | 28,909 | 0 | 6.72 GiB |
| `llama_full` | 312 | 4 | 716 | 0 | 22.10 GiB |

Interpretation:

All surviving numeric columns are either constants or i32 after timestamp normalization; no i64 column remains in these artifacts. This supports the typed-column compaction claim. On `llama_full`, the large resident footprint is now mostly the structural node tree and its child-index storage, not untyped numeric columns.

## 7. Ablation Results

Source files:

- `ablation_storage_from_artifacts.tsv`: 3 datasets x 8 PADOC presets.
- `ablation_analyze.tsv`: historical 3 datasets x 8 presets x 5 tasks = 120 rows.
- `core_kernel_link_ablation.tsv`: current core task timing for default vs `no_kernel_links`.
- `core_kernel_link_coverage.tsv`: semantic coverage ablation for the three layer-aware GPU tasks.

Storage ablation summary:

| Dataset | Default PADOC | Smallest preset | Largest preset | Range |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 37.5 MiB / 23.57x | `padoc_minimal`, 36.0 MiB / 24.54x | `padoc_no_args_dedup`, 37.6 MiB / 23.55x | narrow |
| `qwen3_full` | 272.2 MiB / 25.99x | `padoc_no_kernel_links`, 259.9 MiB / 27.23x | `padoc_no_anchor`, 275.5 MiB / 25.69x | narrow |
| `unifolm_full` | 741.1 MiB / 30.99x | `padoc_minimal`, 672.5 MiB / 34.16x | `padoc_no_args_dedup`, 743.0 MiB / 30.91x | moderate |

Analysis ablation examples for `operator_hotspot`:

| Dataset | Preset | Artifact | Deserialize/decompress | Analyze | Total | RSS |
|---|---|---:|---:|---:|---:|---:|
| `leworldmodel_full` | default | 37.5 MiB | 1.4287 s | 0.0024 s | 1.4853 s | 0.55 GiB |
| `leworldmodel_full` | minimal | 36.0 MiB | 2.5881 s | 0.0026 s | 2.6379 s | 1.40 GiB |
| `qwen3_full` | default | 272.2 MiB | 11.1339 s | 0.0100 s | 11.5418 s | 5.04 GiB |
| `qwen3_full` | minimal | 265.5 MiB | 19.2806 s | 0.0103 s | 19.6071 s | 10.60 GiB |
| `unifolm_full` | default | 741.1 MiB | 35.0344 s | 0.0287 s | 36.2310 s | 14.24 GiB |
| `unifolm_full` | minimal | 672.5 MiB | 72.7825 s | 0.0284 s | 73.6399 s | 41.35 GiB |

Interpretation:

The minimal preset can be smaller on disk, but it is substantially slower to load and can use much more memory. This is a strong result for the paper because it supports the design goal: PADOC is an analysis-ready compressed representation, not only a byte-minimizing codec.

## 8. Scalability

### 8.1 GPU Count

Source: `gpu_scalability.md`, using llama subsets at 32 workers.

| GPUs | Events | Raw size | Artifact | Ratio | Compress time |
|---:|---:|---:|---:|---:|---:|
| 1 | 316,746 | 74.81 MiB | 2.99 MiB | 25.03x | 1.737 s |
| 8 | 2,607,995 | 622.99 MiB | 23.35 MiB | 26.68x | 4.665 s |
| 64 | 19,544,859 | 4.55 GiB | 165.23 MiB | 28.19x | 48.408 s |
| 256 | 75,749,224 | 17.59 GiB | 621.61 MiB | 28.97x | 115.421 s |

Interpretation:

Raw size and artifact size grow roughly with GPU count, while the compression ratio improves from 25.03x to 28.97x as more repeated cross-rank structure becomes available.

### 8.2 Compression Threads

Source: `compress_scalability_full.md`.

| Workers | `qwen3_full` time | `unifolm_full` time | `llama_full` time |
|---:|---:|---:|---:|
| 1 | 290.937 s | 570.182 s | 3153.696 s |
| 2 | 159.692 s | 321.551 s | 1648.617 s |
| 4 | 88.036 s | 200.491 s | 933.970 s |
| 8 | 49.993 s | 203.894 s | 562.100 s |
| 16 | 38.413 s | 199.686 s | 397.718 s |
| 32 | 43.288 s | 206.084 s | 357.691 s |
| 64 | 60.492 s | 206.789 s | 452.036 s |

Interpretation:

Parallel compression scales well up to 16-32 workers depending on dataset and then regresses. For `llama_full`, 32 workers is the best point; 64 workers likely loses to scheduling, memory bandwidth, merge, and serialization overheads.

### 8.3 Synthetic Layers / Iterations

Source: `synthetic_scalability.md`.

| Sweep | Values | Result |
|---|---|---|
| Layers | 8, 16, 32, 64, 128 | Events scale linearly from 1,216 to 19,456; ratio stays around 28-30x. |
| Iterations | 1, 2, 4, 8, 16 | Events scale linearly from 304 to 4,864; ratio improves from 20.47x to 30.42x. |

Interpretation:

The synthetic sweeps provide the expected scalability shape: increasing repeated structure improves or stabilizes the compression ratio, and artifact size grows linearly with event count.

## 9. Analysis Profiling

Source: `analysis_profile_padoc.tsv` for the historical profile, plus `core_layer_analysis.tsv` for the current core task timings.

Representative current `llama_full` analyze-only times:

| Task | Analyze time | Main work |
|---|---:|---|
| `operator_hotspot` | 0.091 s | template tally |
| `rank_load_balance` | 2.074 s | rank tree walk |
| `layer_kernel_hotspot` | 23.620 s | layer/scope attribution over CPU-GPU links |
| `layer_compute_comm_overlap` | 59.063 s | layer/scope attribution plus interval merge |
| `layer_rank_balance` | 31.066 s | layer/scope attribution plus per-rank summaries |

Interpretation:

The profile supports the mechanism section: template-only tasks are very fast, rank tasks are tree-walk bounded, and layer-aware GPU tasks intentionally pay more work to attribute kernels back to repeated CPU scopes. `layer_compute_comm_overlap` is the heaviest because it also constructs and merges intervals.

## 10. Data Quality / Sanity Checks

These checks were performed before this summary was written:

| Check | Result |
|---|---|
| `padoc_5task_analysis.tsv` row count | 20 rows = historical 4 datasets x 5 tasks |
| `ablation_analyze.tsv` row count | 120 rows = historical 3 datasets x 8 presets x 5 tasks |
| `core_layer_analysis.tsv` row count | 20 rows = 4 datasets x 5 current core tasks |
| `core_kernel_link_coverage.tsv` row count | 18 rows = 3 datasets x 2 presets x 3 layer-aware tasks |
| `ablation_storage_from_artifacts.tsv` row count | 24 artifact rows + header |
| `analysis_profile_padoc.jsonl` row count | 20 rows = historical 4 datasets x 5 tasks |
| GPU scalability sections | 4 sections = 1/8/64/256 GPUs |
| Thread scalability sections | 14 sections = 7 worker points for small datasets + 7 for llama |
| Default PADOC artifact bytes in storage vs analysis ablation | exact match for leworldmodel/qwen3/unifolm |
| `cargo test` | passed: 17 library tests + 14 end-to-end tests |
| Worktree before result-summary commit | only result/docs files pending |

The data is internally consistent and has the expected qualitative behavior:

- Compression ratio is stable or improves with larger repeated traces.
- Thread scaling improves until saturation, then regresses at excessive parallelism.
- Analysis time is usually dominated by load/deserialization; layer-aware overlap also pays a real interval-merge cost, especially on `llama_full`.
- Minimal/smaller artifacts can cost more memory/time, which is consistent with the co-design tradeoff.
- The largest dataset (`llama_full`) remains analyzable in one merged PADOC artifact with about 34 GiB peak RSS.
- The kernel-link ablation is semantically strong: default PADOC produces layer-aware GPU rows, while `no_kernel_links` produces zero attributed GPU refs and zero rows on all three small datasets.

## 11. Caveats To Keep In The Paper Honest

1. Do not claim PADOC is always the smallest compressor. ScalaTrace often has a better byte ratio, but it does not preserve the same analysis-ready structure.
2. Do not claim cross-compressor speedup for the three new layer-aware GPU tasks unless the baseline matrix is rerun with equivalent CPU-GPU attribution logic. Current cross-compressor speedups cover the historical task set.
3. Do not present `stream_load_balance`, `layer_operator_balance`, or global `compute_comm_overlap` as core analysis experiments; they are historical/background only.
4. Do not claim every ablation feature improves on-disk size. Some smaller variants are worse for analysis time and memory; frame ablation as the storage-analysis tradeoff.
5. Full llama-scale 8-preset ablation was not run because of cost and memory risk. The completed ablation is full for leworldmodel/qwen3/unifolm.
6. MoE and ViT rows are absent because the corresponding traces are not available locally.

## 12. Result File Index

Primary paper tables:

- `EXPERIMENTS.md`: original compression, memory, baseline-analysis, speedup, and lossless verification tables.
- `results/remaining/core_layer_analysis.tsv`: current five-task core PADOC analysis.
- `results/remaining/core_kernel_link_coverage.tsv`: default vs `no_kernel_links` coverage for layer-aware GPU tasks.
- `results/remaining/core_kernel_link_ablation.tsv`: default vs `no_kernel_links` timing for current core tasks.
- `results/remaining/padoc_5task_analysis.tsv`: historical PADOC five-task analysis, including global `compute_comm_overlap`.
- `results/remaining/on_disk_breakdown.txt`: on-disk region attribution.
- `results/remaining/ablation_storage_from_artifacts.tsv`: storage ablation.
- `results/remaining/ablation_analyze.tsv`: analysis ablation.
- `results/remaining/gpu_scalability.md`: GPU-count scalability.
- `results/remaining/compress_scalability_full.md`: full v6 thread scalability.
- `results/remaining/synthetic_scalability.md`: synthetic layers/iterations scalability.
- `results/remaining/analysis_profile_padoc.tsv`: compact analysis profiling table.
- `results/remaining/analysis_profile_padoc.jsonl`: original profiling JSONL, preserving task outputs and phase timings.

Conclusion:

The current data is reasonable and sufficient to support the main PADOC paper narrative: PADOC is a lossless, analysis-ready compressed trace representation that maintains competitive compression ratios while enabling much faster and lower-memory post-hoc analyses than reconstruct-then-scan baselines.
