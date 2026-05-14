# PADOC Paper Experiment Summary

This file is the consolidated result sheet for writing the paper. It combines the original main experiment tables in `EXPERIMENTS.md` with the completed remaining experiments under `results/remaining/`.

Current status: the core experiment backlog is complete for the available datasets. The remaining non-core items are new MoE/ViT traces, optional multi-threaded analysis, and optional straggler-style analyses.

## 1. Main Claims Supported By The Data

1. PADOC compresses large AI profiler traces to roughly 24-31x while preserving a queryable structural template representation.
2. PADOC is not always the smallest byte stream compared with ScalaTrace, but it keeps enough structure to answer analysis tasks without materializing the full raw trace.
3. In-situ analysis is consistently faster in total time than the strongest baseline on the four baseline-comparison tasks; on template-oriented tasks, the analyze-only speedup is orders of magnitude.
4. The representation scales to 301M events / 1024 ranks (`llama_full`) in a single merged artifact and answers the current five PADOC tasks in roughly 117-125 seconds total per task, dominated by artifact load/deserialization rather than the analysis logic itself.
5. Parallel compression has a clear saturation point: fastest measured `llama_full` compression is 32 workers, after which 64 workers regresses.
6. The ablation data supports the co-design story: smaller on-disk variants can be slower and use more memory, so the right claim is not "every PADOC feature minimizes compressed bytes"; the supported claim is "PADOC preserves analysis-ready structure while maintaining competitive compression."

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

PADOC is competitive with trace-compression baselines on size and beats JSON/gzip-style storage, but the key advantage is not pure byte ratio. ScalaTrace can be smaller because it discards or regularizes structure more aggressively; PADOC keeps structural templates, typed columns, rank trees, and soft links so analyses can run in situ.

## 4. Analysis Results

### 4.1 Cross-Compressor Speedups

The original baseline comparison covers four tasks: `operator_hotspot`, `stream_load_balance`, `layer_operator_balance`, and `rank_load_balance`.

| Dataset | operator_hotspot total speedup | stream_load_balance total speedup | layer_operator_balance total speedup | rank_load_balance total speedup |
|---|---:|---:|---:|---:|
| `leworldmodel_full` | 2.6x | 2.0x | 2.3x | 2.0x |
| `qwen3_full` | 2.2x | 1.6x | 1.9x | 1.7x |
| `unifolm_full` | 3.0x | 2.3x | 2.6x | 2.3x |
| `llama_full` | 4.0x | 3.0x | 3.5x | 3.2x |

For `llama_full`, analyze-only speedups against the best baseline are:

| Task | PADOC analyze-only | Best baseline analyze-only | Speedup |
|---|---:|---:|---:|
| `operator_hotspot` | 0.082 s | 106.69 s | 1,301x |
| `stream_load_balance` | 0.356 s | 0.347 s | 1.0x |
| `layer_operator_balance` | 0.0005 s | 51.84 s | 103,680x |
| `rank_load_balance` | 2.086 s | 19.30 s | 9.3x |

The total-time speedups are smaller because all methods pay load/decompression cost. The analyze-only speedups are the cleanest evidence for the template-indexed analysis story.

### 4.2 Five-Task PADOC Results

The completed PADOC-only five-task matrix adds `compute_comm_overlap`. Columns below use the current `padoc_5task_analysis.tsv` measurements.

| Dataset | Read + deserialize/decompress | Max task analyze time | Slowest task | Total time range | Peak RSS |
|---|---:|---:|---|---:|---:|
| `leworldmodel_full` | 1.424 s | 0.0317 s | `compute_comm_overlap` | 1.424-1.455 s | 0.55 GiB |
| `qwen3_full` | 11.303 s | 0.4403 s | `compute_comm_overlap` | 11.303-11.743 s | 5.03 GiB |
| `unifolm_full` | 36.231 s | 1.6640 s | `compute_comm_overlap` | 36.232-37.895 s | 14.14 GiB |
| `llama_full` | 116.656 s | 8.2837 s | `compute_comm_overlap` | 116.656-124.940 s | 34.11 GiB |

Use this wording:

After the artifact is loaded, most in-situ analyses run in milliseconds to a few seconds even on the 301M-event llama trace. The new overlap task is the slowest because it collects and merges per-rank compute/communication intervals, but even on `llama_full` the analysis phase is 8.28 s; the total is dominated by artifact loading/deserialization.

Important caveat:

The baseline speedup table above is for the original four tasks. `compute_comm_overlap` has PADOC, ablation, and profiling results, but the baseline matrix was not rerun for this fifth task. Do not claim a cross-compressor speedup for `compute_comm_overlap` unless that extra baseline run is added.

## 5. On-Disk Storage Breakdown

Source: `on_disk_breakdown.txt`, generated by `inspect_artifact --on-disk`.

| Dataset | Artifact | Dominant on-disk regions |
|---|---:|---|
| `leworldmodel_full` | 37.52 MiB | args columns 20.50 MB, node soft links 13.57 MB, rank node tree 12.95 MB, timestamp columns 4.71 MB |
| `qwen3_full` | 272.23 MiB | timestamp columns 125.50 MB, node soft links 100.64 MB, rank node tree 100.15 MB, args columns 44.78 MB |
| `unifolm_full` | 741.08 MiB | args columns 352.40 MB, node soft links 264.15 MB, rank node tree 258.25 MB, timestamp columns 144.34 MB |
| `llama_full` | 2.40 GiB | timestamp columns 1.07 GB, rank node tree 946.11 MB, node soft links 925.15 MB, args columns 295.25 MB |

For `llama_full`, the main on-disk contributors are:

| Region | zstd bytes | Share of artifact |
|---|---:|---:|
| `ts_columns` | 1,073,867,885 | 41.7% |
| `rank_node_tree` | 946,110,875 | 36.8% |
| `node_soft_links` | 925,145,515 | 35.9% |
| `args_columns` | 295,247,430 | 11.5% |
| `ids_pids_phases_streams` | 153,426,779 | 6.0% |
| `dur_columns` | 101,498,433 | 3.9% |
| `name_nums` | 2,715,745 | 0.1% |

The shares do not sum to 100% because each region is encoded independently for attribution; it is a contribution profile, not a partition of the exact final zstd stream.

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
- `ablation_analyze.tsv`: 3 datasets x 8 presets x 5 tasks = 120 rows.

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

Source: `analysis_profile_padoc.tsv`.

Representative `llama_full` profile:

| Task | Dominant phase | Phase time |
|---|---|---:|
| `operator_hotspot` | template tally | 0.068 s |
| `stream_load_balance` | template stream aggregate | 0.349 s |
| `compute_comm_overlap` | rank interval collect + merge/json | 4.047 s + 1.655 s |
| `layer_operator_balance` | template layer aggregate | 0.000245 s |
| `rank_load_balance` | rank tree walk | 1.850 s |

Interpretation:

The profile supports the mechanism section: most tasks are template- or tree-walk bounded rather than event-scan bounded. `compute_comm_overlap` is intentionally heavier because it constructs and merges intervals.

## 10. Data Quality / Sanity Checks

These checks were performed before this summary was written:

| Check | Result |
|---|---|
| `padoc_5task_analysis.tsv` row count | 20 rows = 4 datasets x 5 tasks |
| `ablation_analyze.tsv` row count | 120 rows = 3 datasets x 8 presets x 5 tasks |
| `ablation_storage_from_artifacts.tsv` row count | 24 artifact rows + header |
| `analysis_profile_padoc.jsonl` row count | 20 rows = 4 datasets x 5 tasks |
| GPU scalability sections | 4 sections = 1/8/64/256 GPUs |
| Thread scalability sections | 14 sections = 7 worker points for small datasets + 7 for llama |
| Default PADOC artifact bytes in storage vs analysis ablation | exact match for leworldmodel/qwen3/unifolm |
| `cargo test --quiet` | passed: 15 library tests + 11 end-to-end tests |
| Worktree after experiment commits | clean before creating this summary |

The data is internally consistent and has the expected qualitative behavior:

- Compression ratio is stable or improves with larger repeated traces.
- Thread scaling improves until saturation, then regresses at excessive parallelism.
- Analysis time is dominated by load/deserialization, not by the in-situ task logic.
- Minimal/smaller artifacts can cost more memory/time, which is consistent with the co-design tradeoff.
- The largest dataset (`llama_full`) remains analyzable in one merged PADOC artifact with about 34 GiB peak RSS.

## 11. Caveats To Keep In The Paper Honest

1. Do not claim PADOC is always the smallest compressor. ScalaTrace often has a better byte ratio, but it does not preserve the same analysis-ready structure.
2. Do not claim cross-compressor speedup for `compute_comm_overlap` unless the baseline matrix is rerun with that fifth task. Current cross-compressor speedups cover the original four tasks.
3. Do not claim every ablation feature improves on-disk size. Some smaller variants are worse for analysis time and memory; frame ablation as the storage-analysis tradeoff.
4. Full llama-scale 8-preset ablation was not run because of cost and memory risk. The completed ablation is full for leworldmodel/qwen3/unifolm.
5. MoE and ViT rows are absent because the corresponding traces are not available locally.

## 12. Result File Index

Primary paper tables:

- `EXPERIMENTS.md`: original compression, memory, baseline-analysis, speedup, and lossless verification tables.
- `results/remaining/padoc_5task_analysis.tsv`: PADOC five-task analysis, including `compute_comm_overlap`.
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
