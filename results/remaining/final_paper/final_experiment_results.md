# Final Paper Experiment Results

This file is the final consolidated experiment sheet used by `docs/thesis.md`.
The PADOC artifact numbers use the sparse-slot v7 artifacts under
`/mnt/treasure/ljx/artifacts_v7_sparse/`. Historical baseline compressor
numbers come from `EXPERIMENTS.md` and `/mnt/treasure/ljx/artifacts_v6/`.

## Dataset Scale

| Dataset | Workload | Ranks | Events | Raw size |
|---|---|---:|---:|---:|
| `leworldmodel_full` | LeWorldModel inference | 2 | 3,469,389 | 884.37 MiB |
| `qwen3_full` | Qwen3 dense training | 256 | 33,813,574 | 6.91 GiB |
| `unifolm_full` | UniFolm world-model training | 4 | 80,223,071 | 22.43 GiB |
| `llama_full` | LLaMA-70B training | 1024 | 301,288,116 | 69.95 GiB |

## Compression Comparison

| Dataset | PADOC size | PADOC ratio | ScalaTrace size / ratio | TraceZip size / ratio | gzip_json size / ratio | raw_json size / ratio |
|---|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.17 MiB | 23.79x | 14.31 MiB / 60.97x | 28.27 MiB / 32.76x | 42.97 MiB / 21.31x | 732.79 MiB / 1.21x |
| `qwen3_full` | 274.41 MiB | 25.79x | 208.97 MiB / 34.30x | 279.17 MiB / 26.59x | 400.09 MiB / 17.71x | 5.43 GiB / 1.27x |
| `unifolm_full` | 739.05 MiB | 31.08x | 278.82 MiB / 82.39x | 483.62 MiB / 47.50x | 829.34 MiB / 27.70x | 18.01 GiB / 1.25x |
| `llama_full` | 2.44 GiB | 28.72x | 2.00 GiB / 34.94x | 2.48 GiB / 28.24x | 3.24 GiB / 21.59x | 53.63 GiB / 1.30x |

PADOC is not always the smallest byte stream. Its claim is competitive
compression while preserving queryable structure for in-situ analysis.

## PADOC Core Analysis

Source: `results/remaining/final_paper/core_layer_analysis_sparse_v7.tsv`.
`Load to memory` is `read_secs + decompress_secs`. `Resident` is the accounted
resident representation after loading, excluding transient buffers.

| Dataset | Artifact | Load to memory | Max analyze | Slowest task | Total time range | Resident |
|---|---:|---:|---:|---|---:|---:|
| `leworldmodel_full` | 37.17 MiB | 3.075 s | 0.567 s | `layer_kernel_hotspot` | 3.096-3.641 s | 0.237 GiB |
| `qwen3_full` | 274.41 MiB | 14.021 s | 9.126 s | `layer_compute_comm_overlap` | 14.147-23.146 s | 1.899 GiB |
| `unifolm_full` | 739.05 MiB | 87.547 s | 12.920 s | `layer_kernel_hotspot` | 88.070-100.468 s | 4.678 GiB |
| `llama_full` | 2.44 GiB | 133.878 s | 92.393 s | `layer_compute_comm_overlap` | 133.992-226.272 s | 14.375 GiB |

Representative per-task timings:

| Dataset | Task | Load to memory | Analyze | Total |
|---|---|---:|---:|---:|
| `qwen3_full` | `operator_hotspot` | 14.021 s | 0.126 s | 14.147 s |
| `qwen3_full` | `rank_load_balance` | 14.021 s | 0.387 s | 14.408 s |
| `qwen3_full` | `layer_kernel_hotspot` | 14.021 s | 3.089 s | 17.110 s |
| `qwen3_full` | `layer_compute_comm_overlap` | 14.021 s | 9.126 s | 23.146 s |
| `qwen3_full` | `layer_rank_balance` | 14.021 s | 5.336 s | 19.357 s |
| `llama_full` | `operator_hotspot` | 133.878 s | 0.114 s | 133.992 s |
| `llama_full` | `rank_load_balance` | 133.878 s | 3.184 s | 137.063 s |
| `llama_full` | `layer_kernel_hotspot` | 133.878 s | 23.389 s | 157.267 s |
| `llama_full` | `layer_compute_comm_overlap` | 133.878 s | 92.393 s | 226.272 s |
| `llama_full` | `layer_rank_balance` | 133.878 s | 42.084 s | 175.962 s |

## Resident Memory Breakdown

Source: `results/remaining/final_paper/on_disk_breakdown_sparse_v7.txt`.
This is accounted resident representation size after loading, excluding
transient load buffers. Process peak RSS in the raw timing logs is retained as
an engineering diagnostic, but it is not used as the paper's main memory metric.

| Dataset | Accounted resident | ts columns | dur columns | id/pid/stream cols | node storage | args storage |
|---|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 0.237 GiB | 0.013 GiB | 0.012 GiB | 0.000 GiB | 0.160 GiB | 0.049 GiB |
| `qwen3_full` | 1.899 GiB | 0.126 GiB | 0.096 GiB | 0.044 GiB | 1.124 GiB | 0.488 GiB |
| `unifolm_full` | 4.678 GiB | 0.299 GiB | 0.264 GiB | 0.308 GiB | 2.054 GiB | 1.697 GiB |
| `llama_full` | 14.375 GiB | 1.122 GiB | 0.770 GiB | 0.843 GiB | 7.679 GiB | 2.535 GiB |

The large file-to-memory gap is mainly due to structured runtime objects:
node storage, args columns, typed numeric columns, vector headers/capacity and
allocator overhead. The on-disk artifact is also zstd-compressed; the in-memory
representation is not zstd-compressed because analyses need direct random or
tree-structured access.

## On-Disk Breakdown

Each region is encoded independently for attribution, so region bytes are a
contribution profile and are not required to sum to the exact artifact size.

| Dataset | Artifact | ts zstd | dur zstd | ids/pids/streams zstd | name nums zstd | args zstd | tree + refs zstd |
|---|---:|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 37.17 MiB | 4.50 MiB | 0.32 MiB | 0.04 MiB | 0.05 MiB | 19.55 MiB | 24.96 MiB |
| `qwen3_full` | 274.41 MiB | 119.69 MiB | 13.81 MiB | 0.03 MiB | 0.32 MiB | 42.71 MiB | 193.61 MiB |
| `unifolm_full` | 739.05 MiB | 137.66 MiB | 6.36 MiB | 3.66 MiB | 0.89 MiB | 336.07 MiB | 497.06 MiB |
| `llama_full` | 2.44 GiB | 1.00 GiB | 96.80 MiB | 146.32 MiB | 2.59 MiB | 281.57 MiB | 1.78 GiB |

For `llama_full`, timestamps and the tree/reference representation dominate
the on-disk contribution. This supports evaluating timestamp residual coding
and more compact node/reference encodings as future engineering directions.

## Ablations

### Structural Information

Source: `results/remaining/final_paper/no_structural_core_ablation.tsv`.
Removing structural merging can make artifacts slightly smaller and some
layer-aware traversals faster on these traces because repeated scopes are less
compressed, but it increases resident memory substantially and weakens the
analysis-ready representation.

| Dataset | Preset | Artifact | Accounted resident | `rank_load_balance` analyze | `layer_compute_comm_overlap` analyze |
|---|---|---:|---:|---:|---:|
| `qwen3_full` | default | 272.23 MiB | 1.899 GiB | 0.212 s | 6.798 s |
| `qwen3_full` | no structural | 268.48 MiB | 3.558 GiB | 0.350 s | 1.516 s |
| `unifolm_full` | default | 741.08 MiB | 4.678 GiB | 0.634 s | 9.638 s |
| `unifolm_full` | no structural | 692.71 MiB | 9.472 GiB | 1.246 s | 5.941 s |

The correct interpretation is not that every structural optimization always
reduces every query time. The structural representation trades some traversal
shape for much lower resident memory and keeps layer/rank semantics explicit.

### Timestamp Int64 vs Compact Int32

All final artifacts have `i64` numeric column count 0. The final in-memory
timestamp columns are compact `i32` or constants. For `llama_full`, timestamp
columns occupy 1.122 GiB as compact representation; storing the same timestamp
values as int64 would require roughly 2.244 GiB before vector overhead, adding
about 1.122 GiB to resident memory. This validates the current timestamp
normalization/downcast.

### Segmented Linear Timestamp Prototype

Source: `results/remaining/slp_probe_results.md`. This is an offline column
probe, not integrated into the main PADOC artifact format.

| Dataset | Columns | Sampled values | Hybrid vs int64 memory | Hybrid vs int32 memory | Accepted cols | Encode time |
|---|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 128 | 3,906,828 | 7.06x | 3.53x | 128 | 0.153 s |
| `qwen3_full` | 128 | 38,097,496 | 4.08x | 2.04x | 118 | 1.620 s |
| `unifolm_full` | 128 | 58,928,428 | 5.87x | 2.93x | 116 | 2.333 s |
| `llama_full` | 128 | 93,249,920 | 4.04x | 2.02x | 126 | 4.087 s |

The prototype uses integer fixed-point segmented linear prediction and stores
int8/int16 residuals with per-column fallback. It satisfies the in-memory
compression target on sampled columns, but it should be presented as a
validated optimization direction rather than a deployed main artifact feature.

### CPU-GPU Mapping

There are two related experiments:

| Experiment | Source | Interpretation |
|---|---|---|
| `no_kernel_links` semantic ablation | `results/remaining/core_kernel_link_coverage.tsv` | If the compressed tree does not retain CPU-GPU provenance, the current in-situ layer-aware tasks cannot attribute GPU kernels to layer scopes directly. |
| Dynamic correlation lookup | `results/remaining/final_paper/dynamic_kernel_mapping_ablation.tsv` | If GPU pointers are not used directly, the analysis can rebuild a correlation map and look up GPU kernels dynamically, but it pays extra lookup cost and is more sensitive to duplicate/non-global correlation ids. |

For `qwen3_full`, default PADOC attributes 1,592,830 / 1,806,096 GPU refs
with 88.19% coverage. Dynamic lookup attributes 1,588,915 refs with 87.98%
coverage. The default `layer_compute_comm_overlap` analyze time is 9.126 s in
the final v7 run; dynamic lookup spends 0.568 s building the map and 7.389 s
in the simplified equivalent analysis path. This shows the mapping can be
reconstructed from correlation columns, but maintaining explicit links gives a
simpler and more robust in-situ path.

## Result Quality

The final data supports the paper's main claims:

- PADOC compresses four real AI profiler traces to 23.79x-31.08x while keeping queryable structure.
- PADOC is competitive but not always smallest; ScalaTrace is often smaller because it does not preserve the same analysis-ready tree and provenance structures.
- The largest 301M-event, 1024-rank trace is analyzed as one merged artifact with 2.44 GiB on disk, 14.38 GiB accounted resident representation, and 133.99-226.27 s end-to-end task time.
- The storage and memory breakdowns are reasonable: timestamps dominate part of disk, while resident memory is dominated by structure and args rather than raw int64 timestamps.
- The ablations support a nuanced co-design claim: structure and links are not free, but they preserve semantics and reduce resident memory compared with less structured variants.
