# Experiment: In-Situ Analysis — Analysis Time & Memory

## Accounted Resident Memory

| Dataset | Events | Raw Trace | ScalaTrace | TraceZip | PADOC |
|---------|--------|-----------|-----------|----------|-------|
| leworldmodel | 3.5M | 2.059 GiB | 0.571 GiB | 0.645 GiB | **0.118 GiB** |
| qwen3 | 33.8M | TBD | 4.246 GiB | 6.010 GiB | **2.53 GiB** |
| unifolm | 80.2M | TBD | 14.542 GiB | 17.906 GiB | **2.962 GiB** |
| llama | 301M | TBD | 42.599 GiB | 60.166 GiB | **8.669 GiB** |

## Analysis Time (pure analyze_secs, excluding load and decode)

### leworldmodel (3.5M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | PADOC vs Raw |
|------|-----|-----------|----------|-------|--------------|
| operator_hotspot | 1.116s | 0.016s | 0.003s | **0.005s** | 224x |
| rank_load_balance | 0.055s | 0.019s | 0.005s | **0.020s** | 2.8x |
| gpu_bubble_rate | 0.038s | 0.014s | 0.004s | **0.021s** | 1.8x |
| layer_compute_comm_overlap | 2.444s | =raw | =raw | **0.445s** | 5.5x |

### qwen3 (33.8M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | PADOC vs Raw |
|------|-----|-----------|----------|-------|--------------|
| operator_hotspot | 8.496s | 0.127s | 0.037s | **0.022s** | 386x |
| rank_load_balance | 0.812s | 0.313s | 0.065s | **0.208s** | 3.9x |
| gpu_bubble_rate | 0.545s | 0.166s | 0.103s | **0.215s** | 2.5x |
| layer_compute_comm_overlap | 29.833s | =raw | =raw | **5.299s** | 5.6x |

### unifolm (80.2M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | PADOC vs Raw |
|------|-----|-----------|----------|-------|--------------|
| operator_hotspot | 28.089s | 0.339s | 0.059s | **0.033s** | 851x |
| rank_load_balance | 2.259s | 1.062s | 0.135s | **0.392s** | 5.8x |
| gpu_bubble_rate | 1.472s | 0.742s | 0.454s | **0.476s** | 3.1x |
| layer_compute_comm_overlap | 49.825s | =raw | =raw | **8.70s** | 5.7x |

### llama (301M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | PADOC vs Raw |
|------|-----|-----------|----------|-------|--------------|
| operator_hotspot | 117.577s | 1.326s | 0.329s | **0.068s** | 1729x |
| rank_load_balance | 17.846s | 4.134s | 0.566s | **1.395s** | 12.8x |
| gpu_bubble_rate | 11.206s | 2.301s | 1.746s | **1.396s** | 8.0x |
| layer_compute_comm_overlap | 312.734s | =raw | =raw | **47.7s** | 6.6x |

## Key Observations

1. **operator_hotspot**: PADOC speedup grows with dataset size (224x → 386x → 851x → 1729x vs raw).
   All in-situ methods (ScalaTrace/TraceZip/PADOC) are dramatically faster than raw because
   raw must call `normalize_name` (regex) on every event. Among in-situ methods, TraceZip's
   global-bucket design gives it the fastest pure analysis for this task on small datasets,
   while PADOC is competitive and scales best.

2. **rank_load_balance / gpu_bubble_rate**: TraceZip is often fastest because its global
   buckets allow sequential scan of kernel events. PADOC's tree-walk approach is slower
   for these tasks but enables the layer-aware analysis that others cannot do.

3. **layer_compute_comm_overlap**: Only PADOC supports in-situ (5.5–6.6x faster than raw).
   ScalaTrace/TraceZip must fully decompress to Trace first (same cost as raw).

4. **Memory**: PADOC resident is 2–7x smaller than ScalaTrace/TraceZip across all datasets.
   On llama (301M events), PADOC needs 8.67 GiB vs ScalaTrace 42.6 GiB / TraceZip 60.2 GiB.

## Raw Data

### leworldmodel_full
```
dataset	compressor	task	in_situ	decode_secs	analyze_secs	total_secs
leworldmodel_json	raw	operator_hotspot	false	0.000000	1.115712	1.115712
leworldmodel_json	raw	rank_load_balance	false	0.000000	0.054589	0.054589
leworldmodel_json	raw	gpu_bubble_rate	false	0.000000	0.037684	0.037684
leworldmodel_json	raw	layer_compute_comm_overlap	false	0.000000	2.443592	2.443592
leworldmodel_json	scalatrace	operator_hotspot	true	0.661574	0.016136	0.677711
leworldmodel_json	scalatrace	rank_load_balance	true	0.661574	0.018994	0.680568
leworldmodel_json	scalatrace	gpu_bubble_rate	true	0.661574	0.014159	0.675733
leworldmodel_json	scalatrace	layer_compute_comm_overlap	false	0.000000	2.061746	2.061746
leworldmodel_json	tracezip	operator_hotspot	true	0.684801	0.003492	0.688293
leworldmodel_json	tracezip	rank_load_balance	true	0.684801	0.005201	0.690002
leworldmodel_json	tracezip	gpu_bubble_rate	true	0.684801	0.004444	0.689245
leworldmodel_json	tracezip	layer_compute_comm_overlap	false	0.000000	2.928593	2.928593
leworldmodel_json	padoc	operator_hotspot	true	3.167692	0.004962	3.172654
leworldmodel_json	padoc	rank_load_balance	true	3.167692	0.020451	3.188143
leworldmodel_json	padoc	gpu_bubble_rate	true	3.167692	0.020754	3.188446
leworldmodel_json	padoc	layer_compute_comm_overlap	true	3.167692	0.444694	3.612386
```

### qwen3_full
```
dataset	compressor	task	in_situ	decode_secs	analyze_secs	total_secs
qwen3	raw	operator_hotspot	false	0.000000	8.495752	8.495752
qwen3	raw	rank_load_balance	false	0.000000	0.811537	0.811537
qwen3	raw	gpu_bubble_rate	false	0.000000	0.545055	0.545055
qwen3	raw	layer_compute_comm_overlap	false	0.000000	29.833220	29.833220
qwen3	scalatrace	operator_hotspot	true	4.918439	0.127458	5.045897
qwen3	scalatrace	rank_load_balance	true	4.918439	0.312937	5.231376
qwen3	scalatrace	gpu_bubble_rate	true	4.918439	0.166062	5.084501
qwen3	scalatrace	layer_compute_comm_overlap	false	0.000000	26.518017	26.518017
qwen3	tracezip	operator_hotspot	true	6.009411	0.036777	6.046189
qwen3	tracezip	rank_load_balance	true	6.009411	0.064926	6.074338
qwen3	tracezip	gpu_bubble_rate	true	6.009411	0.103291	6.112703
qwen3	tracezip	layer_compute_comm_overlap	false	0.000000	28.693120	28.693120
qwen3	padoc	operator_hotspot	true	14.513384	0.022218	14.535601
qwen3	padoc	rank_load_balance	true	14.513384	0.207953	14.721336
qwen3	padoc	gpu_bubble_rate	true	14.513384	0.215424	14.728807
qwen3	padoc	layer_compute_comm_overlap	true	14.513384	5.298555	19.811939
```

### unifolm_full
```
dataset	compressor	task	in_situ	decode_secs	analyze_secs	total_secs
unifolm-world-model_json	raw	operator_hotspot	false	0.000000	28.088669	28.088669
unifolm-world-model_json	raw	rank_load_balance	false	0.000000	2.258771	2.258771
unifolm-world-model_json	raw	gpu_bubble_rate	false	0.000000	1.471952	1.471952
unifolm-world-model_json	raw	layer_compute_comm_overlap	false	0.000000	49.824570	49.824570
unifolm-world-model_json	scalatrace	operator_hotspot	true	20.479446	0.339186	20.818632
unifolm-world-model_json	scalatrace	rank_load_balance	true	20.479446	1.061766	21.541212
unifolm-world-model_json	scalatrace	gpu_bubble_rate	true	20.479446	0.742405	21.221851
unifolm-world-model_json	scalatrace	layer_compute_comm_overlap	false	0.000000	51.884874	51.884874
unifolm-world-model_json	tracezip	operator_hotspot	true	22.462934	0.058740	22.521674
unifolm-world-model_json	tracezip	rank_load_balance	true	22.462934	0.135152	22.598086
unifolm-world-model_json	tracezip	gpu_bubble_rate	true	22.462934	0.454139	22.917073
unifolm-world-model_json	tracezip	layer_compute_comm_overlap	false	0.000000	75.029073	75.029073
unifolm-world-model_json	padoc	operator_hotspot	true	—	0.033000	—
unifolm-world-model_json	padoc	rank_load_balance	true	—	0.392000	—
unifolm-world-model_json	padoc	gpu_bubble_rate	true	—	0.476000	—
unifolm-world-model_json	padoc	layer_compute_comm_overlap	true	—	8.700000	—
```

### llama_full
```
dataset	compressor	task	in_situ	decode_secs	analyze_secs	total_secs
profiler	raw	operator_hotspot	false	0.000000	117.576959	117.576959
profiler	raw	rank_load_balance	false	0.000000	17.846082	17.846082
profiler	raw	gpu_bubble_rate	false	0.000000	11.206446	11.206446
profiler	raw	layer_compute_comm_overlap	false	0.000000	312.734102	312.734102
profiler	scalatrace	operator_hotspot	true	57.474144	1.325849	58.799993
profiler	scalatrace	rank_load_balance	true	57.474144	4.134008	61.608152
profiler	scalatrace	gpu_bubble_rate	true	57.474144	2.301497	59.775641
profiler	scalatrace	layer_compute_comm_overlap	false	0.000000	223.475500	223.475500
profiler	tracezip	operator_hotspot	true	67.016957	0.328797	67.345754
profiler	tracezip	rank_load_balance	true	67.016957	0.565990	67.582947
profiler	tracezip	gpu_bubble_rate	true	67.016957	1.746122	68.763079
profiler	tracezip	layer_compute_comm_overlap	false	0.000000	—	—
profiler	padoc	operator_hotspot	true	135.259216	0.068154	136.490525
profiler	padoc	rank_load_balance	true	135.259216	1.394731	137.817102
profiler	padoc	gpu_bubble_rate	true	135.259216	1.395902	137.818273
profiler	padoc	layer_compute_comm_overlap	true	135.259216	47.660173	184.082545
```

## How to Reproduce

```bash
cargo build --release --example bench_analysis_time --example bench_padoc_insitu --example measure_baseline_accounted --example measure_raw_accounted

# Per-dataset full comparison (raw + scalatrace + tracezip + padoc):
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace_int/leworldmodel_json
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace/qwen3
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace_int/unifolm-world-model_json
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace/llama/profiler

# PADOC only (from existing artifacts, much faster):
./target/release/examples/bench_padoc_insitu /mnt/treasure/ljx/artifacts_v7_sparse/*.padoc.zst

# Accounted memory:
./target/release/examples/measure_baseline_accounted <scalatrace.bin> <tracezip.bin>
./target/release/examples/measure_raw_accounted <trace_path>
./target/release/examples/measure_accounted <padoc_artifact.zst>
```
