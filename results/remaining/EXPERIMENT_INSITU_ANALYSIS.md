# Experiment: In-Situ Analysis — Analysis Time & Memory

## Accounted Resident Memory

| Dataset | Events | ScalaTrace | TraceZip | PADOC | PADOC vs ScalaTrace | PADOC vs TraceZip |
|---------|--------|-----------|----------|-------|--------------------|--------------------|
| leworldmodel | 3.5M | 0.571 GiB | 0.645 GiB | **0.118 GiB** | 4.8x smaller | 5.5x smaller |
| qwen3 | 33.8M | 4.246 GiB | 6.010 GiB | **2.53 GiB** | 1.7x smaller | 2.4x smaller |
| unifolm | 80.2M | 14.542 GiB | 17.906 GiB | **2.962 GiB** | 4.9x smaller | 6.0x smaller |
| llama | 301M | 42.599 GiB | 60.166 GiB | **8.669 GiB** | 4.9x smaller | 6.9x smaller |

## Analysis Time

### leworldmodel (3.5M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | Speedup vs Raw |
|------|-----|-----------|----------|-------|----------------|
| operator_hotspot | 0.893s | 0.790s | 0.821s | **0.005s** | **185x** |
| rank_load_balance | 0.031s | 0.738s | 0.826s | **0.025s** | **1.2x** |
| gpu_bubble_rate | 0.027s | 0.711s | 0.827s | **0.026s** | **1.0x** |
| layer_compute_comm_overlap | 2.044s | 2.040s | 2.642s | **0.552s** | **3.7x** |

### qwen3 (33.8M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | Speedup vs Raw |
|------|-----|-----------|----------|-------|----------------|
| operator_hotspot | 7.887s | 6.093s | 7.485s | **0.020s** | **395x** |
| rank_load_balance | 0.790s | 6.224s | 7.570s | **0.207s** | **3.8x** |
| gpu_bubble_rate | 0.537s | 6.134s | 7.565s | **0.214s** | **2.5x** |
| layer_compute_comm_overlap | 29.333s | 26.105s | 28.527s | **5.294s** | **5.5x** |

### unifolm (80.2M events)

| Task | Raw | ScalaTrace | TraceZip | PADOC | Speedup vs Raw |
|------|-----|-----------|----------|-------|----------------|
| operator_hotspot | 29.790s | 24.668s | 26.896s | **0.059s** | **505x** |
| rank_load_balance | 2.222s | 24.230s | 26.904s | **0.431s** | **5.2x** |
| gpu_bubble_rate | 1.469s | 22.939s | 28.436s | **0.911s** | **1.6x** |
| layer_compute_comm_overlap | 48.284s | 49.611s | 61.502s | **9.136s** | **5.3x** |

### llama (301M events) — PADOC only (baselines cannot fit in reasonable time)

| Task | PADOC |
|------|-------|
| operator_hotspot | **0.068s** |
| rank_load_balance | **1.395s** |
| gpu_bubble_rate | **1.396s** |
| layer_compute_comm_overlap | **47.7s** |

## Key Observations

1. **operator_hotspot**: PADOC is 185–505x faster than raw baseline, scaling with templates
   rather than events. ScalaTrace/TraceZip in-situ provides marginal improvement over raw.

2. **ScalaTrace/TraceZip in-situ paradox**: Their in-situ is often SLOWER than raw for
   rank_load_balance and gpu_bubble_rate because the decode (zstd+msgpack) overhead is
   included, while raw already has the Trace in memory.

3. **layer_compute_comm_overlap**: Only PADOC supports in-situ (requires call-tree).
   PADOC achieves 3.7–5.5x speedup over raw.

4. **Memory**: PADOC resident is 2–7x smaller than baselines across all datasets.
   On llama (301M events), PADOC needs 8.67 GiB while ScalaTrace would need 42.6 GiB
   and TraceZip 60.2 GiB.

## Raw Data

### leworldmodel_full
```
dataset	compressor	task	in_situ	analyze_secs
leworldmodel_json	raw	operator_hotspot	false	0.892724
leworldmodel_json	raw	rank_load_balance	false	0.030532
leworldmodel_json	raw	gpu_bubble_rate	false	0.026679
leworldmodel_json	raw	layer_compute_comm_overlap	false	2.044084
leworldmodel_json	scalatrace	operator_hotspot	true	0.789982
leworldmodel_json	scalatrace	rank_load_balance	true	0.737858
leworldmodel_json	scalatrace	gpu_bubble_rate	true	0.710784
leworldmodel_json	scalatrace	layer_compute_comm_overlap	false	2.040238
leworldmodel_json	tracezip	operator_hotspot	true	0.820937
leworldmodel_json	tracezip	rank_load_balance	true	0.826001
leworldmodel_json	tracezip	gpu_bubble_rate	true	0.827306
leworldmodel_json	tracezip	layer_compute_comm_overlap	false	2.641652
leworldmodel_json	padoc	operator_hotspot	true	0.004828
leworldmodel_json	padoc	rank_load_balance	true	0.024613
leworldmodel_json	padoc	gpu_bubble_rate	true	0.025515
leworldmodel_json	padoc	layer_compute_comm_overlap	true	0.551533
```

### qwen3_full
```
dataset	compressor	task	in_situ	analyze_secs
qwen3	raw	operator_hotspot	false	7.886655
qwen3	raw	rank_load_balance	false	0.790256
qwen3	raw	gpu_bubble_rate	false	0.536553
qwen3	raw	layer_compute_comm_overlap	false	29.332662
qwen3	scalatrace	operator_hotspot	true	6.093268
qwen3	scalatrace	rank_load_balance	true	6.224175
qwen3	scalatrace	gpu_bubble_rate	true	6.133845
qwen3	scalatrace	layer_compute_comm_overlap	false	26.105128
qwen3	tracezip	operator_hotspot	true	7.484592
qwen3	tracezip	rank_load_balance	true	7.570464
qwen3	tracezip	gpu_bubble_rate	true	7.564770
qwen3	tracezip	layer_compute_comm_overlap	false	28.527357
qwen3	padoc	operator_hotspot	true	0.019809
qwen3	padoc	rank_load_balance	true	0.207262
qwen3	padoc	gpu_bubble_rate	true	0.213785
qwen3	padoc	layer_compute_comm_overlap	true	5.294426
```

### unifolm_full
```
dataset	compressor	task	in_situ	analyze_secs
unifolm-world-model_json	raw	operator_hotspot	false	29.789908
unifolm-world-model_json	raw	rank_load_balance	false	2.222227
unifolm-world-model_json	raw	gpu_bubble_rate	false	1.469050
unifolm-world-model_json	raw	layer_compute_comm_overlap	false	48.284226
unifolm-world-model_json	scalatrace	operator_hotspot	true	24.667764
unifolm-world-model_json	scalatrace	rank_load_balance	true	24.229535
unifolm-world-model_json	scalatrace	gpu_bubble_rate	true	22.938743
unifolm-world-model_json	scalatrace	layer_compute_comm_overlap	false	49.611178
unifolm-world-model_json	tracezip	operator_hotspot	true	26.895844
unifolm-world-model_json	tracezip	rank_load_balance	true	26.904128
unifolm-world-model_json	tracezip	gpu_bubble_rate	true	28.436120
unifolm-world-model_json	tracezip	layer_compute_comm_overlap	false	61.501582
unifolm-world-model_json	padoc	operator_hotspot	true	0.058703
unifolm-world-model_json	padoc	rank_load_balance	true	0.431218
unifolm-world-model_json	padoc	gpu_bubble_rate	true	0.911367
unifolm-world-model_json	padoc	layer_compute_comm_overlap	true	9.136153
```

### llama_full (PADOC only)
```
dataset	artifact_bytes	load_secs	decode_secs	resident_kib	task	analyze_secs	total_secs
llama_full	2614795618	1.163155	135.259216	22038400	operator_hotspot	0.068154	136.490525
llama_full	2614795618	1.163155	135.259216	22038400	rank_load_balance	1.394731	137.817102
llama_full	2614795618	1.163155	135.259216	22038400	gpu_bubble_rate	1.395902	137.818273
llama_full	2614795618	1.163155	135.259216	22038400	layer_compute_comm_overlap	47.660173	184.082545
```

## How to Reproduce

```bash
cargo build --release --example bench_analysis_time --example bench_padoc_insitu

# Per-dataset full comparison (raw + scalatrace + tracezip + padoc):
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace_int/leworldmodel_json
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace/qwen3
./target/release/examples/bench_analysis_time /mnt/treasure/ljx/Trace_int/unifolm-world-model_json

# PADOC only (from existing artifacts):
./target/release/examples/bench_padoc_insitu /mnt/treasure/ljx/artifacts_v7_sparse/llama_full.padoc.zst
```
