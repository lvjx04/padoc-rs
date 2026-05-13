#!/bin/bash
# DO NOT use set -e — we want to continue past any one baseline that fails
# losslessness so we know exactly which ones are lossy.
cd ~/Work/padoc-rs
LOG=~/Work/padoc-rs/queue_sc4_baselines.log
exec >> "$LOG" 2>&1
echo "=== queue_sc4_baselines starting at $(date) ==="

ART=/mnt/treasure/ljx/padoc_artifacts/v4-baselines
mkdir -p "$ART"

# ---------- Phase 1: lewm + qwen3 ----------
# Both fit comfortably with all 4 baselines. Trace::from_dir loads everything,
# then each compressor walks the trace once.
echo "=== PHASE1 lewm+qwen3, all 4 baselines, at $(date) ==="
free -h | head -3
numactl --interleave=all ./target/release/padoc bench compress \
  --datasets /mnt/treasure/ljx/Trace_int/leworldmodel_json,/mnt/treasure/ljx/Trace/qwen3 \
  --compressors raw_json,gzip_json,tracezip,scalatrace \
  --out-dir "$ART" \
  2>&1 | tee /tmp/phase1.log | tail -15

# Roundtrip on lewm dir per baseline (3.5M events, peak ~10 GiB).
# raw_json decompress is a stub so we cant verify it; skip.
echo "=== PHASE1 lewm roundtrip per baseline ==="
for c in gzip_json tracezip scalatrace ; do
  echo "--- lewm $c at $(date) ---"
  numactl --interleave=all ./target/release/padoc roundtrip \
    /mnt/treasure/ljx/Trace_int/leworldmodel_json \
    --compressor $c --dir 2>&1 | tail -20
  echo
done

# ---------- Phase 2: unifolm without raw_json ----------
# raw_json on unifolm: serde_json Value tree alone would be ~100 GiB on top
# of the 80 GiB Trace; skip.
echo "=== PHASE2 unifolm, gzip+tracezip+scalatrace, at $(date) ==="
free -h | head -3
numactl --interleave=all ./target/release/padoc bench compress \
  --datasets /mnt/treasure/ljx/Trace_int/unifolm-world-model_json \
  --compressors gzip_json,tracezip,scalatrace \
  --out-dir "$ART" \
  2>&1 | tee /tmp/phase2.log | tail -15

# Roundtrip on unifolm rank0 only (peak ~50 GiB).
echo "=== PHASE2 unifolm rank0 roundtrip per baseline ==="
for c in gzip_json tracezip scalatrace ; do
  echo "--- unifolm rank0 $c at $(date) ---"
  numactl --interleave=all ./target/release/padoc roundtrip \
    /mnt/treasure/ljx/Trace_int/unifolm-world-model_json/global_rank0.json \
    --compressor $c 2>&1 | tail -20
  echo
done

# ---------- Phase 3: llama via per-rank streaming ----------
# 'bench compress --per-rank' runs each rank file through the compressor
# independently and aggregates the result. Memory bounded by 1 rank.
# Note: cross-rank dedup is OFF in this mode, so the resulting "ratio"
# is a per-rank lower bound, not the cross-rank-dedup result. Useful
# for the paper as long as we are explicit about it.
echo "=== PHASE3 llama per-rank streaming, all 4 baselines, at $(date) ==="
free -h | head -3
mkdir -p "$ART/llama_per_rank"
numactl --interleave=all ./target/release/padoc bench compress \
  --datasets /mnt/treasure/ljx/Trace/llama/profiler \
  --compressors raw_json,gzip_json,tracezip,scalatrace \
  --per-rank \
  --out-dir "$ART/llama_per_rank" \
  2>&1 | tee /tmp/phase3.log | tail -25

# Roundtrip on a single llama rank file per baseline.
echo "=== PHASE3 llama rank0 roundtrip per baseline ==="
for c in gzip_json tracezip scalatrace ; do
  echo "--- llama rank0 $c at $(date) ---"
  numactl --interleave=all ./target/release/padoc roundtrip \
    /mnt/treasure/ljx/Trace/llama/profiler/profiler_0.json \
    --compressor $c 2>&1 | tail -20
  echo
done

echo "=== queue_sc4_baselines finished at $(date) ==="
