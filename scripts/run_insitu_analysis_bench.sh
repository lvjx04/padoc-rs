#!/bin/bash
# Run the full in-situ analysis pipeline experiment.
#
# Measures: load_from_disk + decode + analyze for each (compressor, task) pair.
# Compressors: scalatrace, tracezip, padoc
# Tasks: operator_hotspot, rank_load_balance, gpu_bubble_rate
# Datasets: leworldmodel_full, qwen3_full, unifolm_full
#
# Usage:
#   bash scripts/run_insitu_analysis_bench.sh

set -e

PADOC="./target/release/padoc"
OUT_DIR="results/remaining/insitu_analysis"
mkdir -p "$OUT_DIR"

echo "=== In-Situ Analysis Pipeline Benchmark ==="
echo "Compressors: scalatrace, tracezip, padoc"
echo "Tasks: operator_hotspot, rank_load_balance, gpu_bubble_rate"
echo ""

# Small datasets (fit in memory)
echo "Running analysis matrix on small datasets..."
$PADOC bench analyze \
    --manifest scripts/manifest_small.json \
    --compressors scalatrace,tracezip,padoc \
    --tasks operator_hotspot,rank_load_balance,gpu_bubble_rate \
    > "$OUT_DIR/analysis_small.tsv"

echo "Done! Results in $OUT_DIR/analysis_small.tsv"
cat "$OUT_DIR/analysis_small.tsv"
