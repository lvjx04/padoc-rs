#!/usr/bin/env bash
# Run raw_baseline_stats on all 4 datasets, then combine with existing
# on_disk_breakdown.txt to produce a per-region compression ratio table.
#
# Usage: bash scripts/raw_baseline.sh [output_file]
#
# Requires: cargo build --release --example raw_baseline_stats

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/target/release/examples/raw_baseline_stats"
OUTPUT="${1:-$PROJECT_DIR/results/remaining/raw_baseline.tsv}"

if [[ ! -x "$BINARY" ]]; then
    echo "Building raw_baseline_stats..."
    cargo build --release --example raw_baseline_stats --manifest-path "$PROJECT_DIR/Cargo.toml"
fi

mkdir -p "$(dirname "$OUTPUT")"

# Dataset definitions (name → path)
declare -A DATASETS=(
    [leworldmodel_full]="/mnt/treasure/ljx/Trace_int/leworldmodel_json"
    [qwen3_full]="/mnt/treasure/ljx/Trace/qwen3"
    [unifolm_full]="/mnt/treasure/ljx/Trace_int/unifolm-world-model_json"
    [llama_full]="/mnt/treasure/ljx/Trace/llama/profiler"
)

# Run raw_baseline_stats for each dataset
TMPFILE=$(mktemp)
FIRST=1
for name in leworldmodel_full qwen3_full unifolm_full llama_full; do
    path="${DATASETS[$name]}"
    if [[ ! -e "$path" ]]; then
        echo "WARN: $path not found, skipping $name" >&2
        continue
    fi
    echo "=== $name ===" >&2
    if [[ $FIRST -eq 1 ]]; then
        "$BINARY" "$path" "$name" > "$TMPFILE"
        FIRST=0
    else
        "$BINARY" "$path" "$name" | tail -n +2 >> "$TMPFILE"
    fi
done

cp "$TMPFILE" "$OUTPUT"
rm -f "$TMPFILE"

echo "Raw baseline written to: $OUTPUT"
echo ""
cat "$OUTPUT"
