#!/bin/bash
# PADOC parallel-compression worker sweep.  Writes one markdown table per
# dataset/worker cell into results/remaining/compress_scalability.md.
#
# Usage:
#   scripts/scalability_compress.sh [artifact_out_dir] [result_md] [workers_csv]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts_scalability}
OUT=${2:-"$ROOT/results/remaining/compress_scalability.md"}
WORKERS_CSV=${3:-1,2,4,8,16,32,64}
PADOC="$ROOT/target/release/padoc"

mkdir -p "$ART" "$(dirname "$OUT")"
> "$OUT"

run_one() {
  local ds=$1
  local manifest=$2
  local workers=$3
  local out_dir="$ART/${ds}_w${workers}"
  mkdir -p "$out_dir"
  echo ">>> $ds workers=$workers" >&2
  {
    echo
    echo "## $ds workers=$workers"
  } >> "$OUT"
  if command -v numactl >/dev/null 2>&1; then
    numactl --interleave=all "$PADOC" bench compress \
      --manifest "$manifest" \
      --padoc-presets default \
      --workers "$workers" \
      --out-dir "$out_dir" >> "$OUT"
  else
    "$PADOC" bench compress \
      --manifest "$manifest" \
      --padoc-presets default \
      --workers "$workers" \
      --out-dir "$out_dir" >> "$OUT"
  fi
}

IFS=',' read -r -a workers_list <<< "$WORKERS_CSV"
for workers in "${workers_list[@]}"; do
  run_one small "$ROOT/scripts/manifest_small.json" "$workers"
done

for workers in "${workers_list[@]}"; do
  run_one llama "$ROOT/scripts/manifest_llama.json" "$workers"
done

echo "wrote $OUT" >&2
