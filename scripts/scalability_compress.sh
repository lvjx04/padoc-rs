#!/bin/bash
# PADOC parallel-compression worker sweep.  Writes one markdown table per
# dataset/worker cell into results/remaining/compress_scalability.md.
# By default this does not persist artifacts, so the sweep measures the full
# encode path without filling local temporary storage.  Pass an artifact output
# directory as the first argument only when you intentionally want saved .zst
# files for every worker count.
#
# Usage:
#   scripts/scalability_compress.sh [artifact_out_dir|-] [result_md] [workers_csv]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:--}
OUT=${2:-"$ROOT/results/remaining/compress_scalability.md"}
WORKERS_CSV=${3:-1,2,4,8,16,32,64}
PADOC="$ROOT/target/release/padoc"

mkdir -p "$(dirname "$OUT")"
if [ "$ART" != "-" ]; then
  mkdir -p "$ART"
fi
> "$OUT"

run_one() {
  local ds=$1
  local manifest=$2
  local workers=$3
  local out_args=()
  if [ "$ART" != "-" ]; then
    local out_dir="$ART/${ds}_w${workers}"
    mkdir -p "$out_dir"
    out_args=(--out-dir "$out_dir")
  fi
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
      "${out_args[@]}" >> "$OUT"
  else
    "$PADOC" bench compress \
      --manifest "$manifest" \
      --padoc-presets default \
      --workers "$workers" \
      "${out_args[@]}" >> "$OUT"
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
