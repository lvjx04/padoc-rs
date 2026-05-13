#!/bin/bash
# GPU-count scalability on llama subsets.  This first creates symlinked
# subset directories and manifests, then compresses each subset with PADOC.
#
# Usage:
#   scripts/scalability_gpus.sh [artifact_out_dir] [result_md] [values_csv]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts_gpu_scalability}
OUT=${2:-"$ROOT/results/remaining/gpu_scalability.md"}
VALUES_CSV=${3:-1,8,64,256}
WORKERS=${WORKERS:-32}
PADOC="$ROOT/target/release/padoc"

mkdir -p "$ART" "$(dirname "$OUT")"
> "$OUT"

mapfile -t manifests < <("$ROOT/scripts/make_llama_gpu_subsets.sh" /mnt/treasure/ljx/Trace/llama/profiler /mnt/treasure/ljx/Trace/llama/subsets "$VALUES_CSV")

for manifest in "${manifests[@]}"; do
  name=$(basename "$manifest" .json)
  out_dir="$ART/$name"
  mkdir -p "$out_dir"
  echo ">>> $name workers=$WORKERS" >&2
  {
    echo
    echo "## $name workers=$WORKERS"
  } >> "$OUT"
  if command -v numactl >/dev/null 2>&1; then
    numactl --interleave=all "$PADOC" bench compress \
      --manifest "$manifest" \
      --padoc-presets default \
      --workers "$WORKERS" \
      --out-dir "$out_dir" >> "$OUT"
  else
    "$PADOC" bench compress \
      --manifest "$manifest" \
      --padoc-presets default \
      --workers "$WORKERS" \
      --out-dir "$out_dir" >> "$OUT"
  fi
done

echo "wrote $OUT" >&2
