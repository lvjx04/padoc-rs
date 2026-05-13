#!/bin/bash
# Create manifest files for llama GPU-count scalability by symlinking the
# first N rank JSON files into subset directories.
#
# Usage:
#   scripts/make_llama_gpu_subsets.sh [source_dir] [subset_root] [values_csv]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${1:-/mnt/treasure/ljx/Trace/llama/profiler}
SUBSETS=${2:-/mnt/treasure/ljx/Trace/llama/subsets}
VALUES_CSV=${3:-1,8,64,256}
MANIFEST_DIR="$ROOT/scripts/generated"

mkdir -p "$SUBSETS" "$MANIFEST_DIR"
IFS=',' read -r -a values <<< "$VALUES_CSV"
mapfile -t files < <(find "$SRC" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.gz' \) | sort)

if [ "${#files[@]}" -eq 0 ]; then
  echo "no trace files found under $SRC" >&2
  exit 1
fi

for n in "${values[@]}"; do
  if [ "$n" -gt "${#files[@]}" ]; then
    echo "requested $n files, only found ${#files[@]}" >&2
    exit 1
  fi
  dir="$SUBSETS/llama_${n}gpus"
  rm -rf "$dir"
  mkdir -p "$dir"
  for file in "${files[@]:0:$n}"; do
    ln -s "$file" "$dir/$(basename "$file")"
  done
  manifest="$MANIFEST_DIR/manifest_llama_${n}gpus.json"
  cat > "$manifest" <<JSON
{
  "datasets": [
    {"name": "llama_${n}gpus", "path": "$dir", "is_directory": true, "gpus": $n}
  ]
}
JSON
  echo "$manifest"
done
