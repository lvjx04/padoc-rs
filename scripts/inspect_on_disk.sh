#!/bin/bash
# Field-level on-disk breakdown for every PADOC artifact.  Output goes to
# results/remaining/on_disk_breakdown.txt unless a second argument overrides it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts}
OUT=${2:-"$ROOT/results/remaining/on_disk_breakdown.txt"}
INSPECT="$ROOT/target/release/examples/inspect_artifact"

mkdir -p "$(dirname "$OUT")"
> "$OUT"

for ds in leworldmodel_full qwen3_full unifolm_full llama_full; do
  file="$ART/$ds.padoc.zst"
  if [ ! -f "$file" ]; then
    echo "missing $file" >&2
    continue
  fi
  echo "=== $ds ===" >> "$OUT"
  "$INSPECT" --on-disk "$file" >> "$OUT"
done

echo "wrote $OUT" >&2
