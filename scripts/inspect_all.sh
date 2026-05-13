#!/bin/bash
# Run examples/inspect_artifact on every PADOC artifact and produce
# results/main/inspect_small.txt + results/main/inspect_llama.txt.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts}
INSPECT="$ROOT/target/release/examples/inspect_artifact"

SMALL="$ROOT/results/main/inspect_small.txt"
> "$SMALL"
for ds in leworldmodel_full qwen3_full unifolm_full; do
  echo "=== $ds ===" >> "$SMALL"
  "$INSPECT" "$ART/$ds.padoc.zst" 2>&1 | tail -25 >> "$SMALL"
done

"$INSPECT" "$ART/llama_full.padoc.zst" 2>&1 | tee "$ROOT/results/main/inspect_llama.txt"
echo "wrote $SMALL and inspect_llama.txt" >&2
