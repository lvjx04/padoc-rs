#!/bin/bash
# PADOC ablation experiment.  Produces artifacts for every compressor preset,
# then runs the four in-situ analysis tasks on each artifact.
#
# Usage:
#   scripts/run_ablation.sh [artifact_out_dir] [results_dir] [manifest]
#
# Defaults target the small manifest to keep the smoke run manageable.  For
# llama-scale ablation pass scripts/manifest_llama.json and a suitable artifact
# directory on the cluster.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts_ablation}
RES=${2:-"$ROOT/results/remaining"}
MANIFEST=${3:-"$ROOT/scripts/manifest_small.json"}
WORKERS=${WORKERS:-32}
TASKS=${TASKS:-operator_hotspot,stream_load_balance,layer_operator_balance,rank_load_balance}
PADOC="$ROOT/target/release/padoc"

mkdir -p "$ART" "$RES"

COMPRESS_OUT="$RES/ablation_compress.md"
ANALYZE_OUT="$RES/ablation_analyze.tsv"
> "$COMPRESS_OUT"
> "$ANALYZE_OUT"

echo ">>> ablation compress manifest=$MANIFEST workers=$WORKERS" >&2
if command -v numactl >/dev/null 2>&1; then
  numactl --interleave=all "$PADOC" bench compress \
    --manifest "$MANIFEST" \
    --padoc-presets all \
    --workers "$WORKERS" \
    --out-dir "$ART" > "$COMPRESS_OUT"
else
  "$PADOC" bench compress \
    --manifest "$MANIFEST" \
    --padoc-presets all \
    --workers "$WORKERS" \
    --out-dir "$ART" > "$COMPRESS_OUT"
fi

for file in "$ART"/*.padoc*.zst; do
  [ -f "$file" ] || continue
  base=$(basename "$file")
  dataset=${base%%.padoc*}
  compressor=${base#"$dataset".}
  compressor=${compressor%.zst}
  echo ">>> analyze $dataset $compressor" >&2
  if command -v numactl >/dev/null 2>&1; then
    numactl --interleave=all "$PADOC" bench analyze-batch \
      --compressor padoc \
      --artifact "$file" \
      --tasks "$TASKS" \
      --repeat 1 \
      | awk -v ds="$dataset" -v cfg="$compressor" 'BEGIN{OFS="\t"} {print ds, cfg, $0}' >> "$ANALYZE_OUT"
  else
    "$PADOC" bench analyze-batch \
      --compressor padoc \
      --artifact "$file" \
      --tasks "$TASKS" \
      --repeat 1 \
      | awk -v ds="$dataset" -v cfg="$compressor" 'BEGIN{OFS="\t"} {print ds, cfg, $0}' >> "$ANALYZE_OUT"
  fi
done

echo "wrote $COMPRESS_OUT and $ANALYZE_OUT" >&2
