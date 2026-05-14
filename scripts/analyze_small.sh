#!/bin/bash
# Single-process analyse benchmark for the three small datasets.
# Output (TSV) goes to results/main/analyze_small.tsv.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts}
OUT="$ROOT/results/main/analyze_small.tsv"
PADOC="$ROOT/target/release/padoc"
TASKS=${TASKS:-operator_hotspot,rank_load_balance}
PADOC_CORE_TASKS=${PADOC_CORE_TASKS:-operator_hotspot,rank_load_balance,layer_kernel_hotspot,layer_compute_comm_overlap,layer_rank_balance}
> "$OUT"
for ds in leworldmodel_full qwen3_full unifolm_full; do
  for c in padoc raw_json gzip_json scalatrace tracezip; do
    ext=$( [ "$c" = padoc ] && echo padoc.zst || echo "$c.bin" )
    file="$ART/${ds}.${ext}"
    [ -f "$file" ] || { echo "missing $file" >&2; continue; }
    echo ">>> $ds $c" >&2
    if [ "$c" = padoc ]; then
      tasks="$PADOC_CORE_TASKS"
    else
      tasks="$TASKS"
    fi
    numactl --interleave=all "$PADOC" bench analyze-batch \
        --compressor "$c" --artifact "$file" \
        --tasks "$tasks" --repeat 1 \
        | awk -v ds="$ds" '{ print ds, $0 }' >> "$OUT"
  done
done
echo "wrote $OUT" >&2
