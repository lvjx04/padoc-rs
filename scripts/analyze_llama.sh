#!/bin/bash
# Analyse benchmark for llama_full.  PADOC is one process loading the
# merged artifact; baselines are run per-rank (1024 ranks each) and
# the analyse seconds are summed, peak_rss is taken as the max single
# rank.  Output goes to results/main/analyze_llama_padoc.tsv (PADOC,
# one row per task) and results/main/analyze_llama_baselines.tsv
# (raw/gzip/scalatrace/tracezip, four rows each).
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:-/mnt/treasure/ljx/artifacts}
PAR=${2:-48}
PADOC="$ROOT/target/release/padoc"
TASKS=${TASKS:-operator_hotspot,rank_load_balance}
PADOC_CORE_TASKS=${PADOC_CORE_TASKS:-operator_hotspot,rank_load_balance,layer_kernel_hotspot,layer_compute_comm_overlap,layer_rank_balance}

# --- PADOC: single process, in-situ on merged artifact ---
PADOC_OUT="$ROOT/results/main/analyze_llama_padoc.tsv"
echo ">>> llama padoc (single process)" >&2
numactl --interleave=all "$PADOC" bench analyze-batch \
    --compressor padoc \
    --artifact "$ART/llama_full.padoc.zst" \
    --tasks "$PADOC_CORE_TASKS" --repeat 1 > "$PADOC_OUT"

# --- baselines: per-rank parallel sweep, then aggregate ---
BASE_OUT="$ROOT/results/main/analyze_llama_baselines.tsv"
PERRANK_DIR="$ART/llama_full"
> "$BASE_OUT"
for c in raw_json gzip_json scalatrace tracezip; do
  raw="$BASE_OUT.$c.raw"
  files=( "$PERRANK_DIR"/*."$c".bin )
  total=${#files[@]}
  [ $total -eq 0 ] && { echo "missing per-rank $c artifacts under $PERRANK_DIR" >&2; continue; }
  echo ">>> llama $c ($total files, parallelism=$PAR)" >&2
  printf '%s\n' "${files[@]}" \
    | xargs -P "$PAR" -I{} "$PADOC" bench analyze-batch \
        --compressor "$c" --artifact {} --tasks "$TASKS" --repeat 1 \
        2>/dev/null >> "$raw"
  awk -v c="$c" -F'\t' '
    {
      load[$2]+=$4; dec[$2]+=$5; an[$2]+=$6
      if ($9>rss) rss=$9
      tasks[$2]=1
    }
    END {
      for (t in tasks) {
        printf "%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n",
               c, t, load[t], dec[t], an[t], load[t]+dec[t]+an[t], rss
      }
    }
  ' "$raw" >> "$BASE_OUT"
  rm -f "$raw"
done
echo "wrote $PADOC_OUT and $BASE_OUT" >&2
