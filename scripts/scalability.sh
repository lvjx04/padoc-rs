#!/bin/bash
# Compress-time scalability sweep on padoc.
#
# We must cap *both* the user-configured pool (--workers N) and rayon's
# global pool (RAYON_NUM_THREADS=N), because finalize_templates and the
# merge_shards tree-rewrite use rayon::par_iter() which goes through the
# global pool, not the user-built pool.
#
# CPU binding: taskset -c 0..N-1 keeps us on physical cores 0..N-1
# (logical CPUs 64-127 are SMT siblings).  For N <= 32 we stay inside
# NUMA node 0 (memory local).  For N > 32 we span both sockets and
# interleave memory.
cd ~/Work/padoc-rs
LOG=~/Work/padoc-rs/scalability.log
exec >> "$LOG" 2>&1
echo "=== scalability sweep starting at $(date) ==="

ART=/mnt/treasure/ljx/padoc_artifacts/v5-scalability
mkdir -p "$ART"

run_one () {
  local P=$1 N=$2 w=$3
  echo "=========================================="
  echo "=== $N workers=$w at $(date) ==="
  free -h | head -2
  local last=$((w - 1))
  local mem_bind
  if [ "$w" -le 32 ]; then
    mem_bind="--cpunodebind=0 --membind=0"
  else
    mem_bind="--interleave=all"
  fi
  RAYON_NUM_THREADS=$w taskset -c 0-$last numactl $mem_bind \
    ./target/release/padoc bench compress \
      --datasets "$P" \
      --compressors padoc \
      --workers $w \
      --out-dir "$ART/$N-w$w" \
    2>&1 | grep -E "padoc-parallel done|^\| .*padoc.*\|" | tail -3
  echo
}

# Smaller datasets: full sweep including 1 worker.
SMALL=(
  "/mnt/treasure/ljx/Trace_int/leworldmodel_json:lewm"
  "/mnt/treasure/ljx/Trace/qwen3:qwen3"
)
for ds_pn in "${SMALL[@]}"; do
  P="${ds_pn%:*}"; N="${ds_pn##*:}"
  for w in 1 2 4 8 16 32 64 ; do
    run_one "$P" "$N" "$w"
  done
done

# llama: skip 1, 2 (would each take ~30-90 min).  4 onwards.
P=/mnt/treasure/ljx/Trace/llama/profiler
for w in 4 8 16 32 64 ; do
  run_one "$P" "llama" "$w"
done

echo "=== scalability sweep finished at $(date) ==="
