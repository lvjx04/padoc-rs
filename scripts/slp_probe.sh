#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-results/remaining/slp_probe}"
MIN_LEN="${MIN_LEN:-1024}"
MAX_COLUMNS="${MAX_COLUMNS:-128}"
MAX_VALUES="${MAX_VALUES:-1000000}"
MAX_SEGMENT_LEN="${MAX_SEGMENT_LEN:-65536}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"

dataset_args=()
if [[ -n "${DATASETS:-}" ]]; then
  for dataset in $DATASETS; do
    dataset_args+=(--dataset "$dataset")
  done
fi

cargo run --release --example slp_probe -- \
  "${dataset_args[@]}" \
  --out-dir "$OUT_DIR" \
  --min-len "$MIN_LEN" \
  --max-columns "$MAX_COLUMNS" \
  --max-values "$MAX_VALUES" \
  --max-segment-len "$MAX_SEGMENT_LEN" \
  --zstd-level "$ZSTD_LEVEL"
