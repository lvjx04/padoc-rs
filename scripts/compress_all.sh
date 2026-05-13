#!/bin/bash
# Compress every dataset with every compressor, write artifacts under
# $OUT (default /mnt/treasure/ljx/artifacts/).  Small datasets and
# llama_full use different worker counts and the per-rank streaming
# flag because the cross-rank load step OOMs on llama-1024.
#
# Usage: scripts/compress_all.sh [out_dir]
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-/mnt/treasure/ljx/artifacts}
PADOC=$ROOT/target/release/padoc
mkdir -p "$OUT"

echo ">>> small datasets (workers=32, cross-rank padoc)"
numactl --interleave=all "$PADOC" bench compress \
    --manifest "$ROOT/scripts/manifest_small.json" \
    --compressors padoc,raw_json,gzip_json,scalatrace,tracezip \
    --workers 32 --out-dir "$OUT"

echo ">>> llama_full (workers=64, --per-rank to keep RSS bounded)"
numactl --interleave=all "$PADOC" bench compress \
    --manifest "$ROOT/scripts/manifest_llama.json" \
    --compressors padoc,raw_json,gzip_json,scalatrace,tracezip \
    --workers 64 --per-rank --out-dir "$OUT"
