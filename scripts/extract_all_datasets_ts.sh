#!/bin/bash
# Extract kernel timestamps from all 4 datasets for the piecewise-linear figure.
# Outputs CSV files to /tmp/ts_extract/
#
# Usage: bash scripts/extract_all_datasets_ts.sh

set -e

EXTRACT="./target/release/examples/extract_kernel_ts"
OUT="/tmp/ts_extract"
mkdir -p "$OUT"

LEWM="/mnt/treasure/ljx/Trace/leworldmodel_json/trace_rank0.json"
QWEN3="/mnt/treasure/ljx/Trace/qwen3/profiler_0.json"
UNIFOLM="/mnt/treasure/ljx/Trace_int/unifolm-world-model_json/global_rank0.json"
LLAMA="/mnt/treasure/ljx/Trace/llama/4ranks/profiler_0.json"

echo "=== LeWorldModel ==="
echo "  fused_merge_kernel..."
$EXTRACT "$LEWM" "fused_merge_kernel" > "$OUT/lewm_fused_merge_kernel.csv"
echo "  vectorized_layer_norm_kernel..."
$EXTRACT "$LEWM" "vectorized_layer_norm_kernel" > "$OUT/lewm_layer_norm.csv"
echo "  Cijk_Alik_Bljk..."
$EXTRACT "$LEWM" "Cijk_Alik_Bljk" > "$OUT/lewm_Cijk.csv"
echo "  elementwise_kernel_manual_unroll..."
$EXTRACT "$LEWM" "elementwise_kernel_manual_unroll" > "$OUT/lewm_elementwise.csv"

echo "=== Qwen3 ==="
echo "  ac2g..."
$EXTRACT "$QWEN3" "ac2g" > "$OUT/qwen3_ac2g.csv"
echo "  fwdbwd..."
$EXTRACT "$QWEN3" "fwdbwd" > "$OUT/qwen3_fwdbwd.csv"
echo "  suLaunchKernel..."
$EXTRACT "$QWEN3" "suLaunchKernel" > "$OUT/qwen3_suLaunchKernel.csv"

echo "=== UniFolm ==="
echo "  ac2g..."
$EXTRACT "$UNIFOLM" "ac2g" > "$OUT/unifolm_ac2g.csv"
echo "  conv..."
$EXTRACT "$UNIFOLM" "conv" > "$OUT/unifolm_conv.csv"
echo "  gemm..."
$EXTRACT "$UNIFOLM" "gemm" > "$OUT/unifolm_gemm.csv"
echo "  elementwise_kernel<128, 4..."
$EXTRACT "$UNIFOLM" "elementwise_kernel<128, 4" > "$OUT/unifolm_elementwise.csv"

echo "=== LLaMA-70B ==="
echo "  ac2g..."
$EXTRACT "$LLAMA" "ac2g" > "$OUT/llama_ac2g.csv"
echo "  Gemm_tcu_mr_kernel..."
$EXTRACT "$LLAMA" "Gemm_tcu_mr_kernel" > "$OUT/llama_gemm_tcu.csv"
echo "  elementwise_kernel.*CUDAFunctor_add..."
$EXTRACT "$LLAMA" "CUDAFunctor_add" > "$OUT/llama_elementwise_add.csv"
echo "  genericOp..."
$EXTRACT "$LLAMA" "genericOp" > "$OUT/llama_nccl.csv"

echo ""
echo "=== Done! CSV files in $OUT ==="
ls -la "$OUT"/*.csv
