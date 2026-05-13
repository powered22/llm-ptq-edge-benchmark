#!/usr/bin/env bash
# Run all quantization methods on Qwen2.5-1.5B.
# Output ke ./results/<output_name>/

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B"
NUM_SAMPLES=512
MAX_SEQ_LENGTH=512
OUT_DIR="./results"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"

run_step() {
    local label="$1"; shift
    echo ""
    echo "=========================================="
    echo "  $label"
    echo "=========================================="
    "$@"
}

# 1. AWQ W4A16
# run_step "[1/6] AWQ W4A16" \
#   python quantization/quantize_awq_v2.py \
#     --model "$MODEL" \
#     --output "$OUT_DIR/qwen2.5_0.5b_instruct_awq_w4a16" \
#     --num-calibration-samples "$NUM_SAMPLES" \
#     --max-seq-length "$MAX_SEQ_LENGTH"

# 2. GPTQ W4A16
# run_step "[2/6] GPTQ W4A16" \
#   python quantization/quantize_gptq_v2.py \
#     --model "$MODEL" \
#     --output "$OUT_DIR/qwen2.5_0.5b_instruct_gptq_w4a16" \
#     --num-samples "$NUM_SAMPLES" \
#     --scheme W4A16

# 3. GPTQ W8A16
# run_step "[3/6] GPTQ W8A16" \
#   python quantization/quantize_gptq_v2.py \
#     --model "$MODEL" \
#     --output "$OUT_DIR/qwen2.5_1.5b_gptq_w8a16" \
#     --num-samples "$NUM_SAMPLES" \
#     --scheme W8A16

# 4. RTN W4A16
# run_step "[4/6] RTN W4A16" \
#   python quantization/quantize_rtn.py \
#     --model "$MODEL" \
#     --output "$OUT_DIR/qwen2.5_1.5b_rtn_w4a16" \
#     --scheme W4A16

# 5. RTN W8A16
run_step "[5/6] RTN W8A16" \
  python quantization/quantize_rtn.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_1.5b_rtn_w8a16" \
    --scheme W8A16

# 6. SmoothQuant W8A8x``
# run_step "[6/6] SmoothQuant W8A8" \
#   python quantization/quantize_smoothquant.py \
#     --model "$MODEL" \
#     --output "$OUT_DIR/qwen2.5_1.5b_smooth_w8a8"

echo ""
echo "=========================================="
echo "  All 1 quantization runs complete."
echo "  Output: $OUT_DIR/qwen2.5_1.5b_*"
echo "=========================================="
