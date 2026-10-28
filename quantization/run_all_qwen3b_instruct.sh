#!/usr/bin/env bash
# Run the five Table 1 quantization methods on Qwen2.5-3B-Instruct (scaling check of the 0.5B-Instruct study).
# Output ke ./results/<output_name>/

set -euo pipefail

MODEL="Qwen/Qwen2.5-3B-Instruct"
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
run_step "[1/5] AWQ W4A16" \
  python quantization/quantize_awq_v2.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_3b_instruct_awq_w4a16" \
    --num-calibration-samples "$NUM_SAMPLES" \
    --max-seq-length "$MAX_SEQ_LENGTH"

# 2. GPTQ W4A16
run_step "[2/5] GPTQ W4A16" \
  python quantization/quantize_gptq_v2.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_3b_instruct_gptq_w4a16" \
    --num-samples "$NUM_SAMPLES" \
    --scheme W4A16

# 3. RTN W4A16
run_step "[3/5] RTN W4A16" \
  python quantization/quantize_rtn.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_3b_instruct_rtn_w4a16" \
    --scheme W4A16

# 4. RTN W8A16
run_step "[4/5] RTN W8A16" \
  python quantization/quantize_rtn.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_3b_instruct_rtn_w8a16" \
    --scheme W8A16

# 5. SmoothQuant W8A8
run_step "[5/5] SmoothQuant W8A8" \
  python quantization/quantize_smoothquant.py \
    --model "$MODEL" \
    --output "$OUT_DIR/qwen2.5_3b_instruct_smooth_w8a8"

echo ""
echo "=========================================="
echo "  All 5 quantization runs complete."
echo "  Output: $OUT_DIR/qwen2.5_3b_instruct_*"
echo "=========================================="
