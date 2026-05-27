#!/usr/bin/env bash
# Smoketest pipeline lm-evaluation-harness sebelum jalankan eksperimen besar.
# Hanya FP16 baseline + preset 'quick' (arc_easy + hellaswag, 2 tasks).
# Estimasi: 10-20 menit di Linux GPU.
#
# Tujuan: verifikasi lm-eval, model loading, dan format output sebelum
#         commit ke run lengkap 6 metode × 7 tasks (~6-12 jam).
#
# Prasyarat:
#   pip install 'lm-eval>=0.4.2' compressed-tensors accelerate
#
# Output: ./results/eval/smoketest_qwen2.5-0.5b-instruct_fp16.json
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUT_DIR="./results/eval"
mkdir -p "$OUT_DIR"

OUTPUT_JSON="$OUT_DIR/smoketest_qwen2.5-0.5b-instruct_fp16.json"

echo "=================================================="
echo "  Smoketest pipeline lm-eval-harness"
echo "  Model      : Qwen/Qwen2.5-0.5B-Instruct (FP16)"
echo "  Tasks      : quick (arc_easy + hellaswag)"
echo "  Batch size : 8"
echo "  Output     : $OUTPUT_JSON"
echo "=================================================="

python3 evaluation/run_lm_harness.py \
    --model-path "Qwen/Qwen2.5-0.5B-Instruct" \
    --method fp16 \
    --tasks quick \
    --batch-size 8 \
    --apply-chat-template \
    --output "$OUTPUT_JSON"

echo ""
echo "Smoketest selesai. Periksa angka akurasi:"
echo "  jq '.results' $OUTPUT_JSON"
echo ""
echo "Yang Anda harapkan:"
echo "  arc_easy.acc,none      ~0.60-0.65"
echo "  hellaswag.acc_norm,none ~0.50-0.55"
echo ""
echo "Kalau angka di rentang itu DAN tidak ada error, pipeline OK."
echo "Lanjut jalankan: ./scripts/eval/eval_qwen0.5b_all_methods.sh"
