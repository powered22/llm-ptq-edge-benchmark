#!/usr/bin/env bash
# Bandingkan Qwen2.5-0.5B-Instruct: FP16 baseline VS GPTQ W4A16.
# Pakai lm-evaluation-harness preset 'full' (7 tasks: arc_easy, arc_challenge,
# hellaswag, winogrande, mmlu, truthfulqa_mc, gsm8k). Estimasi: 1-2 jam.
#
# Prasyarat:
#   1. pip install lm-eval>=0.4.2 auto-gptq accelerate
#   2. Model GPTQ sudah di-download:  ./download_qwen_models.sh
#      (atau minimal: HF_TOKEN=hf_xxx huggingface-cli download \
#         poweredshine/qwen2.5_0.5b_instruct_gptq_w4a16 \
#         --local-dir ./results/qwen2.5_0.5b_instruct_gptq_w4a16)
#
# Output:
#   ./results/eval/qwen2.5-0.5b-instruct_fp16.json
#   ./results/eval/qwen2.5-0.5b-instruct_gptqW4A16.json

set -euo pipefail

# Pindah ke repo root.
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

TASKS="${TASKS:-full}"                  # quick | standard | full
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-./results/eval}"
mkdir -p "$OUT_DIR"

GPTQ_LOCAL_PATH="./results/qwen2.5_0.5b_instruct_gptq_w4a16"
if [[ ! -d "$GPTQ_LOCAL_PATH" ]]; then
    echo "ERROR: $GPTQ_LOCAL_PATH belum ada."
    echo "  Download dulu via ./download_qwen_models.sh (atau huggingface-cli)."
    exit 1
fi

run_one() {
    local label="$1" model_path="$2" method="$3"
    local out="$OUT_DIR/qwen2.5-0.5b-instruct_${label}.json"

    echo ""
    echo "=================================================="
    echo "  [$label]  model=$model_path  method=$method"
    echo "  tasks=$TASKS  batch_size=$BATCH_SIZE"
    echo "  -> $out"
    echo "=================================================="

    python3 evaluation/run_lm_harness.py \
        --model-path "$model_path" \
        --method "$method" \
        --tasks "$TASKS" \
        --batch-size "$BATCH_SIZE" \
        --output "$out"
}

# 1. FP16 baseline (download otomatis dari HF official saat pertama jalan).
run_one "fp16" "Qwen/Qwen2.5-0.5B-Instruct" "fp16"

# 2. GPTQ W4A16 (model lokal hasil download).
run_one "gptqW4A16" "$GPTQ_LOCAL_PATH" "gptq"

echo ""
echo "Selesai. Hasil ada di $OUT_DIR/"
echo "Buka kedua JSON untuk bandingkan akurasi per-task."
