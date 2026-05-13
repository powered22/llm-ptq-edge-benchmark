#!/usr/bin/env bash
# ============================================================================
# Evaluasi 12 model Qwen2.5 (0.5B-Instruct + 1.5B) hasil llm-compressor.
#
# Dua skrip dijalankan per model:
#   1. evaluation/run_perplexity.py   → PPL di WikiText-2
#   2. evaluation/run_lm_harness.py   → akurasi task (lm-evaluation-harness)
#
# Sanity-check awal pakai task ringan (ARC-easy + HellaSwag). Task lain
# (MMLU, GSM8K, TruthfulQA, HumanEval) sudah disiapkan tapi di-comment.
#
# Model llm-compressor disimpan dalam format `compressed-tensors`, jadi
# loader-nya cukup AutoModelForCausalLM.from_pretrained — pakai
# `--method fp16` di kedua skrip eval supaya tidak masuk cabang BnB.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RESULTS_DIR="./results"
EVAL_DIR="./results/eval"

# Task preset untuk lm-eval harness. Saat ini sanity-check:
TASKS="arc_easy,hellaswag"
# Pilihan lain (uncomment salah satu kalau mau pakai):
# TASKS="arc_easy,hellaswag,winogrande,arc_challenge"
# TASKS="mmlu"
# TASKS="gsm8k"
# TASKS="truthfulqa_mc2"
# TASKS="humaneval"   # butuh: export HF_ALLOW_CODE_EVAL=1 + --confirm_run_unsafe_code

BATCH_SIZE=4
NUM_FEWSHOT=0
PPL_DATASET="wikitext2"

mkdir -p "$EVAL_DIR"

# ---------------------------------------------------------------------------
# Daftar 12 model: "<folder_basename>"
# Path penuh = $RESULTS_DIR/<folder_basename>
# ---------------------------------------------------------------------------
MODELS=(
  # Qwen2.5-0.5B-Instruct (6)
  "qwen2.5_0.5b_instruct_awq_w4a16"
  "qwen2.5_0.5b_instruct_gptq_w4a16"
  "qwen2.5_0.5b_instruct_gptq_w8a16"
  "qwen2.5_0.5b_instruct_rtn_w8a16"
  "qwen2.5_0.5b_instruct_rtn_w4a16"
  "qwen2.5_0.5b_instruct_smooth_w8a8"
  # Qwen2.5-1.5B (6)
  "qwen2.5-1.5b-awq-int4-sym"
  "qwen2.5-1.5b-gptq-int4"
  "qwen2.5-1.5b-gptq-w8a16"
  "qwen2.5-1.5b-rtn-w4a16"
  "qwen2.5-1.5b-rtn-w8a16"
  "qwen2.5_1.5b_smooth_w8a8"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_step() {
  local label="$1"; shift
  echo ""
  echo "=========================================="
  echo "  $label"
  echo "=========================================="
  "$@"
}

# ---------------------------------------------------------------------------
# Loop evaluasi
# ---------------------------------------------------------------------------
TOTAL=${#MODELS[@]}
IDX=0
SKIPPED=()

for name in "${MODELS[@]}"; do
  IDX=$((IDX + 1))
  MODEL_PATH="$RESULTS_DIR/$name"

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[skip] ($IDX/$TOTAL) $name — folder tidak ditemukan di $MODEL_PATH"
    SKIPPED+=("$name")
    continue
  fi

  PPL_OUT="$EVAL_DIR/${name}__ppl_${PPL_DATASET}.json"
  HARNESS_OUT="$EVAL_DIR/${name}__harness"

  # 1. Perplexity
  if [[ -f "$PPL_OUT" ]]; then
    echo "[cached] ($IDX/$TOTAL) PPL $name — sudah ada di $PPL_OUT"
  else
    run_step "[$IDX/$TOTAL] PPL  $name" \
      python evaluation/run_perplexity.py \
        --model-path "$MODEL_PATH" \
        --method fp16 \
        --dataset "$PPL_DATASET" \
        --output "$PPL_OUT"
  fi

  # 2. lm-evaluation-harness
  if [[ -d "$HARNESS_OUT" && -n "$(ls -A "$HARNESS_OUT" 2>/dev/null)" ]]; then
    echo "[cached] ($IDX/$TOTAL) Harness $name — sudah ada di $HARNESS_OUT"
  else
    run_step "[$IDX/$TOTAL] Harness $name ($TASKS)" \
      python evaluation/run_lm_harness.py \
        --model-path "$MODEL_PATH" \
        --method fp16 \
        --tasks "$TASKS" \
        --batch-size "$BATCH_SIZE" \
        --num-fewshot "$NUM_FEWSHOT" \
        --output "$HARNESS_OUT"
  fi
done

# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Evaluasi selesai"
echo "  Output: $EVAL_DIR"
if (( ${#SKIPPED[@]} > 0 )); then
  echo "  Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"
fi
echo "=========================================="
