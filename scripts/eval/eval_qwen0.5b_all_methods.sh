#!/usr/bin/env bash
# Tabel 1: Akurasi semua metode kuantisasi untuk Qwen2.5-0.5B-Instruct.
# Engine: HF transformers + lm-evaluation-harness, di Linux GPU.
#
# Split tasks (berdasarkan smoketest comparison 2026-05-27):
#   - Likelihood-based (log-prob multiple choice): TANPA chat template
#       arc_easy, arc_challenge, hellaswag, winogrande, mmlu, truthfulqa_mc
#     → Smoketest menunjukkan tanpa template +10pp lebih akurat untuk Qwen-Instruct.
#   - Generative (free-form text): DENGAN chat template
#       gsm8k
#     → Convention: instruct model butuh chat format untuk produce structured answer.
#
# Output per metode: 2 file JSON
#   ./results/eval/qwen2.5-0.5b-instruct_<label>_likelihood.json
#   ./results/eval/qwen2.5-0.5b-instruct_<label>_gsm8k.json
#
# Total: 6 metode × 2 part = 12 lm-eval invocations. Estimasi 3-6 jam.
#
# Prasyarat:
#   1. pip install 'lm-eval>=0.4.2' compressed-tensors accelerate
#   2. export HF_TOKEN=hf_xxxxxxxx   (untuk model private poweredshine/*)
#   3. CUDA aktif (cek: python -c "import torch; print(torch.cuda.is_available())")
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: env var HF_TOKEN belum di-set."
    echo "  export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

# Task split — JANGAN diubah tanpa re-validasi via smoketest.
LIKELIHOOD_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc2"
GEN_TASKS="gsm8k"

BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-./results/eval}"
mkdir -p "$OUT_DIR"

# Format: "label:method:model_path"
METHODS=(
    "fp16:fp16:Qwen/Qwen2.5-0.5B-Instruct"
    "awqW4A16:awq:poweredshine/qwen2.5_0.5b_instruct_awq_w4a16"
    "gptqW4A16:gptq:poweredshine/qwen2.5_0.5b_instruct_gptq_w4a16"
    "rtnW4A16:rtn:poweredshine/qwen2.5_0.5b_instruct_rtn_w4a16"
    "rtnW8A16:rtn:poweredshine/qwen2.5_0.5b_instruct_rtn_w8a16"
    "smoothW8A8:smoothquant:poweredshine/qwen2.5_0.5b_instruct_smooth_w8a8"
)

run_part() {
    # run_part <model_path> <method> <tasks> <output> <use_template:0|1>
    local model_path="$1" method="$2" tasks="$3" out="$4" use_template="$5"

    if [[ -f "$out" ]] && [[ -z "${FORCE:-}" ]]; then
        echo "    [skip] $out sudah ada (set FORCE=1 untuk re-run)"
        return 0
    fi

    local args=(--model-path "$model_path" --method "$method" --tasks "$tasks"
                --batch-size "$BATCH_SIZE" --output "$out")
    [[ "$use_template" == "1" ]] && args+=(--apply-chat-template)

    python3 evaluation/run_lm_harness.py "${args[@]}"
}

n_ok=0; n_fail=0; idx=0
for spec in "${METHODS[@]}"; do
    idx=$((idx+1))
    label="${spec%%:*}"
    rest="${spec#*:}"
    method="${rest%%:*}"
    model_path="${rest#*:}"

    OUT_LH="$OUT_DIR/qwen2.5-0.5b-instruct_${label}_likelihood.json"
    OUT_GEN="$OUT_DIR/qwen2.5-0.5b-instruct_${label}_gsm8k.json"

    echo ""
    echo "##################################################"
    echo "  [$idx/${#METHODS[@]}] $label   ($method)"
    echo "      Model: $model_path"
    echo "##################################################"

    echo ""
    echo "  Part A: 6 likelihood tasks (NO chat template)"
    if run_part "$model_path" "$method" "$LIKELIHOOD_TASKS" "$OUT_LH" 0; then
        n_ok=$((n_ok+1))
        echo "  Part A: OK"
    else
        n_fail=$((n_fail+1))
        echo "  Part A: FAILED"
    fi

    echo ""
    echo "  Part B: gsm8k (WITH chat template)"
    if run_part "$model_path" "$method" "$GEN_TASKS" "$OUT_GEN" 1; then
        n_ok=$((n_ok+1))
        echo "  Part B: OK"
    else
        n_fail=$((n_fail+1))
        echo "  Part B: FAILED"
    fi
done

echo ""
echo "=================================================="
echo "  Tabel 1 selesai."
echo "  Total parts OK   : $n_ok / $((${#METHODS[@]} * 2))"
echo "  Total parts FAIL : $n_fail"
echo "=================================================="
echo ""
echo "Output di $OUT_DIR/:"
ls -lh "$OUT_DIR/"qwen2.5-0.5b-instruct_*.json 2>/dev/null || true
echo ""
echo "Langkah selanjutnya: aggregate 12 JSON ini ke satu CSV Tabel 1."
