#!/usr/bin/env bash
# Tabel 4 FULL: 8 task akurasi via custom llama-cpp-direct backend.
# 0-shot semua tasks → konsisten dengan Tabel 1 yang juga 0-shot (internal apple-to-apple).
#
# Task split sama dengan Tabel 1:
#   - Likelihood (no chat template): arc_easy, arc_challenge, hellaswag,
#                                    winogrande, mmlu, truthfulqa_mc2
#   - Generative (with chat template): gsm8k, ifeval
#
# Output: 3 file JSON per scheme → 5 schemes × 3 part = 15 files
#   ./results/eval/tabel4_gguf/tabel4_<SCHEME>_{likelihood,gsm8k,ifeval}.json
#
# Prasyarat:
#   1. Fix torch CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
#   2. llama-cpp-python terinstall (sudah)
#   3. lm-eval-harness + dependencies (tenacity untuk api models, langdetect untuk ifeval)
#
# Estimasi: ~3-6 jam per scheme dengan LIMIT=500 → ~15-30 jam total.
# Pakai LIMIT untuk subsample atau biarkan kosong untuk full eval.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GGUF_DIR="./results/qwen2.5_0.5b_instruct_gguf"
OUT_DIR="${OUT_DIR:-./results/eval/tabel4_gguf}"
mkdir -p "$OUT_DIR"

SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

# Task groups (sinkron dengan eval_qwen0.5b_all_methods.sh untuk Tabel 1)
LIKELIHOOD_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc2"
GEN_TASKS="gsm8k"
IFEVAL_TASKS="ifeval"

N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CTX="${N_CTX:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"


is_valid_json() {
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null
}

run_part() {
    local gguf_file="$1" tasks="$2" out="$3" use_template="$4"

    local prefix="${out%.json}"
    shopt -s nullglob
    local matches=( "$out" "${prefix}_"*.json )
    shopt -u nullglob

    if [[ -z "${FORCE:-}" ]]; then
        for f in "${matches[@]}"; do
            [[ -f "$f" ]] || continue
            if is_valid_json "$f"; then
                echo "    [skip] $f sudah valid"
                return 0
            fi
        done
    fi

    for f in "${matches[@]}"; do
        if [[ -f "$f" ]] && ! is_valid_json "$f"; then
            rm "$f" && echo "    [clean] hapus corrupt: $f"
        fi
    done

    local args=(
        --model llama-cpp-direct
        --model_args "pretrained=$gguf_file,n_ctx=$N_CTX,n_gpu_layers=$N_GPU_LAYERS"
        --tasks "$tasks"
        --num_fewshot 0
        --batch_size "$BATCH_SIZE"
        --log_samples
        --output_path "$out"
    )
    [[ "$use_template" == "1" ]] && args+=(--apply_chat_template)
    [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")

    python3 scripts/run_lm_eval_gguf.py "${args[@]}"
}


n_ok=0; n_fail=0; idx=0
for scheme in "${SCHEMES[@]}"; do
    idx=$((idx+1))
    gguf_file="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"
    [[ -f "$gguf_file" ]] || { echo "[skip] $gguf_file tidak ada"; continue; }

    OUT_LH="$OUT_DIR/tabel4_${scheme}_likelihood.json"
    OUT_GEN="$OUT_DIR/tabel4_${scheme}_gsm8k.json"
    OUT_IFEVAL="$OUT_DIR/tabel4_${scheme}_ifeval.json"

    echo ""
    echo "##################################################"
    echo "  [$idx/${#SCHEMES[@]}] Scheme: $scheme"
    echo "##################################################"

    echo ""; echo "  Part A: 6 likelihood tasks (0-shot, NO chat template)"
    if run_part "$gguf_file" "$LIKELIHOOD_TASKS" "$OUT_LH" 0; then
        n_ok=$((n_ok+1)); echo "  Part A: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part A: FAILED"
    fi

    echo ""; echo "  Part B: gsm8k (0-shot, WITH chat template)"
    if run_part "$gguf_file" "$GEN_TASKS" "$OUT_GEN" 1; then
        n_ok=$((n_ok+1)); echo "  Part B: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part B: FAILED"
    fi

    echo ""; echo "  Part C: ifeval (0-shot, WITH chat template)"
    if run_part "$gguf_file" "$IFEVAL_TASKS" "$OUT_IFEVAL" 1; then
        n_ok=$((n_ok+1)); echo "  Part C: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part C: FAILED"
    fi
done

echo ""
echo "=================================================="
echo "  Tabel 4 selesai. OK: $n_ok / $((${#SCHEMES[@]} * 3)) | Failed: $n_fail"
echo "=================================================="
echo ""
echo "Aggregate: python3 evaluation/aggregate_tabel4.py"
