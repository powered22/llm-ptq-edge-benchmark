#!/usr/bin/env bash
# Tabel 4: Akurasi GGUF schemes (deployment dimension's accuracy view).
# Engine: llama.cpp via llama-cpp-python + lm-evaluation-harness 'gguf' backend.
# Run di Linux GPU untuk speed (bukan Jetson).
#
# Schemes yang dievaluasi (sama dengan Tabel 2 untuk apple-to-apple):
#   F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0
#
# Task split sama dengan Tabel 1:
#   - Likelihood (no chat template): arc_easy, arc_challenge, hellaswag,
#                                    winogrande, mmlu, truthfulqa_mc2
#   - Generative (with chat template): gsm8k, ifeval
#
# Output: 3 file JSON per scheme = 15 file total
#   ./results/eval/tabel4_<SCHEME>_{likelihood,gsm8k,ifeval}.json
#
# Prasyarat:
#   1. GGUF files ada di ./results/qwen2.5_0.5b_instruct_gguf/
#      (jalankan dulu: bash scripts/jetson/convert_qwen0.5b_to_gguf.sh)
#   2. llama-cpp-python terinstall (idealnya dengan CUDA):
#      CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
#   3. lm-evaluation-harness terinstall
#
# Estimasi: 30-60 menit per scheme × 5 = ~3-5 jam total di Linux GPU.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GGUF_DIR="./results/qwen2.5_0.5b_instruct_gguf"
OUT_DIR="${OUT_DIR:-./results/eval/tabel4_gguf}"
mkdir -p "$OUT_DIR"

# Schemes — sinkron dengan Tabel 2
SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

# Task split — sinkron dengan Tabel 1
LIKELIHOOD_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc2"
GEN_TASKS="gsm8k"
IFEVAL_TASKS="ifeval"

BATCH_SIZE="${BATCH_SIZE:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"  # -1 = semua layer ke GPU
N_CTX="${N_CTX:-2048}"


is_valid_json() {
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null
}


run_part() {
    # run_part <gguf_path> <tasks> <output> <use_template:0|1>
    local gguf_file="$1" tasks="$2" out="$3" use_template="$4"

    # Glob match (lm-eval append _<TIMESTAMP> ke filename)
    local prefix="${out%.json}"
    shopt -s nullglob
    local matches=( "$out" "${prefix}_"*.json )
    shopt -u nullglob

    if [[ -z "${FORCE:-}" ]]; then
        for f in "${matches[@]}"; do
            [[ -f "$f" ]] || continue
            if is_valid_json "$f"; then
                echo "    [skip] $f sudah ada & valid"
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
        --model gguf
        --model_args "pretrained=$gguf_file,n_ctx=$N_CTX,n_gpu_layers=$N_GPU_LAYERS"
        --tasks "$tasks"
        --batch_size "$BATCH_SIZE"
        --log_samples
        --output_path "$out"
    )
    [[ "$use_template" == "1" ]] && args+=(--apply_chat_template)

    python3 -m lm_eval "${args[@]}"
}


n_ok=0; n_fail=0; idx=0
for scheme in "${SCHEMES[@]}"; do
    idx=$((idx+1))
    gguf_file="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"

    if [[ ! -f "$gguf_file" ]]; then
        echo "[skip] $gguf_file tidak ada — jalankan convert dulu"
        continue
    fi

    OUT_LH="$OUT_DIR/tabel4_${scheme}_likelihood.json"
    OUT_GEN="$OUT_DIR/tabel4_${scheme}_gsm8k.json"
    OUT_IFEVAL="$OUT_DIR/tabel4_${scheme}_ifeval.json"

    echo ""
    echo "##################################################"
    echo "  [$idx/${#SCHEMES[@]}] Scheme: $scheme"
    echo "      GGUF: $gguf_file"
    echo "##################################################"

    echo ""
    echo "  Part A: 6 likelihood tasks (NO chat template)"
    if run_part "$gguf_file" "$LIKELIHOOD_TASKS" "$OUT_LH" 0; then
        n_ok=$((n_ok+1)); echo "  Part A: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part A: FAILED"
    fi

    echo ""
    echo "  Part B: gsm8k (WITH chat template)"
    if run_part "$gguf_file" "$GEN_TASKS" "$OUT_GEN" 1; then
        n_ok=$((n_ok+1)); echo "  Part B: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part B: FAILED"
    fi

    echo ""
    echo "  Part C: ifeval (WITH chat template)"
    if run_part "$gguf_file" "$IFEVAL_TASKS" "$OUT_IFEVAL" 1; then
        n_ok=$((n_ok+1)); echo "  Part C: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part C: FAILED"
    fi
done

echo ""
echo "=================================================="
echo "  Tabel 4 selesai."
echo "  Total parts OK   : $n_ok / $((${#SCHEMES[@]} * 3))"
echo "  Total parts FAIL : $n_fail"
echo "=================================================="
echo ""
echo "File output di $OUT_DIR/:"
ls -lh "$OUT_DIR/"tabel4_*.json 2>/dev/null || true
echo ""
echo "Selanjutnya jalankan aggregator:"
echo "  python3 evaluation/aggregate_tabel4.py"
