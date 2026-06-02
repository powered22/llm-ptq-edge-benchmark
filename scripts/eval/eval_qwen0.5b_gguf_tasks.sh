#!/usr/bin/env bash
# Tabel 4 supplementary: HellaSwag + Winogrande task accuracy via llama-perplexity
# built-in modes. Melengkapi metric perplexity dari script utama.
#
# Output per scheme (di tabel4_gguf/):
#   - hellaswag_<SCHEME>.txt
#   - winogrande_<SCHEME>.txt
#
# Prasyarat:
#   1. Jalankan dulu: python3 evaluation/prepare_llama_cpp_eval_data.py
#      → menghasilkan ./data/hellaswag_val_full.txt + winogrande CSV
#   2. Binary llama-perplexity ada di external/llama.cpp/build/bin/
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REPO_ROOT="$(pwd)"
LLAMA_PPL="$REPO_ROOT/external/llama.cpp/build/bin/llama-perplexity"
GGUF_DIR="./results/qwen2.5_0.5b_instruct_gguf"
OUT_DIR="${OUT_DIR:-./results/eval/tabel4_gguf}"
DATA_DIR="${DATA_DIR:-./data}"

HELLASWAG_FILE="$DATA_DIR/hellaswag_val_full.txt"
WINOGRANDE_FILE="$DATA_DIR/winogrande-debiased-eval.csv"

[[ -x "$LLAMA_PPL" ]] || { echo "ERROR: $LLAMA_PPL tidak ada"; exit 1; }
[[ -f "$HELLASWAG_FILE" ]] || {
    echo "ERROR: $HELLASWAG_FILE belum ada."
    echo "  Jalankan dulu: python3 evaluation/prepare_llama_cpp_eval_data.py"
    exit 1
}
[[ -f "$WINOGRANDE_FILE" ]] || {
    echo "ERROR: $WINOGRANDE_FILE belum ada."
    echo "  Jalankan dulu: python3 evaluation/prepare_llama_cpp_eval_data.py"
    exit 1
}

SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CTX="${N_CTX:-2048}"

# Kurangi jumlah tasks untuk speed (default semua = lambat).
# 1000 sample sudah cukup untuk estimate yang reliable (stderr <0.015).
HELLASWAG_TASKS="${HELLASWAG_TASKS:-1000}"
WINOGRANDE_TASKS="${WINOGRANDE_TASKS:-1267}"  # full validation set


n_ok=0; n_fail=0
for scheme in "${SCHEMES[@]}"; do
    gguf_file="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"
    if [[ ! -f "$gguf_file" ]]; then
        echo "[skip] $gguf_file tidak ada"
        continue
    fi

    echo ""
    echo "##################################################"
    echo "  Scheme: $scheme"
    echo "##################################################"

    # --- HellaSwag ---
    HS_OUT="$OUT_DIR/hellaswag_${scheme}.txt"
    if [[ -f "$HS_OUT" ]] && grep -q "Final HellaSwag score" "$HS_OUT" && [[ -z "${FORCE:-}" ]]; then
        echo "  [skip] HellaSwag $scheme sudah ada & valid"
        n_ok=$((n_ok+1))
    else
        echo "  Running HellaSwag (--hellaswag-tasks $HELLASWAG_TASKS)..."
        if "$LLAMA_PPL" \
            -m "$gguf_file" \
            -f "$HELLASWAG_FILE" \
            --hellaswag --hellaswag-tasks "$HELLASWAG_TASKS" \
            -ngl "$N_GPU_LAYERS" \
            -c "$N_CTX" \
            2>&1 | tee "$HS_OUT" > /dev/null; then
            score=$(grep -oP 'Final HellaSwag score\(\d+ tasks\)\s*=\s*\K[\d.]+' "$HS_OUT" || echo "?")
            echo "  HellaSwag $scheme: $score%"
            n_ok=$((n_ok+1))
        else
            echo "  HellaSwag $scheme: FAILED"
            n_fail=$((n_fail+1))
        fi
    fi

    # --- Winogrande ---
    WG_OUT="$OUT_DIR/winogrande_${scheme}.txt"
    if [[ -f "$WG_OUT" ]] && grep -q "Final Winogrande score" "$WG_OUT" && [[ -z "${FORCE:-}" ]]; then
        echo "  [skip] Winogrande $scheme sudah ada & valid"
        n_ok=$((n_ok+1))
    else
        echo "  Running Winogrande (--winogrande-tasks $WINOGRANDE_TASKS)..."
        if "$LLAMA_PPL" \
            -m "$gguf_file" \
            -f "$WINOGRANDE_FILE" \
            --winogrande --winogrande-tasks "$WINOGRANDE_TASKS" \
            -ngl "$N_GPU_LAYERS" \
            -c "$N_CTX" \
            2>&1 | tee "$WG_OUT" > /dev/null; then
            score=$(grep -oP 'Final Winogrande score\(\d+ tasks\)\s*=\s*\K[\d.]+' "$WG_OUT" || echo "?")
            echo "  Winogrande $scheme: $score%"
            n_ok=$((n_ok+1))
        else
            echo "  Winogrande $scheme: FAILED"
            n_fail=$((n_fail+1))
        fi
    fi
done

echo ""
echo "=================================================="
echo "  Selesai. OK: $n_ok / $((${#SCHEMES[@]} * 2)) | Failed: $n_fail"
echo "=================================================="
echo ""
echo "Aggregate hasil:"
echo "  python3 evaluation/aggregate_tabel4_perplexity.py"
echo "  (aggregator akan auto-pick HellaSwag + Winogrande dari output text files)"
