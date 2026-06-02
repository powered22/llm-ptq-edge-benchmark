#!/usr/bin/env bash
# Tabel 4 (revisi): Perplexity GGUF schemes via llama-perplexity (built-in llama.cpp).
# Pure C++, NO HTTP layer, NO lm-eval — bypass semua issue compatibility.
#
# Output: 1 perplexity number per scheme (5 total)
#         → ./results/eval/tabel4_gguf/perplexity_<SCHEME>.txt
#         → ./results/eval/tabel4_gguf/tabel4_perplexity.csv (setelah aggregator)
#
# Prasyarat: llama-perplexity binary di external/llama.cpp/build/bin/
#
# Estimasi: 5-15 menit per scheme × 5 = ~30-60 menit total.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REPO_ROOT="$(pwd)"
LLAMA_PPL="$REPO_ROOT/external/llama.cpp/build/bin/llama-perplexity"
GGUF_DIR="./results/qwen2.5_0.5b_instruct_gguf"
OUT_DIR="${OUT_DIR:-./results/eval/tabel4_gguf}"
DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$OUT_DIR" "$DATA_DIR"

[[ -x "$LLAMA_PPL" ]] || { echo "ERROR: $LLAMA_PPL tidak ada"; exit 1; }

SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CTX="${N_CTX:-2048}"
WIKI_TEST="$DATA_DIR/wikitext-2-raw/wiki.test.raw"

# Download wikitext-2 raw test set kalau belum ada (~1 MB, sekali saja).
# Pakai HuggingFace datasets (lebih reliable dari Salesforce S3 yang sering hang).
if [[ ! -f "$WIKI_TEST" ]]; then
    echo "[setup] Download wikitext-2 dari HuggingFace datasets..."
    mkdir -p "$(dirname "$WIKI_TEST")"
    python3 - <<EOF
from datasets import load_dataset
print("  Loading dataset...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print(f"  Got {len(ds)} rows, writing to file...")
with open("$WIKI_TEST", "w") as f:
    for row in ds:
        f.write(row["text"])
print(f"  Saved to: $WIKI_TEST")
EOF
fi

[[ -f "$WIKI_TEST" ]] || { echo "ERROR: $WIKI_TEST tidak ada setelah download"; exit 1; }

n_ok=0; n_fail=0
for scheme in "${SCHEMES[@]}"; do
    gguf_file="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"
    out_file="$OUT_DIR/perplexity_${scheme}.txt"

    if [[ ! -f "$gguf_file" ]]; then
        echo "[skip] $gguf_file tidak ada"
        continue
    fi

    if [[ -f "$out_file" ]] && [[ -z "${FORCE:-}" ]]; then
        # Cek output mengandung "Final estimate: PPL"
        if grep -q "Final estimate: PPL" "$out_file"; then
            echo "[skip] $out_file sudah ada & valid"
            n_ok=$((n_ok+1))
            continue
        else
            echo "[clean] $out_file ada tapi tidak complete — hapus"
            rm "$out_file"
        fi
    fi

    echo ""
    echo "##################################################"
    echo "  Scheme: $scheme"
    echo "  GGUF: $gguf_file"
    echo "  Output: $out_file"
    echo "##################################################"

    if "$LLAMA_PPL" \
        -m "$gguf_file" \
        -f "$WIKI_TEST" \
        -ngl "$N_GPU_LAYERS" \
        -c "$N_CTX" \
        2>&1 | tee "$out_file"; then

        if grep -q "Final estimate: PPL" "$out_file"; then
            echo "  Result: OK"
            n_ok=$((n_ok+1))
        else
            echo "  Result: INCOMPLETE (no 'Final estimate' di output)"
            n_fail=$((n_fail+1))
        fi
    else
        echo "  Result: FAILED"
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  Selesai. OK: $n_ok / ${#SCHEMES[@]} | Failed: $n_fail"
echo "=================================================="
echo ""
echo "Selanjutnya jalankan aggregator:"
echo "  python3 evaluation/aggregate_tabel4_perplexity.py"
