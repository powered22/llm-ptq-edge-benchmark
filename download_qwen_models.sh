#!/usr/bin/env bash
# Download Qwen2.5 quantized weights dari HuggingFace Hub (akun poweredshine) ke ./results/
# untuk dibenchmark di Jetson dengan benchmark/benchmark_jetson.py.
#
# Repo-repo Anda PRIVATE → script butuh env var HF_TOKEN (scope Read).
#
# Cara pakai di Jetson:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx     # JANGAN commit token ini!
#   chmod +x download_qwen_models.sh
#   ./download_qwen_models.sh                       # download semua
#   MODELS_FILTER="1.5b" ./download_qwen_models.sh  # hanya yang mengandung "1.5b"
#
# Output: ./results/<repo-basename>/   (cocok dipakai sebagai --model-path)

set -euo pipefail

# ---- Cek prasyarat ----
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: env var HF_TOKEN belum di-set."
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxx   (token Read dari https://huggingface.co/settings/tokens)"
    exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli tidak ditemukan — install dulu:"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

OUT_DIR="${OUT_DIR:-./results}"
MODELS_FILTER="${MODELS_FILTER:-}"   # substring filter, kosong = semua
mkdir -p "$OUT_DIR"

# ---- Daftar repo (sesuai screenshot HuggingFace Anda) ----
REPOS=(
    # ----- Qwen2.5-1.5B (base) -----
    "poweredshine/qwen2.5_1.5b_smooth_w8a8"
    "poweredshine/qwen2.5-1.5b-gptq-w8a16"
    "poweredshine/qwen2.5-1.5b-gptq-int4"
    "poweredshine/qwen2.5-1.5b-rtn-w4a16"
    "poweredshine/qwen2.5-1.5b-awq-int4-sym"

    # ----- Qwen2.5-1.5B-Instruct -----
    "poweredshine/qwen2.5_1.5b_instruct_gptq_w4a16"

    # ----- Qwen2.5-0.5B-Instruct -----
    "poweredshine/qwen2.5_0.5b_instruct_smooth_w8a8"
    "poweredshine/qwen2.5_0.5b_instruct_rtn_w8a16"
    "poweredshine/qwen2.5_0.5b_instruct_rtn_w4a16"
    "poweredshine/qwen2.5_0.5b_instruct_gptq_w4a16"
    "poweredshine/qwen2.5_0.5b_instruct_awq_w4a16"
)

echo "=================================================="
echo "  Output dir : $OUT_DIR"
echo "  Filter     : ${MODELS_FILTER:-<none, download all>}"
echo "  Repos      : ${#REPOS[@]} total"
echo "=================================================="

n_ok=0; n_skip=0; n_fail=0
for repo in "${REPOS[@]}"; do
    if [[ -n "$MODELS_FILTER" && "$repo" != *"$MODELS_FILTER"* ]]; then
        n_skip=$((n_skip+1))
        continue
    fi

    name="${repo##*/}"           # basename, mis. qwen2.5-1.5b-rtn-w4a16
    target="$OUT_DIR/$name"
    echo ""
    echo "[$((n_ok+n_fail+1))/${#REPOS[@]}] $repo"
    echo "   -> $target"

    if huggingface-cli download "$repo" \
            --local-dir "$target" \
            --local-dir-use-symlinks False \
            --token "$HF_TOKEN" \
            >/dev/null; then
        echo "   OK"
        n_ok=$((n_ok+1))
    else
        echo "   FAILED — cek apakah repo private & token punya akses."
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  Selesai. OK: $n_ok | Skipped (filter): $n_skip | Failed: $n_fail"
echo "=================================================="
echo ""
echo "Mapping ke --method di benchmark/benchmark_jetson.py:"
echo "  *-awq-*            -> --method awq"
echo "  *-gptq-*           -> --method gptq"
echo "  *-rtn-*            -> --method gptq  (RTN tersimpan dalam format GPTQ-compatible — verifikasi config.json)"
echo "  *-smooth-* (w8a8)  -> BELUM didukung benchmark_jetson.py (perlu tambah branch loader)"
echo ""
echo "Contoh run setelah download selesai:"
echo "  python benchmark/benchmark_jetson.py \\"
echo "    --model-path $OUT_DIR/qwen2.5-1.5b-awq-int4-sym \\"
echo "    --method awq --input-len 256 --output-len 64 \\"
echo "    --output $OUT_DIR/bench_awq.csv"
