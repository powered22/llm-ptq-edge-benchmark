#!/usr/bin/env bash
# Konversi Qwen2.5-0.5B-Instruct ke 5 skema GGUF untuk Tabel 2:
#   F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0
#
# Output: ./results/qwen2.5_0.5b_instruct_gguf/qwen2.5_0.5b_instruct-<SCHEME>.gguf
#
# Prasyarat:
#   - llama.cpp sudah di-build di external/llama.cpp/build/ (Fase 2 selesai)
#   - Model Qwen sudah ada di HF cache (sudah di-download dari run FP16 sebelumnya)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Pastikan Python deps untuk converter ada
if ! python3 -c "import gguf" 2>/dev/null; then
    echo "[setup] Install dependencies untuk convert_hf_to_gguf.py..."
    pip install -r external/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
fi

# Pastikan llama-quantize ada
LLAMA_QUANTIZE="$REPO_ROOT/external/llama.cpp/build/bin/llama-quantize"
if [[ ! -x "$LLAMA_QUANTIZE" ]]; then
    echo "ERROR: $LLAMA_QUANTIZE tidak ditemukan."
    echo "  Selesaikan Fase 2 (build llama.cpp) dulu."
    exit 1
fi

python3 quantization/convert_to_gguf.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output-name qwen2.5_0.5b_instruct \
    --schemes Q4_K_M Q5_K_M Q8_0 Q4_0 \
    --base-outtype f16 \
    --keep-base \
    --llama-quantize "$LLAMA_QUANTIZE"

echo ""
echo "Verifikasi file GGUF yang dibuat:"
ls -lh ./results/qwen2.5_0.5b_instruct_gguf/
