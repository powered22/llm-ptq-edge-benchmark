#!/usr/bin/env bash
# Benchmark Qwen di Jetson dengan method FP16 (baseline, tanpa kuantisasi).
#
# Cara pakai:
#   chmod +x run_jetson_fp16_qwen.sh
#   ./run_jetson_fp16_qwen.sh
#
# Atau override modelnya:
#   MODEL=Qwen/Qwen2.5-3B ./run_jetson_fp16_qwen.sh

set -euo pipefail

# ---- Konfigurasi (boleh di-override lewat env var) ----
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"     # bisa HF Hub ID atau path lokal
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
OUTPUT_CSV="${OUTPUT_CSV:-./results/benchmark_jetson_qwen_fp16.csv}"

# ---- Pindah ke direktori repo supaya `from benchmark.utils import ...` & `from models.load_fp16 import ...` ketemu ----
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- (Direkomendasikan) maximize clocks Jetson untuk mengurangi varian latency ----
# Butuh sudo; abaikan kalau gagal (mis. user bukan sudoer atau bukan di Jetson).
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks || echo "[warn] jetson_clocks gagal — lanjut tanpa max clocks."
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "=================================================="
echo "  Model      : $MODEL"
echo "  Method     : fp16"
echo "  Input len  : $INPUT_LEN  tokens"
echo "  Output len : $OUTPUT_LEN tokens"
echo "  Output CSV : $OUTPUT_CSV"
echo "=================================================="

python3 benchmark/benchmark_jetson.py \
    --model-path "$MODEL" \
    --method fp16 \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --output "$OUTPUT_CSV"
