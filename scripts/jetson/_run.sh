# Shared runner untuk semua script run_jetson_*.sh di direktori ini.
# JANGAN dijalankan langsung — di-source oleh script per-model+method.
#
# Fungsi: run_jetson_benchmark <label> <model_path> <method>
#
# Variabel yang bisa di-override lewat env var saat memanggil script:
#   INPUT_LEN   (default 256)
#   OUTPUT_LEN  (default 64)
#   OUTPUT_CSV  (default ./results/bench_jetson_<label>.csv)
#   MODEL_PATH  (override path/HF-ID model — berguna kalau hasil download
#                disimpan di direktori berbeda)

run_jetson_benchmark() {
    local label="$1"
    local model_path="$2"
    local method="$3"

    local input_len="${INPUT_LEN:-256}"
    local output_len="${OUTPUT_LEN:-64}"
    local output_csv="${OUTPUT_CSV:-./results/bench_jetson_${label}.csv}"

    # Pindah ke repo root supaya import `benchmark.utils` & `models.*` ketemu.
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

    # Validasi: kalau bukan HF-Hub ID (Qwen/...), folder lokal harus ada.
    if [[ "$model_path" != Qwen/* && ! -d "$model_path" ]]; then
        echo "ERROR: '$model_path' tidak ditemukan."
        echo "  Jalankan dulu: ./download_qwen_models.sh"
        echo "  Atau override: MODEL_PATH=/path/lain $0"
        return 1
    fi

    # Maximize Jetson clocks untuk variance latency yang lebih rendah.
    if command -v jetson_clocks >/dev/null 2>&1; then
        sudo jetson_clocks 2>/dev/null || echo "[warn] jetson_clocks gagal (skip)."
    fi

    mkdir -p "$(dirname "$output_csv")"

    echo "=================================================="
    echo "  Run        : $label"
    echo "  Model path : $model_path"
    echo "  Method     : $method"
    echo "  Input/Out  : $input_len / $output_len tokens"
    echo "  Output CSV : $output_csv"
    echo "=================================================="

    python3 benchmark/benchmark_jetson.py \
        --model-path "$model_path" \
        --method "$method" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --output "$output_csv"
}
