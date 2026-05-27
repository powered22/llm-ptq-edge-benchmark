#!/usr/bin/env bash
# Benchmark semua skema GGUF Qwen2.5-0.5B-Instruct di Jetson.
# Mengisi 5 row Tabel 2: F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0.
#
# Prasyarat: convert_qwen0.5b_to_gguf.sh sudah dijalankan.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF_DIR="$REPO_ROOT/results/qwen2.5_0.5b_instruct_gguf"
BENCH="$REPO_ROOT/scripts/jetson/benchmark_gguf_jetson.sh"

SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

n_ok=0; n_skip=0; n_fail=0
for scheme in "${SCHEMES[@]}"; do
    GGUF_FILE="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"
    LABEL="qwen2.5-0.5b-instruct_${scheme}"

    if [[ ! -f "$GGUF_FILE" ]]; then
        echo "[skip] $GGUF_FILE tidak ada"
        n_skip=$((n_skip+1))
        continue
    fi

    echo ""
    echo "##################################################"
    echo "  Benchmark [$((n_ok+n_fail+1))/${#SCHEMES[@]}]: $LABEL"
    echo "##################################################"

    if "$BENCH" "$GGUF_FILE" "$LABEL"; then
        n_ok=$((n_ok+1))
    else
        echo "[error] gagal benchmark $LABEL"
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  OK: $n_ok | Skip: $n_skip | Fail: $n_fail"
echo "=================================================="
echo ""
echo "Summary CSV (Tabel 2 untuk Qwen2.5-0.5B-Instruct):"
echo "  $REPO_ROOT/results/jetson_gguf/summary.csv"
echo ""
column -t -s, "$REPO_ROOT/results/jetson_gguf/summary.csv" 2>/dev/null || \
    cat "$REPO_ROOT/results/jetson_gguf/summary.csv"
