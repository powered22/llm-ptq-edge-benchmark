#!/usr/bin/env bash
# Benchmark satu file GGUF di Jetson dengan llama-bench + tegrastats.
# Mengisi satu row di Tabel 2.
#
# Usage:
#   ./scripts/jetson/benchmark_gguf_jetson.sh <gguf-file> <label>
#
# Output:
#   ./results/jetson_gguf/<label>/bench.json   (llama-bench JSON output)
#   ./results/jetson_gguf/<label>/tegra.log    (tegrastats raw log)
#   ./results/jetson_gguf/summary.csv          (one row appended)
set -euo pipefail

GGUF_PATH="${1:?usage: $0 <gguf-path> <label>}"
LABEL="${2:?usage: $0 <gguf-path> <label>}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LLAMA_BENCH="$REPO_ROOT/external/llama.cpp/build/bin/llama-bench"
[[ -x "$LLAMA_BENCH" ]] || { echo "ERROR: $LLAMA_BENCH tidak ada (build dulu)"; exit 1; }
[[ -f "$GGUF_PATH" ]]  || { echo "ERROR: $GGUF_PATH tidak ada"; exit 1; }

OUT_DIR="$REPO_ROOT/results/jetson_gguf/$LABEL"
mkdir -p "$OUT_DIR"

# Maximize Jetson clocks untuk variance latency rendah
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks 2>/dev/null || echo "[warn] jetson_clocks gagal"
fi

# Mulai tegrastats di background (200ms interval)
TEGRA_LOG="$OUT_DIR/tegra.log"
rm -f "$TEGRA_LOG"
sudo tegrastats --interval 200 --logfile "$TEGRA_LOG" &
TEGRA_PID=$!
sleep 1   # tegrastats stabilize

# Run llama-bench: prompt 256 + generate 64, 5 reps, all layers ke GPU
BENCH_JSON="$OUT_DIR/bench.json"
echo ""
echo "[$LABEL] llama-bench: -p 256 -n 64 -r 5 -ngl 99"
"$LLAMA_BENCH" \
    -m "$GGUF_PATH" \
    -p 256 -n 64 -r 5 \
    -ngl 99 \
    -o json > "$BENCH_JSON"

# Stop tegrastats
sleep 0.5
sudo kill "$TEGRA_PID" 2>/dev/null || true
wait "$TEGRA_PID" 2>/dev/null || true

# Aggregate ke summary.csv
python3 "$REPO_ROOT/benchmark/aggregate_gguf_row.py" \
    --label "$LABEL" \
    --bench-json "$BENCH_JSON" \
    --tegra-log "$TEGRA_LOG" \
    --csv "$REPO_ROOT/results/jetson_gguf/summary.csv"
