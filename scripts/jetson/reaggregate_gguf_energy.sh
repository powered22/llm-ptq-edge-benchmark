#!/usr/bin/env bash
# Recompute the GGUF energy numbers from the saved bench.json + tegra.log of every scheme,
# averaging power only over the active inference window (no benchmark re-run needed).
#
# Usage:
#   ./scripts/jetson/reaggregate_gguf_energy.sh [gpu-pct]     (default 20)
#
# Output: ./results/jetson_gguf/summary_trimmed.csv   (summary.csv is not touched)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
THRESHOLD="${1:-20}"
OUT_CSV="$REPO_ROOT/results/jetson_gguf/summary_trimmed.csv"
rm -f "$OUT_CSV"

for scheme in F16 Q8_0 Q5_K_M Q4_K_M Q4_0; do
    label="qwen2.5-0.5b-instruct_${scheme}"
    dir="$REPO_ROOT/results/jetson_gguf/$label"
    if [[ ! -f "$dir/bench.json" || ! -f "$dir/tegra.log" ]]; then
        echo "[skip] $label: missing bench.json or tegra.log"
        continue
    fi
    python3 "$REPO_ROOT/benchmark/aggregate_gguf_row.py" \
        --label "$label" \
        --bench-json "$dir/bench.json" \
        --tegra-log "$dir/tegra.log" \
        --active-gpu-pct "$THRESHOLD" \
        --csv "$OUT_CSV" | grep -E "Label|Avg power|Energy"
done
echo ""
echo "Written: $OUT_CSV"
