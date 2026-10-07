#!/usr/bin/env bash
# Benchmark one HF (transformers) model on Jetson with tegrastats.
# Fills one row of the HF side of Table 2, using the same aggregation as the GGUF rows.
#
# Usage:
#   ./scripts/jetson/benchmark_hf_jetson.sh <model-path-or-hf-id> <method: fp16|gptq|awq> <label>
#   (gptq = any compressed-tensors checkpoint: gptq, rtn, smoothquant)
#
# Optional env:
#   DROP_CACHES=0     skip dropping the page cache before the run (default 1)
#   DISABLE_SWAP=1    run with swap off (a too-big run then fails instead of swapping); swap is restored after
#
# Output:
#   ./results/jetson_hf/<label>/{bench.json,runtime.json,tegra.log}
#   ./results/jetson_hf/summary.csv   (one row appended)
set -euo pipefail

MODEL_PATH="${1:?usage: $0 <model-path> <fp16|gptq|awq> <label>}"
METHOD="${2:?usage: $0 <model-path> <fp16|gptq|awq> <label>}"
LABEL="${3:?usage: $0 <model-path> <fp16|gptq|awq> <label>}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="$REPO_ROOT/results/jetson_hf/$LABEL"
mkdir -p "$OUT_DIR"

SWAP_WAS_OFF_BY_US=0
cleanup() {
    if [[ "$SWAP_WAS_OFF_BY_US" == "1" ]]; then sudo swapon -a || true; fi
    sudo pkill -x tegrastats 2>/dev/null || true
}
trap cleanup EXIT

if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks 2>/dev/null || echo "[warn] jetson_clocks failed"
fi

if [[ "${DISABLE_SWAP:-0}" == "1" ]]; then
    sudo swapoff -a
    SWAP_WAS_OFF_BY_US=1
fi
if [[ "${DROP_CACHES:-1}" == "1" ]]; then
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
fi

echo "[$LABEL] memory before run:"
free -m

# tegrastats is started inside the python runner so it only covers the timed reps.
python3 "$REPO_ROOT/benchmark/benchmark_hf_jetson.py" \
    --model-path "$MODEL_PATH" \
    --method "$METHOD" \
    --bench-json "$OUT_DIR/bench.json" \
    --runtime-json "$OUT_DIR/runtime.json" \
    --tegra-log "$OUT_DIR/tegra.log"

python3 "$REPO_ROOT/benchmark/aggregate_gguf_row.py" \
    --label "$LABEL" \
    --bench-json "$OUT_DIR/bench.json" \
    --tegra-log "$OUT_DIR/tegra.log" \
    --runtime-json "$OUT_DIR/runtime.json" \
    --csv "$REPO_ROOT/results/jetson_hf/summary.csv"
