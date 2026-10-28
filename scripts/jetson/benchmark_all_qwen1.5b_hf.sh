#!/usr/bin/env bash
# Benchmark the six HF configs of Qwen2.5-1.5B-Instruct on the Jetson (HF side of Table 2).
# FP16 first (largest memory use), then the five quantized checkpoints.
# A failed run (for example the memory guard) does not stop the others.
#
# Prerequisite: ./download_qwen_models.sh with MODELS_FILTER="1.5b_instruct" has been run.
# Output: results/jetson_hf/<label>/ and results/jetson_hf/summary.csv
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# libcudss for the Jetson torch build (see project notes)
export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

BENCH="$REPO_ROOT/scripts/jetson/benchmark_hf_jetson.sh"
R="$REPO_ROOT/results"

# label : method : model path or hub id
CONFIGS=(
    "qwen2.5-1.5b-instruct_hf_fp16:fp16:Qwen/Qwen2.5-1.5B-Instruct"
    "qwen2.5-1.5b-instruct_hf_gptqW4A16:gptq:$R/qwen2.5_1.5b_instruct_gptq_w4a16"
    "qwen2.5-1.5b-instruct_hf_rtnW4A16:gptq:$R/qwen2.5_1.5b_instruct_rtn_w4a16"
    "qwen2.5-1.5b-instruct_hf_rtnW8A16:gptq:$R/qwen2.5_1.5b_instruct_rtn_w8a16"
    "qwen2.5-1.5b-instruct_hf_awqW4A16:gptq:$R/qwen2.5_1.5b_instruct_awq_w4a16"
    "qwen2.5-1.5b-instruct_hf_smoothW8A8:gptq:$R/qwen2.5_1.5b_instruct_smooth_w8a8"
)

n_ok=0; n_fail=0
for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r label method path <<<"$cfg"
    echo ""
    echo "##################################################"
    echo "  HF benchmark: $label"
    echo "##################################################"
    if "$BENCH" "$path" "$method" "$label"; then
        n_ok=$((n_ok+1))
    else
        echo "[error] $label failed"
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  OK: $n_ok | Fail: $n_fail"
echo "=================================================="
column -t -s, "$R/jetson_hf/summary.csv" 2>/dev/null || cat "$R/jetson_hf/summary.csv"
