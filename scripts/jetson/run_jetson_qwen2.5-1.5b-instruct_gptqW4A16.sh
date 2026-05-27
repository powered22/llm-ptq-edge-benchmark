#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-1.5B-Instruct GPTQ W4A16.
# Sumber: poweredshine/qwen2.5_1.5b_instruct_gptq_w4a16 (download via ./download_qwen_models.sh).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-1.5b-instruct_gptqW4A16" \
    "${MODEL_PATH:-./results/qwen2.5_1.5b_instruct_gptq_w4a16}" \
    "gptq"
