#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-1.5B GPTQ INT4 (W4A16-equivalent).
# Sumber: poweredshine/qwen2.5-1.5b-gptq-int4 (download via ./download_qwen_models.sh).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-1.5b_gptqINT4" \
    "${MODEL_PATH:-./results/qwen2.5-1.5b-gptq-int4}" \
    "gptq"
