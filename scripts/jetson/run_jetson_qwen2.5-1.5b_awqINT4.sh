#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-1.5B AWQ INT4 (symmetric).
# Sumber: poweredshine/qwen2.5-1.5b-awq-int4-sym (download via ./download_qwen_models.sh).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-1.5b_awqINT4" \
    "${MODEL_PATH:-./results/qwen2.5-1.5b-awq-int4-sym}" \
    "awq"
