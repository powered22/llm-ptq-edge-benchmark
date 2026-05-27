#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-0.5B-Instruct RTN W4A16.
# Sumber: poweredshine/qwen2.5_0.5b_instruct_rtn_w4a16 (download via ./download_qwen_models.sh).
# CATATAN: pakai --method gptq dengan asumsi RTN disimpan dalam format GPTQ-compatible.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-0.5b-instruct_rtnW4A16" \
    "${MODEL_PATH:-./results/qwen2.5_0.5b_instruct_rtn_w4a16}" \
    "gptq"
