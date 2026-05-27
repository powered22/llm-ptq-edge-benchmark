#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-1.5B RTN W4A16.
# Sumber: poweredshine/qwen2.5-1.5b-rtn-w4a16 (download via ./download_qwen_models.sh).
# CATATAN: pakai --method gptq dengan asumsi RTN disimpan dalam format GPTQ-compatible
#          (compressed-tensors / autogptq). Kalau load gagal, cek config.json repo Anda.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-1.5b_rtnW4A16" \
    "${MODEL_PATH:-./results/qwen2.5-1.5b-rtn-w4a16}" \
    "gptq"
