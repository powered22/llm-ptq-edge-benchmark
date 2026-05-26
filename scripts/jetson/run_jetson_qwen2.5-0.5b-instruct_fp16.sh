#!/usr/bin/env bash
# Jetson benchmark: Qwen2.5-0.5B-Instruct FP16 baseline.
# Model di-download otomatis dari HuggingFace official (Qwen/Qwen2.5-0.5B-Instruct).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/jetson/_run.sh"

run_jetson_benchmark \
    "qwen2.5-0.5b-instruct_fp16" \
    "${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}" \
    "fp16"
