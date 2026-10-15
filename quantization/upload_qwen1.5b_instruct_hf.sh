#!/usr/bin/env bash
# Upload the five Qwen2.5-1.5B-Instruct quantized checkpoints (x86 server) to the
# HuggingFace Hub (private repos, account poweredshine) so the Jetson can download them.
#
# Prerequisite: huggingface-cli login (token with Write scope).
# Re-uploading an existing repo overwrites its files with the new checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
    qwen2.5_1.5b_instruct_awq_w4a16
    qwen2.5_1.5b_instruct_gptq_w4a16
    qwen2.5_1.5b_instruct_rtn_w4a16
    qwen2.5_1.5b_instruct_rtn_w8a16
    qwen2.5_1.5b_instruct_smooth_w8a8
)

for NAME in "${MODELS[@]}"; do
    echo "=========================================="
    [[ -d "./results/$NAME" ]] || { echo "[skip] ./results/$NAME not found"; continue; }
    echo "Uploading poweredshine/$NAME"
    huggingface-cli repo create "$NAME" --type model --private -y 2>/dev/null || true
    huggingface-cli upload "poweredshine/$NAME" "./results/$NAME" . --repo-type model
done
echo "Done."
