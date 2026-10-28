#!/usr/bin/env bash
# Upload the SmolLM2-360M/1.7B-Instruct quantized checkpoints (x86) to the HuggingFace Hub (private, poweredshine)
# so the Jetson can download them. Prerequisite: hf auth login (Write scope).
set -euo pipefail
cd "$(dirname "$0")/.."
for SZ in 360m 1.7b; do
    for M in awq_w4a16 gptq_w4a16 rtn_w4a16 rtn_w8a16 smooth_w8a8; do
        NAME="smollm2_${SZ}_instruct_${M}"
        echo "=========================================="
        [[ -d "./results/$NAME" ]] || { echo "[skip] ./results/$NAME not found"; continue; }
        echo "Uploading poweredshine/$NAME"
        hf repo create "poweredshine/$NAME" --repo-type model --private --exist-ok
        hf upload "poweredshine/$NAME" "./results/$NAME" . --repo-type model
    done
done
echo "Done."
