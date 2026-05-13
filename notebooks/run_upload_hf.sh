#!/bin/bash

MODELS=(
    "qwen2.5_0.5b_instruct_rtn_w8a16"
    "qwen2.5_0.5b_instruct_rtn_w4a16"
    "qwen2.5_0.5b_instruct_smooth_w8a8"
    "qwen2.5-1.5b-gptq-w8a16"
    "qwen2.5-1.5b-rtn-w4a16"
    "qwen2.5_1.5b_smooth_w8a8"
)

for NAME in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Creating repo: $NAME"
    huggingface-cli repo create "$NAME" --type model

    echo "Uploading: poweredshine/$NAME"
    huggingface-cli upload "poweredshine/$NAME" \
        "./results/$NAME" \
        . \
        --repo-type model

    echo "Done: $NAME"
    echo ""
done

echo "All models processed!"