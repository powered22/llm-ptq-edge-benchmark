#!/usr/bin/env bash
# Convert SmolLM2-360M-Instruct and SmolLM2-1.7B-Instruct to the five GGUF schemes (F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0).
# Run on the Jetson (the Jetson-side GGUF files must be created or copied there).
# Output: ./results/smollm2_<size>_instruct_gguf/smollm2_<size>_instruct-<SCHEME>.gguf
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import gguf" 2>/dev/null; then
    echo "[setup] installing converter dependencies..."
    pip install -r external/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
fi

LLAMA_QUANTIZE="$REPO_ROOT/external/llama.cpp/build/bin/llama-quantize"
[[ -x "$LLAMA_QUANTIZE" ]] || { echo "ERROR: $LLAMA_QUANTIZE not found (build llama.cpp first)."; exit 1; }

for SZ in 360M 1.7B; do
    LOW="$(echo "$SZ" | tr A-Z a-z)"
    python3 quantization/convert_to_gguf.py \
        --model "HuggingFaceTB/SmolLM2-${SZ}-Instruct" \
        --output-name "smollm2_${LOW}_instruct" \
        --schemes Q4_K_M Q5_K_M Q8_0 Q4_0 \
        --base-outtype f16 \
        --keep-base \
        --llama-quantize "$LLAMA_QUANTIZE"
    ls -lh "./results/smollm2_${LOW}_instruct_gguf/"
done
