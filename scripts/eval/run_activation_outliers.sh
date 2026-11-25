#!/usr/bin/env bash
# Activation-outlier check for the four FP16 models (explains SmoothQuant W8A8 losses).
# Run from the repo root on the x86 GPU machine, in the same env used for quantization.
# Output: results/analysis/outliers/{summary_alpha0.5.csv, layers_<model>.csv}
set -euo pipefail
python analysis/activation_outliers.py \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
           HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --n-prompts 32 --alpha 0.5 --out-dir results/analysis/outliers "$@"
