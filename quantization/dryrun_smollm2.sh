#!/usr/bin/env bash
# Dry run: can the existing pipeline quantize SmolLM2-Instruct (360M and 1.7B) with our 5 HF methods + 5 GGUF schemes?
# Run on the x86 machine (env llm_experiments for HF, gguf_eval is not needed here).
# Reports OK/FAIL per step and smoke-tests every HF output by loading it and generating a few tokens.
# Output: ./results/smollm2_<size>_instruct_*  and  ./results/smollm2_<size>_instruct_gguf/
# Log:    ./results/dryrun_smollm2_log.txt
# Usage:  bash quantization/dryrun_smollm2.sh [360M|1.7B ...]     (default: both)
#   HF steps only (env llm_experiments):  SKIP_GGUF=1 bash quantization/dryrun_smollm2.sh
#   GGUF only     (env gguf_eval):        SKIP_HF=1   bash quantization/dryrun_smollm2.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p results
exec > >(tee -a results/dryrun_smollm2_log.txt) 2>&1

SIZES=("$@"); [[ ${#SIZES[@]} -eq 0 ]] && SIZES=(360M 1.7B)
NUM_SAMPLES=512; MAX_SEQ=512
declare -a TALLY=()

step() {  # step <label> <cmd...>
    local label="$1"; shift
    echo ""; echo "=========== $label ==========="
    if "$@"; then TALLY+=("OK    $label"); else TALLY+=("FAIL  $label"); fi
}

smoke() {  # smoke <dir>: load the checkpoint and generate 16 tokens
    python - "$1" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu")
ids = tok("The capital of France is", return_tensors="pt").to(m.device)
out = m.generate(**ids, max_new_tokens=16, do_sample=False)
print("SMOKE:", tok.decode(out[0], skip_special_tokens=True))
PY
}

for SZ in "${SIZES[@]}"; do
    MODEL="HuggingFaceTB/SmolLM2-${SZ}-Instruct"
    TAG="smollm2_$(echo "$SZ" | tr 'A-Z' 'a-z')_instruct"
    OUT="./results/$TAG"
    # SmolLM2-360M has hidden size 960, not divisible by the preset group size 128 -> use 64 for the group-quantized methods
    GS=(); [[ "$SZ" == "360M" ]] && GS=(--group-size 64)
    echo ""; echo "################ $MODEL  ($(date)) ################"

    if [[ "${SKIP_HF:-0}" != "1" ]]; then
    step "$TAG awq_w4a16"   python quantization/quantize_awq_v2.py   --model "$MODEL" --output "${OUT}_awq_w4a16"  --num-calibration-samples $NUM_SAMPLES --max-seq-length $MAX_SEQ "${GS[@]}"
    step "$TAG gptq_w4a16"  python quantization/quantize_gptq_v2.py  --model "$MODEL" --output "${OUT}_gptq_w4a16" --num-samples $NUM_SAMPLES --scheme W4A16 "${GS[@]}"
    step "$TAG rtn_w4a16"   python quantization/quantize_rtn.py      --model "$MODEL" --output "${OUT}_rtn_w4a16"  --scheme W4A16 "${GS[@]}"
    step "$TAG rtn_w8a16"   python quantization/quantize_rtn.py      --model "$MODEL" --output "${OUT}_rtn_w8a16"  --scheme W8A16 "${GS[@]}"
    step "$TAG smooth_w8a8" python quantization/quantize_smoothquant.py --model "$MODEL" --output "${OUT}_smooth_w8a8"

    for m in awq_w4a16 gptq_w4a16 rtn_w4a16 rtn_w8a16 smooth_w8a8; do
        [[ -d "${OUT}_${m}" ]] && step "$TAG smoke_$m" smoke "${OUT}_${m}"
    done
    fi

    if [[ "${SKIP_GGUF:-0}" != "1" ]]; then
    LQ="$REPO_ROOT/external/llama.cpp/build/bin/llama-quantize"
    step "$TAG gguf (F16,Q8_0,Q5_K_M,Q4_K_M,Q4_0)" python3 quantization/convert_to_gguf.py \
        --model "$MODEL" --output-name "$TAG" --schemes Q4_K_M Q5_K_M Q8_0 Q4_0 \
        --base-outtype f16 --keep-base --llama-quantize "$LQ"
    fi
done

echo ""; echo "=========== SUMMARY ==========="
printf '%s\n' "${TALLY[@]}"
echo "OK: $(printf '%s\n' "${TALLY[@]}" | grep -c '^OK') | FAIL: $(printf '%s\n' "${TALLY[@]}" | grep -c '^FAIL')"
echo "Log: results/dryrun_smollm2_log.txt"
