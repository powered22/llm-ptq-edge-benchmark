#!/usr/bin/env bash
# Tabel 1: Akurasi semua metode kuantisasi untuk Qwen2.5-0.5B-Instruct.
# Engine: HF transformers + lm-evaluation-harness, di Linux GPU.
# Preset 'full' (7 tasks): arc_easy, arc_challenge, hellaswag, winogrande,
#                          mmlu, truthfulqa_mc, gsm8k.
#
# Estimasi: 30-60 menit per metode × 6 metode = 3-6 jam total di Linux GPU.
# (Lebih lama di MMLU karena banyak subtask.)
#
# Prasyarat:
#   1. pip install 'lm-eval>=0.4.2' compressed-tensors accelerate
#   2. export HF_TOKEN=hf_xxxxxxxx   (untuk model private poweredshine/*)
#
# Output: ./results/eval/qwen2.5-0.5b-instruct_<method>.json  (6 file)
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: env var HF_TOKEN belum di-set."
    echo "  Generate token Read di https://huggingface.co/settings/tokens"
    echo "  Lalu: export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

TASKS="${TASKS:-full}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-./results/eval}"
mkdir -p "$OUT_DIR"

# Format: "label:method:model_path"
# - label : nama untuk file output
# - method: argumen --method untuk run_lm_harness.py (label utk model loading)
# - model_path: HF Hub ID atau path lokal
METHODS=(
    "fp16:fp16:Qwen/Qwen2.5-0.5B-Instruct"
    "awqW4A16:awq:poweredshine/qwen2.5_0.5b_instruct_awq_w4a16"
    "gptqW4A16:gptq:poweredshine/qwen2.5_0.5b_instruct_gptq_w4a16"
    "rtnW4A16:rtn:poweredshine/qwen2.5_0.5b_instruct_rtn_w4a16"
    "rtnW8A16:rtn:poweredshine/qwen2.5_0.5b_instruct_rtn_w8a16"
    "smoothW8A8:smoothquant:poweredshine/qwen2.5_0.5b_instruct_smooth_w8a8"
)

n_ok=0; n_fail=0; idx=0
for spec in "${METHODS[@]}"; do
    idx=$((idx+1))
    label="${spec%%:*}"
    rest="${spec#*:}"
    method="${rest%%:*}"
    model_path="${rest#*:}"

    OUT="$OUT_DIR/qwen2.5-0.5b-instruct_${label}.json"

    echo ""
    echo "##################################################"
    echo "  [$idx/${#METHODS[@]}] $label"
    echo "    Method  : $method"
    echo "    Model   : $model_path"
    echo "    Tasks   : $TASKS"
    echo "    Output  : $OUT"
    echo "##################################################"

    if [[ -f "$OUT" ]] && [[ -z "${FORCE:-}" ]]; then
        echo "[skip] $OUT sudah ada. Set FORCE=1 untuk re-run."
        n_ok=$((n_ok+1))
        continue
    fi

    if python3 evaluation/run_lm_harness.py \
        --model-path "$model_path" \
        --method "$method" \
        --tasks "$TASKS" \
        --batch-size "$BATCH_SIZE" \
        --apply-chat-template \
        --output "$OUT"; then
        n_ok=$((n_ok+1))
    else
        echo "[error] gagal eval $label"
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  Tabel 1 selesai. OK: $n_ok | Failed: $n_fail"
echo "=================================================="
echo ""
echo "File hasil di $OUT_DIR/:"
ls -lh "$OUT_DIR/"qwen2.5-0.5b-instruct_*.json 2>/dev/null || true
echo ""
echo "Selanjutnya: aggregate 6 JSON ini ke satu CSV Tabel 1."
echo "Mau saya buatkan script aggregator-nya setelah ini? Beritahu saya."
