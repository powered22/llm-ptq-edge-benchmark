#!/usr/bin/env bash
# Group-size diagnostic for Section 4.2 (Qwen2.5-0.5B-Instruct, HF RTN W4A16).
#
# Question: does sharing one scale per 32 weights (like GGUF Q4_0) close the
# quality gap between RTN W4A16 (default group 128) and Q4_0?
#
# Runs, in order:
#   1. rtnW4A16g32        : RTN W4A16 with group size 32 (the test)
#   2. rtnW4A16g128local  : RTN W4A16 with the default group 128, re-quantized
#                           in today's environment (control; skip with CONTROL=0)
# Each: quantize (skipped if the checkpoint exists), then the same 3-part
# evaluation as Table 1 (eval_qwen0.5b_all_methods.sh):
#   A. 6 likelihood tasks, no chat template
#   B. gsm8k, chat template
#   C. ifeval, chat template
# Results go to results/eval/tabel1_hf/, next to the Table 1 runs.
# Completed parts are skipped on re-run (FORCE=1 to redo).
#
# Usage (from the repo root, inside tmux):
#   bash scripts/eval/eval_qwen0.5b_rtn_groupsize.sh 2>&1 | tee results/eval/rtn_groupsize.log
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PREFIX="qwen2.5-0.5b-instruct"
LIKELIHOOD_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc2"
GEN_TASKS="gsm8k"
IFEVAL_TASKS="ifeval"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-./results/eval/tabel1_hf}"
mkdir -p "$OUT_DIR"

# Format: "label:group_size:checkpoint_dir"
RUNS=("rtnW4A16g32:32:./results/qwen2.5_0.5b_instruct_rtn_w4a16_g32")
if [[ "${CONTROL:-1}" == "1" ]]; then
    RUNS+=("rtnW4A16g128local:128:./results/qwen2.5_0.5b_instruct_rtn_w4a16_g128_local")
fi

is_valid_json() {
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null
}

run_part() {
    # run_part <model_path> <tasks> <output> <use_template:0|1>
    local model_path="$1" tasks="$2" out="$3" use_template="$4"
    local prefix="${out%.json}"
    shopt -s nullglob
    local matches=( "$out" "${prefix}_"*.json )
    shopt -u nullglob
    if [[ -z "${FORCE:-}" ]]; then
        for f in "${matches[@]}"; do
            [[ -f "$f" ]] && is_valid_json "$f" && { echo "    [skip] $f exists"; return 0; }
        done
    fi
    local args=(--model-path "$model_path" --method rtn --tasks "$tasks"
                --batch-size "$BATCH_SIZE" --output "$out")
    [[ "$use_template" == "1" ]] && args+=(--apply-chat-template)
    python3 evaluation/run_lm_harness.py "${args[@]}"
}

for spec in "${RUNS[@]}"; do
    label="${spec%%:*}"; rest="${spec#*:}"
    gs="${rest%%:*}"; ckpt="${rest#*:}"
    echo ""
    echo "##################################################"
    echo "  $label  (group size $gs)  $(date '+%F %T')"
    echo "##################################################"

    if [[ -f "$ckpt/config.json" ]]; then
        echo "  [skip] checkpoint $ckpt exists"
    else
        python3 quantization/quantize_rtn.py --model "$MODEL" --output "$ckpt" \
            --scheme W4A16 --group-size "$gs"
    fi
    # Show the stored group size, so the log proves what was evaluated.
    python3 -c "import json,sys; q=json.load(open(sys.argv[1]+'/config.json'))['quantization_config']; print('  stored group_size:', [g['weights']['group_size'] for g in q['config_groups'].values()])" "$ckpt"

    echo "  Part A: likelihood tasks (no chat template)"
    run_part "$ckpt" "$LIKELIHOOD_TASKS" "$OUT_DIR/${PREFIX}_${label}_likelihood.json" 0
    echo "  Part B: gsm8k (chat template)"
    run_part "$ckpt" "$GEN_TASKS" "$OUT_DIR/${PREFIX}_${label}_gsm8k.json" 1
    echo "  Part C: ifeval (chat template)"
    run_part "$ckpt" "$IFEVAL_TASKS" "$OUT_DIR/${PREFIX}_${label}_ifeval.json" 1
done

echo ""
echo "All runs done $(date '+%F %T'). Aggregating:"
python3 evaluation/aggregate_tabel1.py --eval-dir "$OUT_DIR" \
    --output "$OUT_DIR/tabel1_qwen2.5-0.5b-instruct_groupsize.csv"
