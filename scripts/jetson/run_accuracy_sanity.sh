#!/usr/bin/env bash
# Jetson accuracy sanity check: 200 systematically sampled items on arc_easy, hellaswag, winogrande,
# for one model: HF FP16, HF GPTQ W4A16, GGUF Q8_0, GGUF Q4_0. 0-shot, no chat template.
# usage: bash scripts/jetson/run_accuracy_sanity.sh [qwen0.5b|smollm2-360m]   (default qwen0.5b)
# Run in the text console, inside tmux. Log: <out dir>/log.txt
# Output: results/jetson_acc/<label>/ for qwen0.5b, results/jetson_acc_smollm2_360m/<label>/ for smollm2-360m
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"
MODEL="${1:-qwen0.5b}"
R="$REPO_ROOT/results"
case "$MODEL" in
  qwen0.5b)     O="$R/jetson_acc"; FP16_ID="Qwen/Qwen2.5-0.5B-Instruct"
                GPTQ="$R/qwen2.5_0.5b_instruct_gptq_w4a16"; GD="$R/qwen2.5_0.5b_instruct_gguf"; GP="qwen2.5_0.5b_instruct" ;;
  smollm2-360m) O="$R/jetson_acc_smollm2_360m"; FP16_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
                GPTQ="$R/smollm2_360m_instruct_gptq_w4a16"; GD="$R/smollm2_360m_instruct_gguf"; GP="smollm2_360m_instruct" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
mkdir -p "$O"
exec > >(tee -a "$O/log.txt") 2>&1
IDS="$REPO_ROOT/analysis/sanity_ids_200.json"
TASKS="arc_easy,hellaswag,winogrande"
echo "== accuracy sanity $MODEL $(date) =="
pgrep -x -f "gnome-shell|Xorg" >/dev/null 2>&1 && { echo "ERROR: desktop is running, use the text console"; exit 1; }
ok=0; fail=0
prep() { sudo swapoff -a && sudo swapon -a; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
run_hf() {  # label path
  echo "--- $1 ---"; prep
  python3 -m lm_eval --model hf --model_args "pretrained=$2,trust_remote_code=True,dtype=float16" \
    --tasks "$TASKS" --samples "$IDS" --num_fewshot 0 --batch_size 1 --log_samples --output_path "$O/$1" \
    && ok=$((ok+1)) || fail=$((fail+1))
}
run_gguf() {
  echo "--- $1 ---"; prep
  [[ -f "$2" ]] || { echo "missing $2"; fail=$((fail+1)); return; }
  python3 scripts/run_lm_eval_gguf.py --model llama-cpp-direct \
    --model_args "pretrained=$2,n_ctx=2048,n_gpu_layers=99" \
    --tasks "$TASKS" --samples "$IDS" --num_fewshot 0 --batch_size 1 --log_samples --output_path "$O/$1" \
    && ok=$((ok+1)) || fail=$((fail+1))
}
run_hf   hf_fp16        "$FP16_ID"
run_hf   hf_gptqW4A16   "$GPTQ"
run_gguf gguf_Q8_0      "$GD/${GP}-Q8_0.gguf"
run_gguf gguf_Q4_0      "$GD/${GP}-Q4_0.gguf"
echo "== done: OK $ok  FAILED $fail =="
