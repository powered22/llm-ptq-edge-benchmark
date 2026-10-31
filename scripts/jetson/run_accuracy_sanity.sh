#!/usr/bin/env bash
# Jetson accuracy sanity check: 200 systematically sampled items on arc_easy, hellaswag, winogrande,
# for Qwen2.5-0.5B-Instruct: HF FP16, HF GPTQ W4A16, GGUF Q8_0, GGUF Q4_0. 0-shot, no chat template.
# Run in the text console, inside tmux. Log: results/jetson_acc/log.txt
# Output: results/jetson_acc/<label>/ (lm-eval json + samples_*.jsonl)
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"
O="$REPO_ROOT/results/jetson_acc"; mkdir -p "$O"
exec > >(tee -a "$O/log.txt") 2>&1
IDS="$REPO_ROOT/analysis/sanity_ids_200.json"
TASKS="arc_easy,hellaswag,winogrande"
echo "== accuracy sanity $(date) =="
pgrep -x -f "gnome-shell|Xorg" >/dev/null 2>&1 && { echo "ERROR: desktop is running, use the text console"; exit 1; }
R="$REPO_ROOT/results"
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
run_hf   hf_fp16        Qwen/Qwen2.5-0.5B-Instruct
run_hf   hf_gptqW4A16   "$R/qwen2.5_0.5b_instruct_gptq_w4a16"
run_gguf gguf_Q8_0      "$R/qwen2.5_0.5b_instruct_gguf/qwen2.5_0.5b_instruct-Q8_0.gguf"
run_gguf gguf_Q4_0      "$R/qwen2.5_0.5b_instruct_gguf/qwen2.5_0.5b_instruct-Q4_0.gguf"
echo "== done: OK $ok  FAILED $fail =="
