"""Wrapper for lm-evaluation-harness (EleutherAI).

Runs zero-shot or few-shot evaluation on standard benchmarks:
ARC, HellaSwag, MMLU, WinoGrande, TruthfulQA, GSM8K.

Usage:
    python evaluation/run_lm_harness.py \
        --model-path ./results/qwen1.5b-awq \
        --method awq \
        --tasks standard \
        --output ./results/harness_awq.json

Requires: pip install lm-eval>=0.4.2
"""
import argparse
import subprocess
import sys
import json
import os

TASK_PRESETS = {
    "quick":    "arc_easy,hellaswag",
    "standard": "arc_easy,arc_challenge,hellaswag,winogrande,mmlu",
    "full":     "arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc,gsm8k",
}

# BitsAndBytes needs special model_args
BNB_ARGS = {
    "bnb4": "load_in_4bit=True",
    "bnb8": "load_in_8bit=True",
}


def run_harness(model_path: str, tasks: str, method: str,
                batch_size: int = 8, num_fewshot: int = 0,
                output_path: str = None):
    resolved_tasks = TASK_PRESETS.get(tasks, tasks)
    model_args = f"pretrained={model_path},trust_remote_code=True"

    if method in BNB_ARGS:
        model_args += f",{BNB_ARGS[method]}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", resolved_tasks,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--log_samples",
    ]
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cmd += ["--output_path", output_path]

    print(f"Running lm_eval harness:")
    print(f"  Tasks  : {resolved_tasks}")
    print(f"  Method : {method}")
    print(f"  Output : {output_path or 'stdout'}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lm-eval harness wrapper")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default="fp16",
                        choices=["fp16", "awq", "gptq", "bnb4", "bnb8"])
    parser.add_argument("--tasks", default="standard",
                        help="Preset name (quick/standard/full) or comma-separated task names")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results (optional)")
    args = parser.parse_args()
    run_harness(args.model_path, args.tasks, args.method,
                args.batch_size, args.num_fewshot, args.output)
