"""Compute perplexity on WikiText-2 or PTB for any supported quantization method.

Usage:
    python evaluation/run_perplexity.py \
        --model-path ./results/qwen1.5b-awq \
        --method awq \
        --dataset wikitext2
"""
import argparse
import math
import json
import os
import time
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


DATASET_MAP = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "ptb":       ("ptb_text_only", "penn_treebank"),
}


def compute_perplexity(model, tokenizer, dataset_name: str = "wikitext2",
                       stride: int = 512, max_length: int = 1024) -> float:
    ds_hf, ds_config = DATASET_MAP.get(dataset_name, DATASET_MAP["wikitext2"])
    data = load_dataset(ds_hf, ds_config, split="test")
    field = "text" if "text" in data.column_names else data.column_names[0]
    text = "\n\n".join(data[field])

    encodings = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls, prev_end_loc = [], 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_slice, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return math.exp(torch.stack(nlls).sum() / end_loc)


def load_model(model_path: str, method: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if method == "awq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
    elif method == "gptq":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(model_path, device="cuda")
    elif method in ("bnb4", "bnb8"):
        from models.load_bnb import load_bnb
        bits = 4 if method == "bnb4" else 8
        model, tokenizer = load_bnb(model_path, bits=bits)
    else:  # fp16 baseline
        from models.load_fp16 import load_fp16
        model, tokenizer = load_fp16(model_path)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perplexity evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default="fp16",
                        choices=["fp16", "awq", "gptq", "bnb4", "bnb8"])
    parser.add_argument("--dataset", default="wikitext2",
                        choices=list(DATASET_MAP.keys()))
    parser.add_argument("--output", default=None, help="Save JSON result to this path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.method)

    t0 = time.perf_counter()
    ppl = compute_perplexity(model, tokenizer, dataset_name=args.dataset)
    elapsed = time.perf_counter() - t0

    result = {
        "model": args.model_path,
        "method": args.method,
        "dataset": args.dataset,
        "perplexity": round(ppl, 4),
        "eval_time_s": round(elapsed, 2),
    }

    print(f"\n{'='*45}")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*45}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")
