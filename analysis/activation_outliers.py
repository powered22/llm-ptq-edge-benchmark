#!/usr/bin/env python3
"""Measure activation outliers of FP16 models, to explain SmoothQuant W8A8 losses.

For every nn.Linear (except lm_head) this script records, over a set of GSM8K
prompts formatted with the chat template:

  * per input channel, the largest |activation| over all tokens;
  * outlier ratio = largest channel max / median channel max;
  * the relative error of dynamic per-token INT8 quantization of the layer input,
    once on the raw activations and once after SmoothQuant-style smoothing with
    strength alpha (s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)).

The smoothed error is an approximation: it smooths every linear layer with its
own weight, whereas llm-compressor smooths only the layers in its mappings.
It shows how hard each model's activations are to quantize, not the exact
error of the released checkpoints.

Two passes over the same prompts: pass 1 collects per-channel maxima (the
smoothing scales need them), pass 2 measures the quantization error.

Example:
  python analysis/activation_outliers.py \
      --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
      --n-prompts 32 --out-dir results/analysis/outliers
"""
import argparse
import csv
import math
import os
import statistics

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompts(tokenizer, n, max_len, with_answer):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    texts = []
    for row in ds.select(range(n)):
        msgs = [{"role": "user", "content": row["question"]}]
        if with_answer:
            msgs.append({"role": "assistant", "content": row["answer"]})
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=not with_answer)
        else:
            text = row["question"] + ("\n" + row["answer"] if with_answer else "")
        texts.append(text)
    return [tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len,
                      add_special_tokens=False) for t in texts]


def int8_per_token(x):
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (x / scale).round().clamp(-128, 127) * scale


def analyse(model_id, args, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    cfg = model.config
    layers = {n: m for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    inputs = build_prompts(tok, args.n_prompts, args.max_len, not args.prompt_only)

    act_max = {n: torch.zeros(m.in_features, device=device, dtype=torch.float32)
               for n, m in layers.items()}
    err = {n: [0.0, 0.0, 0.0] for n in layers}  # raw err^2, smoothed err^2, |x|^2
    smooth = {}
    state = {"pass": 1}

    def make_hook(name):
        def hook(_mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if state["pass"] == 1:
                torch.maximum(act_max[name], x.abs().amax(dim=0), out=act_max[name])
            else:
                norm = (x ** 2).sum().item()
                err[name][0] += ((x - int8_per_token(x)) ** 2).sum().item()
                xs = x / smooth[name]
                err[name][1] += (((xs - int8_per_token(xs)) * smooth[name]) ** 2).sum().item()
                err[name][2] += norm
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in layers.items()]
    n_tokens = 0
    with torch.no_grad():
        for enc in inputs:
            model(**{k: v.to(device) for k, v in enc.items()})
            n_tokens += enc["input_ids"].shape[1]
        for n, m in layers.items():
            w_max = m.weight.detach().float().abs().amax(dim=0).clamp(min=1e-5)
            a = act_max[n].clamp(min=1e-5)
            smooth[n] = (a.pow(args.alpha) / w_max.pow(1 - args.alpha)).clamp(min=1e-5)
        state["pass"] = 2
        for enc in inputs:
            model(**{k: v.to(device) for k, v in enc.items()})
    for h in handles:
        h.remove()

    rows = []
    for n in layers:
        a = act_max[n]
        med = a.median().item()
        top = a.max().item()
        raw, smo, norm = err[n]
        rows.append({
            "model": model_id,
            "layer": n,
            "kind": n.split(".")[-1],
            "block": next((int(p) for p in n.split(".") if p.isdigit()), -1),
            "max_channel_abs": round(top, 4),
            "median_channel_abs": round(med, 6),
            "outlier_ratio": round(top / max(med, 1e-8), 2),
            "top_channel": int(a.argmax().item()),
            "int8_rel_err_raw": round(math.sqrt(raw / max(norm, 1e-12)), 5),
            "int8_rel_err_smoothed": round(math.sqrt(smo / max(norm, 1e-12)), 5),
        })
    summary = {
        "model": model_id,
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "n_prompts": len(inputs),
        "n_tokens": n_tokens,
        "median_outlier_ratio": round(statistics.median(r["outlier_ratio"] for r in rows), 2),
        "max_outlier_ratio": max(r["outlier_ratio"] for r in rows),
        "max_channel_abs": max(r["max_channel_abs"] for r in rows),
        "mean_int8_rel_err_raw": round(statistics.mean(r["int8_rel_err_raw"] for r in rows), 5),
        "mean_int8_rel_err_smoothed": round(statistics.mean(r["int8_rel_err_smoothed"] for r in rows), 5),
        "worst_layer_smoothed": max(rows, key=lambda r: r["int8_rel_err_smoothed"])["layer"],
        "worst_int8_rel_err_smoothed": max(r["int8_rel_err_smoothed"] for r in rows),
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant smoothing strength")
    ap.add_argument("--prompt-only", action="store_true",
                    help="feed only the question (default: question + reference answer)")
    ap.add_argument("--out-dir", default="results/analysis/outliers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    for model_id in args.models:
        print(f"== {model_id}", flush=True)
        rows, summary = analyse(model_id, args, args.device, dtype)
        tag = model_id.split("/")[-1].lower()
        with open(os.path.join(args.out_dir, f"layers_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summaries.append(summary)
        for k, v in summary.items():
            print(f"   {k}: {v}")

    path = os.path.join(args.out_dir, f"summary_alpha{args.alpha}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSummary written to {path}")


if __name__ == "__main__":
    main()
