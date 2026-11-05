"""List the CUDA kernels that actually run when an HF checkpoint generates text on the Jetson.

Purpose: check whether the SmoothQuant W8A8 checkpoint uses INT8 GEMM kernels
(INT8 Tensor Cores) or falls back to FP16 matmuls plus extra quantize/dequantize ops.

It loads the model exactly like benchmark/benchmark_hf_jetson.py (torch_dtype=float16,
device_map="cuda"), runs the same 256-token prompt, and profiles generation of a few tokens.

Usage on the Jetson (run once for SmoothQuant, once for FP16 as the reference):
    python3 scripts/jetson/profile_hf_kernels.py --model-path results/qwen2.5_0.5b_instruct_smooth_w8a8_v2 --out results/profile_sq.txt
    python3 scripts/jetson/profile_hf_kernels.py --model-path Qwen/Qwen2.5-0.5B-Instruct --out results/profile_fp16.txt

How to read the output:
  1. "Linear layer classes / weight dtypes": shows how transformers loaded the quantized layers.
     torch.int8 weights mean the checkpoint stayed compressed; float16 means it was decompressed at load.
  2. "INT8 GEMM kernels found": any kernel whose name looks like an integer matmul
     (imma, igemm, i8, s8, int8 in a gemm/cutlass/cublas kernel). Empty = no INT8 Tensor Core path.
  3. "Top kernels by GPU time": for SmoothQuant, expect the same FP16 gemm kernels as the FP16 run,
     plus extra elementwise/reduction kernels (abs, amax, div, round, clamp, mul) from the
     on-the-fly activation quantize/dequantize.
  4. "Kernel launches per generated token": a large increase over FP16 means extra small ops per layer,
     which matches the low GPU utilisation (9 %) seen in Table 2.
"""
import argparse
import re
from collections import Counter

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_LEN = 256  # same prompt length as benchmark_hf_jetson.py
INT8_PATTERN = re.compile(r"(imma|igemm|i8|s8|int8|u8)", re.IGNORECASE)
GEMM_PATTERN = re.compile(r"(gemm|cutlass|cublas|sm\d+_xmma|ampere_|matmul)", re.IGNORECASE)


def build_inputs(tokenizer):
    text = "The weather in the mountains is expected to change rapidly in the coming days. " * 80
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :INPUT_LEN].to("cuda")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def generate(model, tokenizer, inputs, n_new):
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()


def profile_kernels(model, tokenizer, inputs, n_new):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        generate(model, tokenizer, inputs, n_new)
    kernels = Counter()
    gpu_us = Counter()
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            kernels[evt.name] += 1
            gpu_us[evt.name] += evt.device_time_total if hasattr(evt, "device_time_total") else evt.cuda_time_total
    return kernels, gpu_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--new-tokens", type=int, default=8, help="tokens generated while profiling")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    lines = []
    log = lambda s="": (print(s), lines.append(s))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()

    log(f"Model: {args.model_path}")
    log(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    qcfg = getattr(model.config, "quantization_config", None)
    log(f"quantization_config: {qcfg if qcfg is not None else 'none (FP16)'}")

    # 1. How the linear layers were loaded
    classes, wdtypes = Counter(), Counter()
    for name, mod in model.named_modules():
        if "Linear" in type(mod).__name__:
            classes[type(mod).__module__ + "." + type(mod).__name__] += 1
            for pname, p in mod.named_parameters(recurse=False):
                if "weight" in pname:
                    wdtypes[f"{pname}: {p.dtype}"] += 1
    log("\n== Linear layer classes ==")
    for k, v in classes.most_common():
        log(f"  {v:4d}  {k}")
    log("== Linear weight dtypes ==")
    for k, v in wdtypes.most_common():
        log(f"  {v:4d}  {k}")

    inputs = build_inputs(tokenizer)
    generate(model, tokenizer, inputs, 2)  # warm-up (cuBLAS init, caches)

    # Profile prompt-only (1 new token) and full run, so decode kernels per token can be isolated
    k1, _ = profile_kernels(model, tokenizer, inputs, 1)
    kn, us = profile_kernels(model, tokenizer, inputs, args.new_tokens)
    per_token = (sum(kn.values()) - sum(k1.values())) / (args.new_tokens - 1)

    int8 = {k: v for k, v in kn.items() if INT8_PATTERN.search(k) and GEMM_PATTERN.search(k)}
    log("\n== INT8 GEMM kernels found ==")
    if int8:
        for k, v in sorted(int8.items(), key=lambda x: -x[1]):
            log(f"  {v:6d}x  {k[:140]}")
    else:
        log("  NONE: no integer matmul kernel ran. Matmuls are executed in floating point.")

    total_us = sum(us.values())
    log(f"\n== Top {args.top} kernels by GPU time ({args.new_tokens} new tokens, 256-token prompt) ==")
    log(f"  {'share':>6} {'calls':>7}  kernel")
    for k, t in us.most_common(args.top):
        log(f"  {100 * t / total_us:5.1f}% {kn[k]:7d}  {k[:120]}")

    log("\n== Summary ==")
    log(f"  total kernel launches: {sum(kn.values())}")
    log(f"  kernel launches per generated token (decode): {per_token:.0f}")
    log(f"  total GPU kernel time: {total_us / 1e3:.1f} ms")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
