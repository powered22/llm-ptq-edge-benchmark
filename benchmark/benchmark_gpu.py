"""Latency & memory profiling on GPU (server / workstation).

Usage:
    python benchmark/benchmark_gpu.py \
        --model-path ./results/qwen1.5b-awq \
        --method awq \
        --input-len 512 \
        --output-len 128 \
        --batch-size 1 \
        --output ./results/benchmark_gpu.csv
"""
import argparse
import time
import torch
from transformers import AutoTokenizer

from benchmark.utils import BenchmarkLogger, BenchmarkResult, get_hardware_name

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


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
    else:
        from models.load_fp16 import load_fp16
        model, tokenizer = load_fp16(model_path)
    return model, tokenizer


def run_benchmark(model, tokenizer, input_len: int, output_len: int, batch_size: int):
    # Build a prompt of approximately input_len tokens
    prompt_text = "The future of artificial intelligence in edge computing is "
    prompt_text = (prompt_text * ((input_len // len(prompt_text.split())) + 1))
    inputs = tokenizer(
        [prompt_text] * batch_size,
        return_tensors="pt",
        truncation=True,
        max_length=input_len,
    ).to(next(model.parameters()).device)

    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=output_len, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=output_len, do_sample=False)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) / BENCHMARK_RUNS * 1000
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    throughput = (output_len * batch_size) / (elapsed_ms / 1000)
    return elapsed_ms, throughput, peak_mem_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU latency & memory benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default="fp16",
                        choices=["fp16", "awq", "gptq", "bnb4", "bnb8"])
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="./results/benchmark_gpu.csv")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path} ({args.method})")
    model, tokenizer = load_model(args.model_path, args.method)

    print(f"Running benchmark: input={args.input_len}, output={args.output_len}, bs={args.batch_size}")
    latency, tput, mem = run_benchmark(
        model, tokenizer, args.input_len, args.output_len, args.batch_size
    )

    result = BenchmarkResult(
        model_name=args.model_path,
        method=args.method,
        hardware=get_hardware_name(),
        input_length=args.input_len,
        output_length=args.output_len,
        batch_size=args.batch_size,
        latency_ms=latency,
        throughput_tok_per_sec=tput,
        peak_memory_mb=mem,
    )
    logger = BenchmarkLogger(args.output)
    logger.log(result)

    print(f"\n{'='*45}")
    print(f"  Hardware : {result.hardware}")
    print(f"  Method   : {args.method}")
    print(f"  Latency  : {latency:.1f} ms")
    print(f"  Tput     : {tput:.1f} tok/s")
    print(f"  Peak Mem : {mem:.1f} MB")
    print(f"{'='*45}")
    print(f"Saved to {args.output}")
