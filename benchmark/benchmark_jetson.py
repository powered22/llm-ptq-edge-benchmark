"""Latency & memory profiling on NVIDIA Jetson (edge device).

Jetson-specific setup:
    # Maximize clocks before benchmarking (reduces variance)
    sudo jetson_clocks

    # Optional: monitor power & thermals during run
    jtop  (pip install jetson-stats)

Usage:
    python benchmark/benchmark_jetson.py \
        --model-path ./results/qwen1.5b-bnb4 \
        --method bnb4 \
        --input-len 256 \
        --output-len 64 \
        --output ./results/benchmark_jetson.csv
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.utils import BenchmarkLogger, BenchmarkResult

try:
    from jtop import jtop as JTop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def get_jetson_memory_mb() -> float:
    """Read available RAM from /proc/meminfo (works on Jetson Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024   # kB -> MB
    except Exception:
        pass
    return 0.0


def load_model_jetson(model_path: str, method: str):
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


def run_benchmark_jetson(model, tokenizer, input_len: int, output_len: int):
    prompt = "The weather in the mountains is expected to change rapidly in " * 10
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=input_len).to(device)

    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=output_len, do_sample=False)

    # Benchmark
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=output_len, do_sample=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    std_lat = (sum((x - avg_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
    throughput = output_len / (avg_lat / 1000)
    mem_mb = get_jetson_memory_mb()

    print(f"Latency: {avg_lat:.1f} ± {std_lat:.1f} ms over {BENCHMARK_RUNS} runs")
    return avg_lat, throughput, mem_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson edge benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", default="bnb4",
                        choices=["fp16", "awq", "gptq", "bnb4", "bnb8"])
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--output", default="./results/benchmark_jetson.csv")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path} ({args.method}) on Jetson")
    model, tokenizer = load_model_jetson(args.model_path, args.method)

    latency, tput, mem = run_benchmark_jetson(
        model, tokenizer, args.input_len, args.output_len
    )

    result = BenchmarkResult(
        model_name=args.model_path,
        method=args.method,
        hardware="Jetson",
        input_length=args.input_len,
        output_length=args.output_len,
        batch_size=1,
        latency_ms=latency,
        throughput_tok_per_sec=tput,
        peak_memory_mb=mem,
    )
    logger = BenchmarkLogger(args.output)
    logger.log(result)

    print(f"\n{'='*45}")
    print(f"  Hardware : Jetson")
    print(f"  Method   : {args.method}")
    print(f"  Latency  : {latency:.1f} ms")
    print(f"  Tput     : {tput:.1f} tok/s")
    print(f"  Mem avail: {mem:.0f} MB")
    print(f"{'='*45}")
    print(f"Saved to {args.output}")
