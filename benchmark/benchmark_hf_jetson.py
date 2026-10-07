"""HF-transformers benchmark on Jetson, measured the same way as the GGUF path (Table 2).

Mirrors scripts/jetson/benchmark_gguf_jetson.sh + llama-bench:
  - workload: 256 prompt tokens + 64 generated tokens, 5 timed reps
  - tegrastats at 200 ms, VDD_IN power
  - output is a llama-bench-style JSON (pp256 / tg64 tok/s), so the SAME
    benchmark/aggregate_gguf_row.py turns it into a Table 2 row

Differences that exist only because HF is a different engine:
  - exactly 256 prompt tokens and exactly 64 new tokens (min_new_tokens), so EOS cannot cut a run short
  - tegrastats runs during the timed reps only; python import + model load take far longer
    than llama.cpp's load and would otherwise dilute the average power with idle time
  - memory guards (pre-flight free RAM check, swap detection) because Orin Nano has 8 GB shared
    between CPU and GPU

Run it through scripts/jetson/benchmark_hf_jetson.sh, not directly.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time

# Must match PROMPT_TOKENS / GENERATE_TOKENS in benchmark/aggregate_gguf_row.py
INPUT_LEN = 256
OUTPUT_LEN = 64
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5
TEGRA_INTERVAL_MS = 200


def read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            info[key] = int(val.split()[0]) / 1024  # kB -> MB
    return info


def used_mb(info):
    return info["MemTotal"] - info["MemAvailable"]


def swap_used_mb(info):
    return info["SwapTotal"] - info["SwapFree"]


class MemMonitor(threading.Thread):
    """Samples /proc/meminfo every 100 ms for the whole process lifetime."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.peak_used = 0.0
        self.min_avail = float("inf")
        self.max_swap = 0.0
        self.win_max_swap = None

    def begin_window(self):
        self.win_max_swap = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            info = read_meminfo()
            self.peak_used = max(self.peak_used, used_mb(info))
            self.min_avail = min(self.min_avail, info["MemAvailable"])
            swap = swap_used_mb(info)
            self.max_swap = max(self.max_swap, swap)
            if self.win_max_swap is not None:
                self.win_max_swap = max(self.win_max_swap, swap)
            time.sleep(0.1)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1)


def start_tegrastats(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    return subprocess.Popen(
        ["sudo", "tegrastats", "--interval", str(TEGRA_INTERVAL_MS), "--logfile", log_path]
    )


def stop_tegrastats(proc):
    time.sleep(0.5)
    subprocess.run(["sudo", "kill", str(proc.pid)], check=False)
    proc.wait()


def load_model(model_path, method, fuse_layers):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if method == "awq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=fuse_layers)
    else:
        # fp16 and compressed-tensors checkpoints (gptq / rtn / smoothquant).
        # device_map="cuda" puts weights straight on the GPU; low_cpu_mem_usage avoids a
        # second full copy in CPU RAM, which on Orin Nano is the same 8 GB the GPU uses.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, device):
    import torch
    text = "The weather in the mountains is expected to change rapidly in the coming days. " * 80
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :INPUT_LEN]
    if ids.shape[1] != INPUT_LEN:
        raise RuntimeError(f"prompt is {ids.shape[1]} tokens, expected {INPUT_LEN}")
    ids = ids.to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def timed_generate(model, tokenizer, inputs, n_new):
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=n_new,
            min_new_tokens=n_new,
            do_sample=False,
            synced_gpus=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    produced = out.shape[1] - inputs["input_ids"].shape[1]
    if produced != n_new:
        raise RuntimeError(f"generated {produced} tokens, expected {n_new}")
    return dt


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--method", required=True, choices=["fp16", "gptq", "awq"],
                    help="gptq = any compressed-tensors checkpoint (gptq, rtn, smoothquant)")
    ap.add_argument("--bench-json", required=True, help="llama-bench-style output")
    ap.add_argument("--runtime-json", required=True, help="memory / swap diagnostics")
    ap.add_argument("--tegra-log", required=True, help="tegrastats log for the timed window")
    ap.add_argument("--min-free-mb", type=float, default=2500,
                    help="abort if MemAvailable is below this before loading")
    ap.add_argument("--swap-tolerance-mb", type=float, default=64,
                    help="invalidate the run if swap grows more than this during the timed window")
    ap.add_argument("--no-fuse-layers", action="store_true",
                    help="AWQ only: skip fused kernels (less memory, slower)")
    args = ap.parse_args()

    start = read_meminfo()
    baseline_used = used_mb(start)
    swap_start = swap_used_mb(start)
    print(f"[mem] before import: used {baseline_used:.0f} MB, available {start['MemAvailable']:.0f} MB, "
          f"swap used {swap_start:.0f} MB")
    if start["MemAvailable"] < args.min_free_mb:
        print(f"ABORT: only {start['MemAvailable']:.0f} MB available (< {args.min_free_mb:.0f} MB). "
              "Loading now would push the model into swap and corrupt the energy numbers.\n"
              "  Free memory first: boot headless (sudo systemctl set-default multi-user.target), "
              "stop other services, or close other processes.")
        return 2

    monitor = MemMonitor()
    monitor.start()

    t_load = time.perf_counter()
    model, tokenizer = load_model(args.model_path, args.method, not args.no_fuse_layers)
    import torch
    device = next(model.parameters()).device
    inputs = build_inputs(tokenizer, device)
    gc.collect()
    torch.cuda.empty_cache()
    load_seconds = time.perf_counter() - t_load
    print(f"[mem] after load: used {used_mb(read_meminfo()):.0f} MB ({load_seconds:.1f} s)")

    for _ in range(WARMUP_RUNS):
        timed_generate(model, tokenizer, inputs, 1)
        timed_generate(model, tokenizer, inputs, OUTPUT_LEN)

    monitor.begin_window()
    tegra = start_tegrastats(args.tegra_log)
    time.sleep(1)  # same stabilisation pause as benchmark_gguf_jetson.sh
    pp_runs, tg_runs = [], []
    try:
        for _ in range(BENCHMARK_RUNS):
            t_first = timed_generate(model, tokenizer, inputs, 1)
            t_full = timed_generate(model, tokenizer, inputs, OUTPUT_LEN)
            pp_runs.append(INPUT_LEN / t_first)
            tg_runs.append((OUTPUT_LEN - 1) / (t_full - t_first))
    finally:
        stop_tegrastats(tegra)
    window_swap_growth = monitor.win_max_swap - swap_start
    monitor.stop()

    runtime = {
        "baseline_used_mb": round(baseline_used, 1),
        "peak_used_mb": round(monitor.peak_used, 1),
        "min_mem_available_mb": round(monitor.min_avail, 1),
        "swap_used_start_mb": round(swap_start, 1),
        "max_swap_used_mb": round(monitor.max_swap, 1),
        "swap_growth_in_timed_window_mb": round(window_swap_growth, 1),
        "torch_peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "load_seconds": round(load_seconds, 1),
    }
    with open(args.runtime_json, "w") as f:
        json.dump(runtime, f, indent=2)
    print(f"[mem] {runtime}")

    if window_swap_growth > args.swap_tolerance_mb:
        print(f"INVALID RUN: swap grew by {window_swap_growth:.0f} MB during the timed window, "
              "so latency and energy include swap I/O. No bench.json written.")
        return 3

    pp_mean, pp_std = mean_std(pp_runs)
    tg_mean, tg_std = mean_std(tg_runs)
    rows = [
        {"test": f"pp{INPUT_LEN}", "n_prompt": INPUT_LEN, "n_gen": 0, "avg_ts": pp_mean, "stddev_ts": pp_std},
        {"test": f"tg{OUTPUT_LEN}", "n_prompt": 0, "n_gen": OUTPUT_LEN, "avg_ts": tg_mean, "stddev_ts": tg_std},
    ]
    with open(args.bench_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"pp{INPUT_LEN}: {pp_mean:.2f} ± {pp_std:.2f} tok/s | tg{OUTPUT_LEN}: {tg_mean:.2f} ± {tg_std:.2f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
