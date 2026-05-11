"""Logging & metrics collection utilities for benchmarking."""
import csv
import json
import time
import os
from dataclasses import dataclass, asdict, field
from typing import Optional
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


@dataclass
class BenchmarkResult:
    model_name: str
    method: str            # fp16 | awq | gptq | bnb4 | bnb8
    hardware: str          # e.g. "RTX 3090" or "Jetson Orin"
    input_length: int
    output_length: int
    batch_size: int
    latency_ms: float
    throughput_tok_per_sec: float
    peak_memory_mb: float
    perplexity: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB."""
    if NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 ** 2
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def get_hardware_name() -> str:
    """Return GPU device name or 'CPU'."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


class BenchmarkLogger:
    """Accumulates BenchmarkResult objects and writes to CSV + JSON."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        self._rows: list = []

    def log(self, result: BenchmarkResult):
        self._rows.append(asdict(result))
        self._flush_csv()

    def _flush_csv(self):
        if not self._rows:
            return
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

    def save_json(self, path: str = None):
        path = path or self.output_path.replace(".csv", ".json")
        with open(path, "w") as f:
            json.dump(self._rows, f, indent=2)
        print(f"JSON saved to {path}")

    def summary(self):
        if not self._rows:
            return
        print(f"\n{'Method':<12} {'Latency(ms)':<14} {'Tput(tok/s)':<14} {'Mem(MB)':<12} {'PPL'}")
        print("-" * 65)
        for r in self._rows:
            ppl = f"{r['perplexity']:.2f}" if r['perplexity'] else "N/A"
            print(f"{r['method']:<12} {r['latency_ms']:<14.1f} "
                  f"{r['throughput_tok_per_sec']:<14.1f} {r['peak_memory_mb']:<12.1f} {ppl}")
