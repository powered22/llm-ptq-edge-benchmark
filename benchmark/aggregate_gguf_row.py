"""Parse llama-bench JSON + tegrastats log → satu row CSV untuk Tabel 2.

Workload diasumsikan: prompt 256 tok + generate 64 tok (matching baseline HF FP16).
Baseline FP16 di HF transformers untuk speedup: 11.5 tok/s (Qwen2.5-0.5B-Instruct, Jetson Orin Nano).
"""
import argparse
import csv
import json
import re
from pathlib import Path

# Konstanta workload — harus sinkron dengan -p / -n di llama-bench
PROMPT_TOKENS = 256
GENERATE_TOKENS = 64

# Baseline HF FP16 untuk hitung speedup. Update kalau ada baseline baru.
HF_FP16_BASELINE_TOK_S = 11.5

# Regex tegrastats (format Jetson Orin Nano)
RE_POWER_VDD_IN = re.compile(r'VDD_IN (\d+)mW')
RE_RAM = re.compile(r'RAM (\d+)/\d+MB')
RE_GR3D = re.compile(r'GR3D_FREQ (\d+)%')


def parse_tegra(path: str):
    """Return dict: avg_power_mw, peak_ram_mb, ram_delta_mb, avg_gpu_util_pct."""
    powers, rams, gpus = [], [], []
    with open(path) as f:
        for line in f:
            m = RE_POWER_VDD_IN.search(line)
            if m:
                powers.append(int(m.group(1)))
            m = RE_RAM.search(line)
            if m:
                rams.append(int(m.group(1)))
            m = RE_GR3D.search(line)
            if m:
                gpus.append(int(m.group(1)))

    if not powers:
        print(f"[warn] No VDD_IN samples in {path} — apakah tegrastats jalan dengan sudo?")
    if not rams:
        print(f"[warn] No RAM samples in {path}")

    avg_power = sum(powers) / len(powers) if powers else 0.0
    peak_ram = max(rams) if rams else 0
    baseline_ram = rams[0] if rams else 0   # sebelum llama-bench mulai
    ram_delta = peak_ram - baseline_ram
    avg_gpu = sum(gpus) / len(gpus) if gpus else 0.0

    return {
        "avg_power_mw": avg_power,
        "peak_ram_mb": peak_ram,
        "ram_delta_mb": ram_delta,
        "avg_gpu_util_pct": avg_gpu,
        "n_samples": len(powers),
    }


def parse_bench(path: str):
    """Return dict per test_name. Keys we care about: pp<N>, tg<N>."""
    with open(path) as f:
        data = json.load(f)

    out = {}
    for row in data:
        # llama-bench dapat pakai field "test" atau "test_name" tergantung versi
        name = row.get("test_name") or row.get("test") or row.get("n_prompt") or "?"
        out[name] = {
            "avg_ts": row.get("avg_ts", 0.0),
            "stddev_ts": row.get("stddev_ts", 0.0),
            "n_prompt": row.get("n_prompt", 0),
            "n_gen": row.get("n_gen", 0),
        }
    return out


def find_test(bench_rows: dict, prefix: str):
    """Find pp256-style test row in bench output (versi llama-bench bisa beda format)."""
    for key, val in bench_rows.items():
        if isinstance(key, str) and key.startswith(prefix):
            return val
        if prefix == "pp" and val.get("n_prompt", 0) > 0 and val.get("n_gen", 0) == 0:
            return val
        if prefix == "tg" and val.get("n_gen", 0) > 0 and val.get("n_prompt", 0) == 0:
            return val
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--bench-json", required=True)
    ap.add_argument("--tegra-log", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    bench = parse_bench(args.bench_json)
    tegra = parse_tegra(args.tegra_log)

    pp = find_test(bench, "pp") or {}
    tg = find_test(bench, "tg") or {}
    pp_tps = pp.get("avg_ts", 0.0)
    pp_std = pp.get("stddev_ts", 0.0)
    tg_tps = tg.get("avg_ts", 0.0)
    tg_std = tg.get("stddev_ts", 0.0)

    if pp_tps and tg_tps:
        latency_total_ms = (PROMPT_TOKENS / pp_tps + GENERATE_TOKENS / tg_tps) * 1000
        throughput_overall = GENERATE_TOKENS / (latency_total_ms / 1000)
    else:
        latency_total_ms = 0.0
        throughput_overall = 0.0

    # energy/token = power * latency / tokens
    # mW * ms = µJ, jadi mJ = (mW * ms) / 1000
    energy_per_token_mj = (
        tegra["avg_power_mw"] * latency_total_ms / GENERATE_TOKENS / 1000
        if latency_total_ms else 0.0
    )

    speedup = throughput_overall / HF_FP16_BASELINE_TOK_S if HF_FP16_BASELINE_TOK_S else 0.0

    row = {
        "label": args.label,
        "prompt_eval_tok_s": round(pp_tps, 2),
        "prompt_eval_std": round(pp_std, 2),
        "generation_tok_s": round(tg_tps, 2),
        "generation_std": round(tg_std, 2),
        "latency_total_ms": round(latency_total_ms, 1),
        "throughput_overall_tok_s": round(throughput_overall, 2),
        "peak_ram_mb": tegra["peak_ram_mb"],
        "ram_delta_mb": tegra["ram_delta_mb"],
        "avg_power_mw": round(tegra["avg_power_mw"], 0),
        "energy_per_token_mj": round(energy_per_token_mj, 2),
        "avg_gpu_util_pct": round(tegra["avg_gpu_util_pct"], 1),
        "speedup_vs_hf_fp16": round(speedup, 2),
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print()
    print("=" * 62)
    print(f"  Label                 : {args.label}")
    print(f"  Prompt eval (pp256)   : {pp_tps:.2f} ± {pp_std:.2f} tok/s")
    print(f"  Generation  (tg64)    : {tg_tps:.2f} ± {tg_std:.2f} tok/s")
    print(f"  Latency (256+64 tok)  : {latency_total_ms:.1f} ms")
    print(f"  Throughput overall    : {throughput_overall:.2f} tok/s")
    print(f"  Peak RAM              : {tegra['peak_ram_mb']} MB  (Δ {tegra['ram_delta_mb']} MB)")
    print(f"  Avg power (VDD_IN)    : {tegra['avg_power_mw']:.0f} mW  ({tegra['n_samples']} samples)")
    print(f"  Avg GPU util          : {tegra['avg_gpu_util_pct']:.1f} %")
    print(f"  Energy / token        : {energy_per_token_mj:.2f} mJ")
    print(f"  Speedup vs HF-FP16    : {speedup:.2f}x  (HF baseline = {HF_FP16_BASELINE_TOK_S} tok/s)")
    print("=" * 62)
    print(f"  Row appended ke: {csv_path}")


if __name__ == "__main__":
    main()
