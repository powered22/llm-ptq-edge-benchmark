"""Aggregate Tabel 4 — akurasi GGUF schemes via llama.cpp engine.

Input: ./results/eval/tabel4_<SCHEME>_{likelihood,gsm8k,ifeval}*.json
Output: ./results/eval/tabel4_qwen2.5-0.5b-instruct_gguf.csv

Schemes diukur: F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0 (sama dengan Tabel 2).
Metric per task sama dengan Tabel 1 → comparable cross-table (modulo
engine-level numerical precision differences, ~<0.5pp).

Usage:
    python evaluation/aggregate_tabel4.py
"""
import argparse
import csv
import json
from pathlib import Path

TASK_METRICS = {
    "arc_easy":        "acc_norm,none",
    "arc_challenge":   "acc_norm,none",
    "hellaswag":       "acc_norm,none",
    "winogrande":      "acc,none",
    "mmlu":            "acc,none",
    "truthfulqa_mc2":  "acc,none",
    "gsm8k":           ["exact_match,flexible-extract", "exact_match,strict-match"],
    "ifeval":          ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
}
TASK_ORDER = list(TASK_METRICS.keys())

SCHEMES = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]


def find_result_file(eval_dir: Path, scheme: str, part: str):
    prefix = f"tabel4_{scheme}_{part}"
    candidates = list(eval_dir.glob(f"{prefix}.json")) + \
                 list(eval_dir.glob(f"{prefix}_*.json"))
    candidates = [c for c in candidates if c.suffix == ".json"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data.get("results", {})


def get_metric(results: dict, task: str):
    task_data = results.get(task, {})
    keys = TASK_METRICS[task]
    if isinstance(keys, str):
        keys = [keys]
    for metric_key in keys:
        if metric_key in task_data:
            base, _, filt = metric_key.partition(",")
            err_key = f"{base}_stderr,{filt}"
            return task_data.get(metric_key), task_data.get(err_key)
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="./results/eval")
    ap.add_argument("--output", default="./results/eval/tabel4_qwen2.5-0.5b-instruct_gguf.csv")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    rows = []
    f16_avg = None

    for scheme in SCHEMES:
        row = {
            "model": "Qwen2.5-0.5B-Instruct",
            "engine": "llama.cpp",
            "scheme": scheme,
        }

        lh_file = find_result_file(eval_dir, scheme, "likelihood")
        gen_file = find_result_file(eval_dir, scheme, "gsm8k")
        ifeval_file = find_result_file(eval_dir, scheme, "ifeval")

        if lh_file is None and gen_file is None and ifeval_file is None:
            print(f"[skip] {scheme}: tidak ada file result di {eval_dir}")
            row["_status"] = "MISSING"
            for task in TASK_ORDER:
                row[task] = None
            row["avg"] = None
            rows.append(row)
            continue

        results = {}
        if lh_file: results.update(load_results(lh_file))
        if gen_file: results.update(load_results(gen_file))
        if ifeval_file: results.update(load_results(ifeval_file))

        task_scores = []
        for task in TASK_ORDER:
            val, err = get_metric(results, task)
            row[task] = round(val, 4) if val is not None else None
            row[f"{task}_stderr"] = round(err, 4) if err is not None else None
            if val is not None:
                task_scores.append(val)

        row["avg"] = round(sum(task_scores) / len(task_scores), 4) if task_scores else None
        row["n_tasks_completed"] = len(task_scores)
        row["_lh_file"] = lh_file.name if lh_file else "MISSING"
        row["_gsm8k_file"] = gen_file.name if gen_file else "MISSING"
        row["_ifeval_file"] = ifeval_file.name if ifeval_file else "MISSING"
        row["_status"] = "OK" if len(task_scores) == len(TASK_ORDER) \
                              else f"PARTIAL ({len(task_scores)}/{len(TASK_ORDER)})"

        if scheme == "F16" and row["avg"] is not None:
            f16_avg = row["avg"]

        rows.append(row)

    # Delta vs F16 (baseline within deployment dimension)
    for r in rows:
        if f16_avg is not None and r.get("avg") is not None:
            r["delta_vs_f16_gguf_pp"] = round((r["avg"] - f16_avg) * 100, 2)
        else:
            r["delta_vs_f16_gguf_pp"] = None

    # Print summary
    print()
    print("=" * 130)
    print("  Tabel 4 — Akurasi GGUF schemes (Qwen2.5-0.5B-Instruct, llama.cpp engine)")
    print("=" * 130)
    short = {"arc_easy": "arc_e", "arc_challenge": "arc_c", "hellaswag": "hella",
             "winogrande": "wino", "mmlu": "mmlu", "truthfulqa_mc2": "tqa_mc2",
             "gsm8k": "gsm8k", "ifeval": "ifeval"}
    header = f"{'Scheme':<10}"
    for t in TASK_ORDER:
        header += f"{short[t]:>9}"
    header += f"{'Avg':>9}{'Δ vs F16':>11}{'Status':>14}"
    print(header)
    print("-" * 130)
    for r in rows:
        line = f"{r['scheme']:<10}"
        for t in TASK_ORDER:
            v = r.get(t)
            line += f"{v:>9.4f}" if v is not None else f"{'-':>9}"
        avg = r.get("avg")
        line += f"{avg:>9.4f}" if avg is not None else f"{'-':>9}"
        delta = r.get("delta_vs_f16_gguf_pp")
        line += f"{delta:>+10.2f}pp" if delta is not None else f"{'-':>11}"
        line += f"{r.get('_status', '?'):>14}"
        print(line)
    print("=" * 130)

    # Write CSV
    fieldnames = ["model", "engine", "scheme"]
    for t in TASK_ORDER:
        fieldnames.append(t)
        fieldnames.append(f"{t}_stderr")
    fieldnames += ["avg", "delta_vs_f16_gguf_pp", "n_tasks_completed",
                   "_status", "_lh_file", "_gsm8k_file", "_ifeval_file"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print()
    print(f"CSV saved: {args.output}")
    print()
    print("Untuk paper: bandingkan Tabel 4 (akurasi GGUF) dengan Tabel 2 (kinerja GGUF)")
    print("supaya dapat deployment dimension picture yang lengkap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
