"""Aggregate hasil lm-eval-harness (6 metode × 2 part) → 1 CSV untuk Tabel 1.

Input: ./results/eval/qwen2.5-0.5b-instruct_<label>_{likelihood,gsm8k}*.json
       (12 file total, dengan timestamp suffix dari lm-eval)
Output: ./results/eval/tabel1_qwen2.5-0.5b-instruct.csv

Usage:
    python evaluation/aggregate_tabel1.py
    python evaluation/aggregate_tabel1.py --eval-dir ./results/eval --output ./mytable.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

# Metric mana dari lm-eval JSON yang kita pakai per task.
# Konvensi standar di paper PTQ/LLM benchmarks.
# Per-task metric. Bisa string (1 key) atau list (try in order, ambil yang ada).
# Untuk gsm8k pakai flexible-extract dulu — strict-match terlalu strict untuk
# Instruct models yang jawabnya conversational (bukan format "#### NNN" dataset).
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

# Methods (label di filename) — urutan untuk row CSV
METHODS = ["fp16", "awqW4A16", "gptqW4A16", "rtnW4A16", "rtnW8A16", "smoothW8A8"]

# Map label → (Method, Bits) yang readable untuk tabel paper
LABEL_PARSE = {
    "fp16":       ("(baseline)",   "FP16"),
    "awqW4A16":   ("AWQ",          "W4A16"),
    "gptqW4A16":  ("GPTQ",         "W4A16"),
    "rtnW4A16":   ("RTN",          "W4A16"),
    "rtnW8A16":   ("RTN",          "W8A16"),
    "smoothW8A8": ("SmoothQuant",  "W8A8"),
}


def find_result_file(eval_dir: Path, label: str, part: str):
    """Cari file output lm-eval yang match prefix (dengan/tanpa timestamp).
    Kalau ada beberapa (multiple runs), pilih yang paling baru by mtime.
    """
    prefix = f"qwen2.5-0.5b-instruct_{label}_{part}"
    candidates = list(eval_dir.glob(f"{prefix}.json")) + \
                 list(eval_dir.glob(f"{prefix}_*.json"))
    # Filter: hanya .json, exclude .jsonl (samples files)
    candidates = [c for c in candidates if c.suffix == ".json"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results(json_path: Path) -> dict:
    """Load lm-eval JSON → dict {task: {metric_key: value}}."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("results", {})


def get_metric(results: dict, task: str):
    """Return (value, stderr) untuk task tertentu. Support fallback list of keys.
    Iterasi candidate keys; ambil yang pertama ditemukan di results."""
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
    ap.add_argument("--eval-dir", default="./results/eval",
                    help="Directory berisi JSON output lm-eval")
    ap.add_argument("--output", default="./results/eval/tabel1_qwen2.5-0.5b-instruct.csv",
                    help="Path CSV output")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} tidak ada")
        return 1

    rows = []
    fp16_avg = None

    for label in METHODS:
        method_name, bits = LABEL_PARSE[label]
        row = {
            "model": "Qwen2.5-0.5B-Instruct",
            "method_label": label,
            "method": method_name,
            "bits": bits,
        }

        lh_file = find_result_file(eval_dir, label, "likelihood")
        gen_file = find_result_file(eval_dir, label, "gsm8k")
        ifeval_file = find_result_file(eval_dir, label, "ifeval")

        if lh_file is None and gen_file is None and ifeval_file is None:
            print(f"[skip] {label}: tidak ada file result ditemukan di {eval_dir}")
            row["_status"] = "MISSING"
            for task in TASK_ORDER:
                row[task] = None
            row["avg"] = None
            rows.append(row)
            continue

        # Merge results dari semua part (likelihood + gsm8k + ifeval)
        results = {}
        if lh_file:
            results.update(load_results(lh_file))
        if gen_file:
            results.update(load_results(gen_file))
        if ifeval_file:
            results.update(load_results(ifeval_file))

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
        n_total = len(TASK_ORDER)
        row["_status"] = "OK" if len(task_scores) == n_total else f"PARTIAL ({len(task_scores)}/{n_total})"

        if label == "fp16" and row["avg"] is not None:
            fp16_avg = row["avg"]

        rows.append(row)

    # Hitung delta vs FP16
    for r in rows:
        avg = r.get("avg")
        if fp16_avg is not None and avg is not None:
            r["delta_vs_fp16_pp"] = round((avg - fp16_avg) * 100, 2)
        else:
            r["delta_vs_fp16_pp"] = None

    # Print summary table ke terminal
    print()
    print("=" * 130)
    print("  Tabel 1 — Akurasi metode kuantisasi (Qwen2.5-0.5B-Instruct, lm-eval-harness)")
    print("=" * 130)
    header = f"{'Method':<14}{'Bits':<8}"
    short_tasks = {"arc_easy": "arc_e", "arc_challenge": "arc_c",
                   "hellaswag": "hella", "winogrande": "wino",
                   "mmlu": "mmlu", "truthfulqa_mc2": "tqa_mc2",
                   "gsm8k": "gsm8k", "ifeval": "ifeval"}
    for task in TASK_ORDER:
        header += f"{short_tasks[task]:>9}"
    header += f"{'Avg':>9}{'Δ vs FP16':>11}{'Status':>14}"
    print(header)
    print("-" * 130)
    for r in rows:
        line = f"{r['method']:<14}{r['bits']:<8}"
        for task in TASK_ORDER:
            v = r.get(task)
            line += f"{v:>9.4f}" if v is not None else f"{'-':>9}"
        avg = r.get("avg")
        line += f"{avg:>9.4f}" if avg is not None else f"{'-':>9}"
        delta = r.get("delta_vs_fp16_pp")
        line += f"{delta:>+10.2f}pp" if delta is not None else f"{'-':>11}"
        line += f"{r.get('_status', '?'):>14}"
        print(line)
    print("=" * 130)

    # Tulis CSV
    fieldnames = ["model", "method_label", "method", "bits"]
    for task in TASK_ORDER:
        fieldnames.append(task)
        fieldnames.append(f"{task}_stderr")
    fieldnames += ["avg", "delta_vs_fp16_pp", "n_tasks_completed",
                   "_status", "_lh_file", "_gsm8k_file", "_ifeval_file"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print()
    print(f"CSV saved: {args.output}")
    print(f"Kolom utama untuk paper: method, bits, {', '.join(TASK_ORDER)}, avg, delta_vs_fp16_pp")
    print(f"Kolom _* di akhir CSV adalah metadata audit trail (file source, status).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
