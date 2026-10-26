"""Build Table 3 (accuracy, throughput, energy, memory + Pareto flags) for one model scale.

Inputs: the Jetson summary CSVs (GGUF and HF) and analysis/accuracy_by_scale.json.
Rows that did not run on the 8 GB Jetson (OOM / not run) are listed with blank speed and energy and are
left out of the Pareto test.

Usage:
  python analysis/build_table3.py --model Qwen2.5-1.5B-Instruct --gguf-csv summary_gguf.csv \
      --hf-csv summary_hf.csv --out-prefix table3_1.5b
"""
import argparse
import csv
import json
from pathlib import Path

HF_LABELS = [("fp16", "FP16", "16"), ("awqW4A16", "AWQ", "W4A16"), ("gptqW4A16", "GPTQ", "W4A16"),
             ("rtnW4A16", "RTN", "W4A16"), ("rtnW8A16", "RTN", "W8A16"), ("smoothW8A8", "SmoothQuant", "W8A8")]
GGUF_LABELS = [("F16", "16"), ("Q8_0", "8"), ("Q5_K_M", "~5.5"), ("Q4_K_M", "~4.8"), ("Q4_0", "4")]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_gguf(path, prefix):
    """GGUF summary rows by label. Older rows were written with a shifted header (energy under
    avg_gpu_util_pct, power under energy_per_token_mj); detect and realign them."""
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        if not r["label"].startswith(prefix):
            continue
        e = fnum(r["energy_per_token_mj"])
        if e is not None and e > 5000:      # this is a power in mW, the row is shifted by one column
            r["energy_per_token_mj"] = r["avg_gpu_util_pct"]
        out[r["label"][len(prefix):]] = r     # later rows win (re-runs)
    return out


def load_hf(path, prefix):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        if r["label"].startswith(prefix):
            out[r["label"][len(prefix):]] = r
    return out


def dominates(q, p, tol):
    return (q["acc"] >= p["acc"] - tol and q["thr"] >= p["thr"] and q["energy"] <= p["energy"]
            and (q["acc"] > p["acc"] - tol or q["thr"] > p["thr"] or q["energy"] < p["energy"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf-csv", required=True)
    ap.add_argument("--hf-csv", required=True)
    ap.add_argument("--acc-json", default=str(Path(__file__).with_name("accuracy_by_scale.json")))
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    acc = json.load(open(a.acc_json))[a.model]
    key = a.model.lower()
    gguf = load_gguf(a.gguf_csv, f"{key}_")
    hf = load_hf(a.hf_csv, f"{key}_hf_")

    rows = []
    for lab, method, bits in HF_LABELS:
        r = hf.get(lab)
        rows.append({"engine": "HF", "method": method, "bits": bits, "acc": acc["hf"][lab],
                     "thr": fnum(r["throughput_overall_tok_s"]) if r else None,
                     "energy": fnum(r["energy_per_token_mj"]) if r else None,
                     "mem": fnum(r["ram_delta_mb"]) if r else None})
    for lab, bits in GGUF_LABELS:
        r = gguf.get(lab)
        rows.append({"engine": "GGUF", "method": lab, "bits": bits, "acc": acc["gguf"][lab],
                     "thr": fnum(r["throughput_overall_tok_s"]) if r else None,
                     "energy": fnum(r["energy_per_token_mj"]) if r else None,
                     "mem": fnum(r["ram_delta_mb"]) if r else None})

    ran = [r for r in rows if r["thr"] is not None and r["energy"] is not None]
    for r in rows:
        if r not in ran:
            r["strict"] = r["tol"] = "n/a"
            continue
        r["strict"] = "no" if any(q is not r and dominates(q, r, 0.0) for q in ran) else "yes"
        r["tol"] = "no" if any(q is not r and dominates(q, r, 1.0) for q in ran) else "yes"

    def f(v, d=2):
        return "OOM" if v is None else f"{v:.{d}f}"

    with open(f"{a.out_prefix}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["engine", "method", "bits", "avg_accuracy_pct", "throughput_overall_tok_s",
                    "energy_mj_per_token", "ram_delta_mb", "pareto_strict", "pareto_tol_1pp"])
        for r in rows:
            w.writerow([r["engine"], r["method"], r["bits"], f"{r['acc']:.2f}",
                        "" if r["thr"] is None else f"{r['thr']:.2f}",
                        "" if r["energy"] is None else f"{r['energy']:.2f}",
                        "" if r["mem"] is None else f"{r['mem']:.0f}", r["strict"], r["tol"]])

    with open(f"{a.out_prefix}.md", "w") as fh:
        fh.write(f"# Table 3: Pareto analysis, {a.model} on Jetson Orin Nano\n\n")
        fh.write("| Engine | Method | Bits | Avg acc (%) | Throughput (tok/s) | Energy (mJ/token) | RAM delta (MB) | Pareto strict | Pareto (1 pp tol.) |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['engine']} | {r['method']} | {r['bits']} | {r['acc']:.2f} | {f(r['thr'])} | "
                     f"{f(r['energy'], 1)} | {f(r['mem'], 0)} | {r['strict']} | {r['tol']} |\n")
    for r in rows:
        print(r["engine"], r["method"], r["bits"], f"{r['acc']:.2f}", f(r["thr"]), f(r["energy"], 1),
              f(r["mem"], 0), r["strict"], r["tol"])


if __name__ == "__main__":
    main()
