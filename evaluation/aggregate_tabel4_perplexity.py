"""Aggregate Tabel 4 perplexity hasil llama-perplexity.

Input: ./results/eval/tabel4_gguf/perplexity_<SCHEME>.txt (raw output llama-perplexity)
Output: ./results/eval/tabel4_gguf/tabel4_perplexity.csv

Usage: python evaluation/aggregate_tabel4_perplexity.py
"""
import argparse
import csv
import re
from pathlib import Path

SCHEMES = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]

# Pattern: "Final estimate: PPL = 8.7351 +/- 0.04123"
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")

# Pattern: "Final HellaSwag score(1000 tasks) = 45.20 +/- 1.5736"
HS_RE = re.compile(r"Final HellaSwag score\(\d+ tasks\)\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")

# Pattern: "Final Winogrande score(1267 tasks) = 55.42 +/- 1.3993"
WG_RE = re.compile(r"Final Winogrande score\(\d+ tasks\)\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


def parse_perplexity(path: Path):
    """Return (ppl, stderr) atau (None, None)."""
    with open(path) as f:
        text = f.read()
    m = PPL_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def parse_score(path: Path, regex: re.Pattern):
    """Generic parser untuk HellaSwag/Winogrande score lines."""
    if not path.exists():
        return None, None
    with open(path) as f:
        text = f.read()
    m = regex.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="./results/eval/tabel4_gguf")
    ap.add_argument("--output", default="./results/eval/tabel4_gguf/tabel4_perplexity.csv")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    f16_ppl = None

    for scheme in SCHEMES:
        ppl_file = input_dir / f"perplexity_{scheme}.txt"
        hs_file = input_dir / f"hellaswag_{scheme}.txt"
        wg_file = input_dir / f"winogrande_{scheme}.txt"

        row = {"scheme": scheme}

        # Perplexity (required)
        if ppl_file.exists():
            ppl, ppl_err = parse_perplexity(ppl_file)
            if ppl is not None:
                row["perplexity"] = round(ppl, 4)
                row["perplexity_stderr"] = round(ppl_err, 4)
                if scheme == "F16":
                    f16_ppl = ppl
            else:
                row["perplexity"] = None
                row["perplexity_stderr"] = None
        else:
            row["perplexity"] = None
            row["perplexity_stderr"] = None

        # HellaSwag (optional, supplementary)
        hs, hs_err = parse_score(hs_file, HS_RE)
        row["hellaswag"] = round(hs, 2) if hs is not None else None
        row["hellaswag_stderr"] = round(hs_err, 2) if hs_err is not None else None

        # Winogrande (optional, supplementary)
        wg, wg_err = parse_score(wg_file, WG_RE)
        row["winogrande"] = round(wg, 2) if wg is not None else None
        row["winogrande_stderr"] = round(wg_err, 2) if wg_err is not None else None

        # Status
        if row["perplexity"] is None:
            row["_status"] = "MISSING"
        elif row["hellaswag"] is None or row["winogrande"] is None:
            row["_status"] = "PPL-ONLY"
        else:
            row["_status"] = "OK"

        rows.append(row)

    # Hitung delta vs F16 untuk perplexity
    for r in rows:
        if f16_ppl is not None and r.get("perplexity") is not None:
            r["delta_vs_f16"] = round(r["perplexity"] - f16_ppl, 4)
        else:
            r["delta_vs_f16"] = None

    # Print summary
    print()
    print("=" * 100)
    print("  Tabel 4 — Akurasi GGUF schemes (Qwen2.5-0.5B-Instruct, llama.cpp engine)")
    print("=" * 100)
    print(f"  {'Scheme':<10}{'PPL ↓':>10}{'±err':>8}{'ΔPPL vs F16':>14}{'HellaSwag ↑':>13}{'±err':>8}{'Winogrande ↑':>14}{'±err':>8}{'Status':>12}")
    print("-" * 100)
    for r in rows:
        ppl = r.get("perplexity")
        ppl_err = r.get("perplexity_stderr")
        delta = r.get("delta_vs_f16")
        hs = r.get("hellaswag")
        hs_err = r.get("hellaswag_stderr")
        wg = r.get("winogrande")
        wg_err = r.get("winogrande_stderr")
        status = r.get("_status", "?")
        ppl_s = f"{ppl:.4f}" if ppl is not None else "-"
        ppl_err_s = f"±{ppl_err:.3f}" if ppl_err is not None else "-"
        delta_s = f"{delta:+.4f}" if delta is not None else "-"
        hs_s = f"{hs:.2f}%" if hs is not None else "-"
        hs_err_s = f"±{hs_err:.2f}" if hs_err is not None else "-"
        wg_s = f"{wg:.2f}%" if wg is not None else "-"
        wg_err_s = f"±{wg_err:.2f}" if wg_err is not None else "-"
        print(f"  {r['scheme']:<10}{ppl_s:>10}{ppl_err_s:>8}{delta_s:>14}{hs_s:>13}{hs_err_s:>8}{wg_s:>14}{wg_err_s:>8}{status:>12}")
    print("=" * 100)
    print("  Catatan: PPL lebih rendah = lebih baik. HellaSwag/Winogrande dalam % (lebih tinggi = lebih baik).")
    print()

    # Tulis CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scheme", "perplexity", "perplexity_stderr", "delta_vs_f16",
                           "hellaswag", "hellaswag_stderr",
                           "winogrande", "winogrande_stderr", "_status"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
