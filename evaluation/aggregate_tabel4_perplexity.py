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


def parse_perplexity(path: Path):
    """Return (ppl, stderr) atau (None, None) kalau tidak ada Final estimate."""
    with open(path) as f:
        text = f.read()
    m = PPL_RE.search(text)
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
        f = input_dir / f"perplexity_{scheme}.txt"
        if not f.exists():
            print(f"[skip] {f} tidak ada")
            rows.append({"scheme": scheme, "perplexity": None, "stderr": None,
                         "delta_vs_f16": None, "_status": "MISSING"})
            continue

        ppl, stderr = parse_perplexity(f)
        if ppl is None:
            print(f"[warn] {f} tidak punya 'Final estimate' — mungkin gagal di tengah jalan")
            rows.append({"scheme": scheme, "perplexity": None, "stderr": None,
                         "delta_vs_f16": None, "_status": "INCOMPLETE"})
            continue

        rows.append({
            "scheme": scheme,
            "perplexity": round(ppl, 4),
            "stderr": round(stderr, 4),
            "delta_vs_f16": None,   # diisi nanti
            "_status": "OK",
        })

        if scheme == "F16":
            f16_ppl = ppl

    # Hitung delta vs F16
    for r in rows:
        if f16_ppl is not None and r["perplexity"] is not None:
            r["delta_vs_f16"] = round(r["perplexity"] - f16_ppl, 4)

    # Print summary
    print()
    print("=" * 70)
    print("  Tabel 4 — Perplexity GGUF schemes (Qwen2.5-0.5B-Instruct, wikitext-2)")
    print("=" * 70)
    print(f"  {'Scheme':<10}{'Perplexity':>14}{'±Stderr':>12}{'Δ vs F16':>14}{'Status':>14}")
    print("-" * 70)
    for r in rows:
        ppl = r.get("perplexity")
        err = r.get("stderr")
        delta = r.get("delta_vs_f16")
        status = r.get("_status", "?")
        ppl_s = f"{ppl:.4f}" if ppl is not None else "-"
        err_s = f"±{err:.4f}" if err is not None else "-"
        delta_s = f"{delta:+.4f}" if delta is not None else "-"
        print(f"  {r['scheme']:<10}{ppl_s:>14}{err_s:>12}{delta_s:>14}{status:>14}")
    print("=" * 70)
    print("  Catatan: PPL lebih rendah = lebih baik. Δ positif = quality drop dari F16.")
    print()

    # Tulis CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scheme", "perplexity", "stderr", "delta_vs_f16", "_status"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
