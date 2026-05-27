"""Visualize PTQ benchmark + evaluation results.

Loads two sources:
  1. CSV benchmark dari benchmark/benchmark_gpu.py  → latency, throughput, peak memory
  2. JSON perplexity dari evaluation/run_perplexity.py → perplexity, dataset

Lalu menghasilkan 4 figure:
  1. Perplexity comparison: bar chart per (model × method)    — butuh hanya PPL JSON
  2. Latency comparison: bar chart per (model × method)        — butuh benchmark CSV
  3. Memory–perplexity trade-off: scatter                       — butuh keduanya
  4. Throughput heatmap: model × method                         — butuh benchmark CSV

Pemakaian CLI:
    python notebooks/analyze_results.py \
        --benchmark-glob "./results/benchmark_*.csv" \
        --ppl-glob "./results/eval/*__ppl_*.json" \
        --output-dir "./results/figures"

Atau import fungsinya:
    from notebooks.analyze_results import (
        load_results,
        plot_latency_comparison,
        plot_memory_perplexity_tradeoff,
        plot_throughput_heatmap,
    )
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Mapping kode method → label cantik untuk plot
METHOD_LABELS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "rtn": "RTN",
    "smooth": "SmoothQuant",
    "fp16": "FP16",
    "bnb4": "BnB-4",
    "bnb8": "BnB-8",
}

_METHOD_TOKENS = {
    "awq": "awq",
    "gptq": "gptq",
    "rtn": "rtn",
    "smooth": "smooth",
    "smoothq": "smooth",
    "smoothquant": "smooth",
    "bnb": "bnb",
    "fp16": "fp16",
}


# ---------------------------------------------------------------------------
# Folder-name parser
# ---------------------------------------------------------------------------

def parse_model_dir(name: str) -> dict:
    """Pecah nama folder model jadi (model_family, method, scheme).

    Mendukung dua konvensi: underscore (`qwen2.5_0.5b_instruct_awq_w4a16`)
    dan dash (`qwen2.5-1.5b-awq-int4-sym`).
    """
    norm = re.sub(r"[-_]+", "-", name.lower())
    parts = norm.split("-")

    # Family = semua token sebelum method token muncul
    family_tokens, method = [], "unknown"
    for tok in parts:
        if tok in _METHOD_TOKENS:
            method = _METHOD_TOKENS[tok]
            break
        family_tokens.append(tok)
    family = "-".join(family_tokens) if family_tokens else name

    # Scheme: cari "w<N>a<M>" atau alias "int4"/"int8"
    scheme = "unknown"
    m = re.search(r"w(\d+)a(\d+)", norm)
    if m:
        scheme = f"W{m.group(1)}A{m.group(2)}"
    elif "int4" in norm:
        scheme = "W4A16"
    elif "int8" in norm:
        scheme = "W8A16"

    return {"model_family": family, "method": method, "scheme": scheme}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_results(
    benchmark_glob: str = "./results/benchmark_*.csv",
    ppl_glob: str = "./results/eval/*__ppl_*.json",
) -> pd.DataFrame:
    """Gabungkan CSV benchmark + JSON perplexity → satu DataFrame tidy."""

    # 1. CSV benchmark (latency / throughput / memory)
    bench_files = sorted(Path().glob(benchmark_glob))
    bench_df = pd.DataFrame()
    if bench_files:
        bench_df = pd.concat([pd.read_csv(f) for f in bench_files], ignore_index=True)
        bench_df["model_dir"] = bench_df["model_name"].apply(lambda p: Path(str(p)).name)
        print(f"[load] benchmark rows : {len(bench_df)} ({len(bench_files)} csv)")
    else:
        print(f"[warn] no benchmark CSV matched {benchmark_glob}")

    # 2. JSON perplexity
    ppl_files = sorted(Path().glob(ppl_glob))
    ppl_rows = []
    for f in ppl_files:
        with open(f) as fh:
            d = json.load(fh)
        ppl_rows.append({
            "model_dir": Path(d.get("model", f.stem.split("__ppl_")[0])).name,
            "perplexity": d.get("perplexity"),
            "ppl_dataset": d.get("dataset"),
        })
    ppl_df = pd.DataFrame(ppl_rows)
    if ppl_rows:
        print(f"[load] perplexity rows: {len(ppl_rows)} ({len(ppl_files)} json)")

    # Merge
    if not bench_df.empty and not ppl_df.empty:
        # Hapus perplexity di benchmark CSV (kalau ada) — pakai yg dari JSON eval
        if "perplexity" in bench_df.columns:
            bench_df = bench_df.drop(columns=["perplexity"])
        merged = bench_df.merge(ppl_df, on="model_dir", how="outer")
    elif not bench_df.empty:
        merged = bench_df
    else:
        merged = ppl_df

    if merged.empty:
        print("[warn] empty result set — tidak ada CSV/JSON yang ditemukan.")
        return merged

    # Tambah kolom turunan dari nama folder
    parsed = merged["model_dir"].apply(parse_model_dir).apply(pd.Series)
    for col in ("model_family", "method", "scheme"):
        if col in merged.columns:
            merged = merged.drop(columns=[col])
        merged[col] = parsed[col]

    return merged


# ---------------------------------------------------------------------------
# 1. Latency comparison
# ---------------------------------------------------------------------------

def plot_latency_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Grouped bar: latency_ms per (model_family × method)."""
    sub = df.dropna(subset=["latency_ms"]).copy() if "latency_ms" in df.columns else pd.DataFrame()
    if sub.empty:
        print("[warn] no latency data — jalankan benchmark/benchmark_gpu.py dulu")
        return None

    sub["method_label"] = sub["method"].map(METHOD_LABELS).fillna(sub["method"])
    pivot = sub.pivot_table(
        index="model_family", columns="method_label",
        values="latency_ms", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.6, width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)
    ax.set_ylabel("Latency (ms) ↓")
    ax.set_xlabel("Model")
    ax.set_title("Inference Latency — Method × Model", fontweight="bold")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 2. Memory vs Perplexity trade-off
# ---------------------------------------------------------------------------

def plot_memory_perplexity_tradeoff(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Scatter: peak_memory_mb (x) vs perplexity (y); warna=method, marker=model_family."""
    needed = {"peak_memory_mb", "perplexity"}
    if not needed.issubset(df.columns):
        print(f"[warn] perlu kolom {needed} — pastikan benchmark + perplexity sudah jalan")
        return None

    sub = df.dropna(subset=list(needed)).copy()
    if sub.empty:
        print("[warn] tidak ada baris dengan peak_memory_mb dan perplexity sekaligus")
        return None

    sub["method_label"] = sub["method"].map(METHOD_LABELS).fillna(sub["method"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    families = sorted(sub["model_family"].unique())
    methods = sorted(sub["method_label"].unique())
    palette = dict(zip(methods, sns.color_palette("muted", len(methods))))

    for i, fam in enumerate(families):
        for method in methods:
            g = sub[(sub["model_family"] == fam) & (sub["method_label"] == method)]
            if g.empty:
                continue
            ax.scatter(
                g["peak_memory_mb"], g["perplexity"],
                label=f"{fam} / {method}",
                s=140, marker=markers[i % len(markers)],
                color=palette[method],
                edgecolor="black", linewidth=0.6, alpha=0.85, zorder=3,
            )

    ax.set_xlabel("Peak Memory (MB) →")
    ax.set_ylabel("Perplexity ↓ (lower is better)")
    ax.set_title("Memory–Perplexity Trade-off", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3. Throughput heatmap
# ---------------------------------------------------------------------------

def plot_throughput_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Heatmap throughput (tok/s) — model_family (row) × method (col)."""
    if "throughput_tok_per_sec" not in df.columns:
        print("[warn] tidak ada kolom throughput_tok_per_sec — jalankan benchmark dulu")
        return None

    sub = df.dropna(subset=["throughput_tok_per_sec"]).copy()
    if sub.empty:
        print("[warn] no throughput data")
        return None

    sub["method_label"] = sub["method"].map(METHOD_LABELS).fillna(sub["method"])
    pivot = sub.pivot_table(
        index="model_family", columns="method_label",
        values="throughput_tok_per_sec", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.7 * len(pivot) + 2)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, cbar_kws={"label": "tok/s"}, ax=ax,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Model")
    ax.set_title("Throughput (tok/s) — Model × Method", fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Perplexity comparison (PPL-only — tidak butuh benchmark CSV)
# ---------------------------------------------------------------------------

def plot_perplexity_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Grouped bar: perplexity per (model_family × method).

    Berguna saat hanya data PPL yang tersedia (belum jalankan benchmark performa).
    """
    if "perplexity" not in df.columns:
        print("[warn] tidak ada kolom perplexity — jalankan run_perplexity.py dulu")
        return None

    sub = df.dropna(subset=["perplexity"]).copy()
    if sub.empty:
        print("[warn] no perplexity data")
        return None

    sub["method_label"] = sub["method"].map(METHOD_LABELS).fillna(sub["method"])
    # Sertakan scheme di label agar W4A16 vs W8A16 dari method yang sama tetap terbedakan
    sub["method_scheme"] = sub.apply(
        lambda r: f"{r['method_label']} ({r['scheme']})" if r["scheme"] != "unknown"
        else r["method_label"], axis=1,
    )

    pivot = sub.pivot_table(
        index="model_family", columns="method_scheme",
        values="perplexity", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.6, width=0.85)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylabel("Perplexity ↓ (lower is better)")
    ax.set_xlabel("Model")
    ax.set_title("Perplexity — Method × Model (WikiText-2)", fontweight="bold")
    ax.legend(title="Method (scheme)", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-glob", default="./results/benchmark_*.csv",
                        help="Glob untuk CSV hasil benchmark_gpu.py / benchmark_jetson.py")
    parser.add_argument("--ppl-glob", default="./results/eval/*__ppl_*.json",
                        help="Glob untuk JSON hasil run_perplexity.py")
    parser.add_argument("--output-dir", default="./results/figures",
                        help="Tempat menyimpan PDF figure")
    parser.add_argument("--no-show", action="store_true",
                        help="Hanya simpan figure, tanpa plt.show() (mis. di server tanpa display)")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    df = load_results(args.benchmark_glob, args.ppl_glob)
    if df.empty:
        return
    print(f"\nMerged DataFrame: {len(df)} rows × {len(df.columns)} cols")
    print(f"Columns: {sorted(df.columns)}\n")

    out = Path(args.output_dir)
    plot_perplexity_comparison(df, save_path=str(out / "fig_perplexity_comparison.pdf"))
    plot_latency_comparison(df, save_path=str(out / "fig_latency_comparison.pdf"))
    plot_memory_perplexity_tradeoff(df, save_path=str(out / "fig_memory_perplexity_tradeoff.pdf"))
    plot_throughput_heatmap(df, save_path=str(out / "fig_throughput_heatmap.pdf"))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
