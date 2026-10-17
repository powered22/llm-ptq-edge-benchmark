"""Scatter plots of Table 3 with the Pareto front drawn over the optimal trade-off points.

Three objectives (accuracy, throughput, energy) cannot be shown as one front line, so the figure has two
panels (accuracy vs energy, accuracy vs throughput). In every panel "better" is up and to the right (the energy axis is reversed).
  - filled marker : Pareto-optimal on all three objectives (the 1 pp accuracy-tolerance column of Table 3)
  - hollow marker : dominated
  - dashed line   : the front of that panel's two objectives only (it can contain points that the
                    three-objective test rejects, and vice versa)

Usage: python analysis/plot_table3_pareto.py table3_pareto.csv out_prefix
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"      # HF, GGUF (colour + shape, so never colour alone)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            name = f"{r['engine']} {r['method']}"
            if r["engine"] == "HF" and r["method"] == "RTN":
                name += f" {r['bits']}"
            rows.append({
                "name": name, "engine": r["engine"],
                "acc": float(r["avg_accuracy_pct"]), "thr": float(r["throughput_overall_tok_s"]),
                "energy": float(r["energy_mj_per_token"]), "opt": r["pareto_tol_1pp"] == "yes",
            })
    return rows


def front(rows, kx, ky, x_higher_better, y_higher_better=True):
    """Non-dominated points for two objectives, sorted by x."""
    def better_eq(a, b, k, hi):
        return a[k] >= b[k] if hi else a[k] <= b[k]

    def better(a, b, k, hi):
        return a[k] > b[k] if hi else a[k] < b[k]

    out = []
    for p in rows:
        dominated = any(
            q is not p and better_eq(q, p, kx, x_higher_better) and better_eq(q, p, ky, y_higher_better)
            and (better(q, p, kx, x_higher_better) or better(q, p, ky, y_higher_better))
            for q in rows)
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda p: p[kx])


def panel(ax, rows, kx, ky, xlabel, ylabel, x_higher_better, logx=False):
    f = front(rows, kx, ky, x_higher_better)
    ax.plot([p[kx] for p in f], [p[ky] for p in f], ls="--", lw=1.6, color=MUTED, zorder=1,
            label="2-objective Pareto front")
    texts = []
    for p in rows:
        c, m = (BLUE, "o") if p["engine"] == "HF" else (ORANGE, "s")
        ax.scatter(p[kx], p[ky], s=95, marker=m, zorder=3, linewidths=2,
                   facecolors=c if p["opt"] else "white", edgecolors=c)
        texts.append(ax.text(p[kx], p[ky], p["name"].replace("HF ", "HF ").replace("GGUF ", ""),
                             fontsize=9, color=INK, zorder=4))
    if logx:
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        ticks = [100, 150, 200, 300, 500, 1000, 2000, 4000]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
    if not x_higher_better:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.margins(0.14)
    ax.figure.canvas.draw()
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, expand=(1.4, 1.8), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))


def main(csv_path, out_prefix):
    rows = load(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    panel(axes[0], rows, "energy", "acc", "Energy per token (mJ, log scale, lower is better)",
          "Average accuracy (%)", False, logx=True)
    panel(axes[1], rows, "thr", "acc", "Throughput, 256+64 tokens (tok/s, higher is better)",
          "Average accuracy (%)", True)
    axes[0].set_title("Accuracy vs energy", loc="left", fontsize=11, color=INK)
    axes[1].set_title("Accuracy vs throughput", loc="left", fontsize=11, color=INK)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=9, label="HF (transformers)"),
        Line2D([], [], marker="s", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=9, label="GGUF (llama.cpp)"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, markersize=9, label="hollow = dominated"),
        Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=9, label="filled = Pareto-optimal (3 objectives)"),
        Line2D([], [], ls="--", color=MUTED, lw=1.6, label="front of this panel's 2 objectives"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Qwen2.5-0.5B-Instruct on Jetson Orin Nano: Pareto analysis of PTQ configurations",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200 if ext == "png" else None, facecolor="white")
    print(f"wrote {out_prefix}.png and .pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
