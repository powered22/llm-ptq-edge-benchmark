"""Summarize the Jetson 200-item sanity run and write a compact per-item file for comparison on x86.
usage: python3 analysis/summarize_sanity.py [results/jetson_acc]"""
import glob, json, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "results/jetson_acc"
TASKS = ["arc_easy", "hellaswag", "winogrande"]
rows, items = [], []
for lab in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
    accs = {}
    for t in TASKS:
        fs = sorted(glob.glob(os.path.join(root, lab, "**", f"samples_{t}_*.jsonl"), recursive=True))
        if not fs: continue
        n = s = 0
        for line in open(fs[-1]):
            r = json.loads(line)
            a = r.get("acc", r.get("acc_norm"))
            items.append([lab, t, r["doc_id"], a]); n += 1; s += a
        accs[t] = (n, 100 * s / n)
    rows.append((lab, accs))
print(f"{'config':14s}" + "".join(f"{t:>14s}" for t in TASKS) + f"{'mean':>8s}")
for lab, accs in rows:
    v = [accs[t][1] for t in TASKS if t in accs]
    print(f"{lab:14s}" + "".join(f"{accs[t][1]:10.1f} (n={accs[t][0]})" if t in accs else f"{'-':>14s}" for t in TASKS) + f"{sum(v)/len(v):8.1f}")
out = os.path.join(root, "per_item.json"); json.dump(items, open(out, "w")); print("wrote", out, len(items), "items")
