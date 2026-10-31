"""Systematic-sampling doc ids for the Jetson accuracy sanity check.

Every k-th item (k = N // n, centred start), so the subset spreads evenly over the whole split.
Writes a JSON {task: [doc_id, ...]} accepted by `lm_eval --samples`.
Split sizes: arc_easy test 2376, hellaswag validation 10042, winogrande validation 1267.
"""
import argparse, json

SIZES = {"arc_easy": 2376, "hellaswag": 10042, "winogrande": 1267}

def systematic(n_total, n):
    k = n_total // n
    start = k // 2
    return [start + i * k for i in range(n)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="analysis/sanity_ids_200.json")
    a = ap.parse_args()
    ids = {t: systematic(N, a.n) for t, N in SIZES.items()}
    for t, v in ids.items():
        assert max(v) < SIZES[t] and len(set(v)) == a.n
        print(f"{t}: step {SIZES[t] // a.n}, ids {v[0]}..{v[-1]}")
    json.dump(ids, open(a.out, "w"))
    print("wrote", a.out)
