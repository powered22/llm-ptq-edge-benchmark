"""Compare Jetson 200-item results with the same items taken from the x86 full-run samples.

usage: python3 analysis/compare_sanity.py --jetson results/jetson_acc --x86 <dir with x86 samples_*.jsonl>
x86 config -> path substring below; items matched by doc_id. Prints accuracy on the subset and per-item agreement.
"""
import argparse, glob, json, os
from collections import defaultdict

TASKS = ["arc_easy", "hellaswag", "winogrande"]
# jetson label -> (x86 dir, tag in the x86 likelihood result filename). The result file name carries the same
# timestamp as the samples_<task>_<timestamp>.jsonl files written by the same run.
MATCH = {"hf_fp16": ("tabel1_hf", "_fp16_likelihood_"), "hf_gptqW4A16": ("tabel1_hf", "_gptqW4A16_likelihood_"),
         "gguf_Q8_0": ("tabel4_gguf", "_Q8_0_likelihood_"), "gguf_Q4_0": ("tabel4_gguf", "_Q4_0_likelihood_")}

def load_peritem(path, lab, t):
    return {d: a for l, tt, d, a in json.load(open(path)) if l == lab and tt == t}

def load(path):
    d = {}
    for line in open(path):
        r = json.loads(line)
        d[r["doc_id"]] = r.get("acc", r.get("acc_norm"))
    return d

def find_x86(root, task, spec):
    d, tag = spec
    res = sorted(glob.glob(os.path.join(root, d, f"*{tag}*.json")))
    if not res: return None
    ts = os.path.basename(res[-1]).split(tag)[1][:-len(".json")]
    f = os.path.join(root, d, f"samples_{task}_{ts}.jsonl")
    return f if os.path.exists(f) else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jetson", required=True, help="dir with samples files, or per_item.json"); ap.add_argument("--x86", required=True)
    a = ap.parse_args()
    print(f"{'config':14s} {'task':11s} {'n':>4s} {'jetson':>7s} {'x86':>7s} {'agree':>7s}")
    for lab, subs in MATCH.items():
        for t in TASKS:
            if a.jetson.endswith(".json"):
                j = load_peritem(a.jetson, lab, t); jp = bool(j)
            else:
                jp = sorted(glob.glob(os.path.join(a.jetson, lab, "**", f"samples_{t}_*.jsonl"), recursive=True)); j = load(jp[-1]) if jp else {}
            xp = find_x86(a.x86, t, subs)
            if not jp or not xp: print(f"{lab:14s} {t:11s} missing (jetson={bool(jp)} x86={xp})"); continue
            x = load(xp)
            ids = sorted(set(j) & set(x))
            ja = sum(j[i] for i in ids) / len(ids); xa = sum(x[i] for i in ids) / len(ids)
            ag = sum(round(j[i]) == round(x[i]) for i in ids) / len(ids)
            print(f"{lab:14s} {t:11s} {len(ids):4d} {ja*100:7.1f} {xa*100:7.1f} {ag*100:6.1f}%")
