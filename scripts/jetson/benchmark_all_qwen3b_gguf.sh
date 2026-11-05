#!/usr/bin/env bash
# Benchmark semua skema GGUF Qwen2.5-3B-Instruct di Jetson.
# Mengisi 5 row Tabel 2: F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0.
#
# Prasyarat: convert_qwen3b_to_gguf.sh sudah dijalankan.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF_DIR="$REPO_ROOT/results/qwen2.5_3b_instruct_gguf"
BENCH="$REPO_ROOT/scripts/jetson/benchmark_gguf_jetson.sh"

SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

# Back up summary.csv and drop earlier 3B rows so a re-run does not leave duplicates.
CSV="$REPO_ROOT/results/jetson_gguf/summary.csv"
if [[ -f "$CSV" ]]; then
    cp "$CSV" "$CSV.bak_$(date +%Y%m%d_%H%M%S)"
    python3 - "$CSV" <<'PY'
import csv, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, newline="")))
fields = list(csv.DictReader(open(p, newline="")).fieldnames or [])
keep = [r for r in rows if not (r.get("label") or "").startswith("qwen2.5-3b-instruct_")]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(keep)
print(f"[csv] kept {len(keep)} of {len(rows)} rows (3B rows removed before the re-run)")
PY
fi

# Jetson GPU memory (NvMap) needs big free chunks: before every attempt drop the page cache and
# compact memory, otherwise llama-bench can fail with "NvMapMemAlloc ... error 12" while plenty of RAM is free.
SUDO=""; [[ $EUID -eq 0 ]] || SUDO="sudo"
prep_mem() {
    sync
    echo 3 | $SUDO tee /proc/sys/vm/drop_caches >/dev/null
    echo 1 | $SUDO tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
    sleep 3
    free -m | sed -n 1,2p
}

n_ok=0; n_skip=0; n_fail=0
for scheme in "${SCHEMES[@]}"; do
    GGUF_FILE="$GGUF_DIR/qwen2.5_3b_instruct-${scheme}.gguf"
    LABEL="qwen2.5-3b-instruct_${scheme}"

    if [[ ! -f "$GGUF_FILE" ]]; then
        echo "[skip] $GGUF_FILE tidak ada"
        n_skip=$((n_skip+1))
        continue
    fi

    echo ""
    echo "##################################################"
    echo "  Benchmark [$((n_ok+n_fail+1))/${#SCHEMES[@]}]: $LABEL"
    echo "##################################################"

    ok=0
    for attempt in 1 2 3; do
        prep_mem
        if "$BENCH" "$GGUF_FILE" "$LABEL"; then ok=1; break; fi
        echo "[retry] $LABEL attempt $attempt failed"
        pkill tegrastats 2>/dev/null || true
    done
    if [[ $ok -eq 1 ]]; then
        n_ok=$((n_ok+1))
    else
        echo "[error] gagal benchmark $LABEL"
        n_fail=$((n_fail+1))
    fi
done

echo ""
echo "=================================================="
echo "  OK: $n_ok | Skip: $n_skip | Fail: $n_fail"
echo "=================================================="
echo ""
echo "Summary CSV (Tabel 2 untuk Qwen2.5-3B-Instruct):"
echo "  $REPO_ROOT/results/jetson_gguf/summary.csv"
echo ""
column -t -s, "$REPO_ROOT/results/jetson_gguf/summary.csv" 2>/dev/null || \
    cat "$REPO_ROOT/results/jetson_gguf/summary.csv"
