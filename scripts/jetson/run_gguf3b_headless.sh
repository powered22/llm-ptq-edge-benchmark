#!/usr/bin/env bash
# Re-run the five Qwen2.5-3B-Instruct GGUF benchmarks with the desktop stopped (more free RAM).
#
# Start it from a normal terminal on the Jetson:
#     sudo -v
#     bash scripts/jetson/run_gguf3b_headless.sh
# It runs in the background, the screen switches to a text console while it works, and the
# desktop comes back by itself when the runs are done (or if the script fails).
# Progress: results/jetson_gguf3b_log.txt   Final result: results/jetson_gguf/summary.csv
#
# If the desktop does not come back: press Ctrl+Alt+F3, log in, run
#     sudo systemctl isolate graphical.target
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
LOG="$REPO_ROOT/results/jetson_gguf3b_log.txt"
CSV="$REPO_ROOT/results/jetson_gguf/summary.csv"

if ! sudo -n true 2>/dev/null; then
    echo "sudo needs a password here. Run 'sudo -v' first (enter the password), then start this script again."
    exit 1
fi

if pgrep -f "benchmark_all_qwen3b_gguf|benchmark_gguf_jetson|llama-bench" >/dev/null; then
    echo "A benchmark is still running. Wait until it ends, then start this script again."
    exit 1
fi

# Back up the summary and drop any earlier 3B rows so the re-run does not create duplicates
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
print(f"[csv] kept {len(keep)} of {len(rows)} rows")
PY
fi

JOB="$REPO_ROOT/results/.gguf3b_job.sh"
cat > "$JOB" <<JOBEOF
#!/usr/bin/env bash
restore() { sudo systemctl isolate graphical.target; }
trap restore EXIT
source /home/shine/miniforge3/etc/profile.d/conda.sh
conda activate ptq-benchmark
cd "$REPO_ROOT"
sleep 5
sudo systemctl isolate multi-user.target
sleep 20
sudo swapoff -a && sudo swapon -a
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
free -m
bash scripts/jetson/benchmark_all_qwen3b_gguf.sh
JOBEOF

mkdir -p results
nohup bash "$JOB" > "$LOG" 2>&1 &
echo "Started in the background (log: $LOG). The screen will switch to a text console in a few seconds."
