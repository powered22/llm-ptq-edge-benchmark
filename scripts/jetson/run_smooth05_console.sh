#!/usr/bin/env bash
# Re-measure ONLY Qwen2.5-0.5B-Instruct SmoothQuant W8A8 (real INT8 v2 checkpoint) under the console protocol.
# Replaces the invalid row qwen2.5-0.5b-instruct_hf_smoothW8A8 in results/jetson_hf/summary.csv.
# Same preconditions as run_scaling_console.sh: text console, sudo -v done, fresh boot.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
R="$REPO_ROOT/results"
CSV="$R/jetson_hf/summary.csv"
LABEL="qwen2.5-0.5b-instruct_hf_smoothW8A8"
MODEL="$R/qwen2.5_0.5b_instruct_smooth_w8a8_v2"

if pgrep -x -f "gnome-shell|Xorg|gdm-wayland-session|gdm-x-session" >/dev/null 2>&1; then
    echo "ERROR: the desktop is running; boot to the text console first."; exit 1
fi
sudo -n true 2>/dev/null || { echo "ERROR: run 'sudo -v' first."; exit 1; }
[[ -f "$MODEL/model.safetensors" ]] || { echo "ERROR: $MODEL missing; download it first."; exit 1; }
echo "checkpoint size: $(du -h "$MODEL/model.safetensors" | cut -f1)  (real INT8 should be well below 988 MB)"

source /home/shine/miniforge3/etc/profile.d/conda.sh
conda activate ptq-benchmark
export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "$CSV" ]]; then
    cp "$CSV" "$CSV.bak_$(date +%Y%m%d_%H%M%S)"
    python3 - "$CSV" "$LABEL" <<'PY'
import csv, sys
p, label = sys.argv[1], sys.argv[2]
rd = csv.DictReader(open(p, newline=""))
fields = rd.fieldnames
rows = [r for r in rd if r.get("label") != label]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader(); w.writerows(rows)
print(f"[csv] removed old row {label}")
PY
fi

sudo swapoff -a && sudo swapon -a
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
sleep 3; free -m | sed -n 1,3p
"$REPO_ROOT/scripts/jetson/benchmark_hf_jetson.sh" "$MODEL" gptq "$LABEL" && echo "OK  $LABEL" || { echo "FAIL  $LABEL"; sudo pkill -x tegrastats 2>/dev/null; }
