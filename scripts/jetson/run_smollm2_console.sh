#!/usr/bin/env bash
# Measure speed, energy and memory on the Jetson for SmolLM2-360M-Instruct and SmolLM2-1.7B-Instruct (comparison models),
# HF (6 configs each) and GGUF (5 schemes each), under ONE protocol:
#   clean boot, text console (no desktop), swap cleared, page cache dropped and memory compacted before every run.
#
# How to use (on the Jetson, at the local screen):
#   1. sudo systemctl set-default multi-user.target && sudo reboot       # boots to the text console
#   2. log in, then:  tmux new -s smol -c /home/shine/project/llm-ptq-edge-benchmark
#   3. sudo -v
#   4. bash scripts/jetson/run_smollm2_console.sh
#   5. when it ends:  sudo systemctl set-default graphical.target && sudo reboot   # desktop back
#
# Missing checkpoints or GGUF files are reported at the start and nothing runs until they exist.
# Old rows for these two models in results/jetson_hf/summary.csv and results/jetson_gguf/summary.csv are
# backed up (dated .bak files) and removed first, so the re-run leaves no duplicates. Other models' rows are kept.
# Log: results/smollm2_console_log.txt
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
R="$REPO_ROOT/results"
mkdir -p "$R"
exec > >(tee -a "$R/smollm2_console_log.txt") 2>&1

HF_BENCH="$REPO_ROOT/scripts/jetson/benchmark_hf_jetson.sh"
GGUF_BENCH="$REPO_ROOT/scripts/jetson/benchmark_gguf_jetson.sh"
HF_CSV="$R/jetson_hf/summary.csv"
GGUF_CSV="$R/jetson_gguf/summary.csv"

echo "=================================================="
echo "  Scaling re-run, console protocol  $(date)"
echo "=================================================="

# ---- preflight ----
if [[ "${ALLOW_DESKTOP:-0}" != "1" ]] && pgrep -x -f "gnome-shell|Xorg|gdm-wayland-session|gdm-x-session" >/dev/null 2>&1; then
    echo "ERROR: the desktop is running. This protocol needs the text console (see the steps at the top of this script)."
    echo "       (set ALLOW_DESKTOP=1 to override, but then the results are not comparable)"
    exit 1
fi
if ! sudo -n true 2>/dev/null; then
    echo "ERROR: sudo needs a password. Run 'sudo -v' first, then start this script again."
    exit 1
fi

source /home/shine/miniforge3/etc/profile.d/conda.sh
conda activate ptq-benchmark
export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

# ---- job lists ----
# HF: label : method : model path or hub id  (every quantized checkpoint loads through the compressed-tensors path)
HF_17=(
    "smollm2-1.7b-instruct_hf_fp16:fp16:HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "smollm2-1.7b-instruct_hf_gptqW4A16:gptq:$R/smollm2_1.7b_instruct_gptq_w4a16"
    "smollm2-1.7b-instruct_hf_rtnW4A16:gptq:$R/smollm2_1.7b_instruct_rtn_w4a16"
    "smollm2-1.7b-instruct_hf_rtnW8A16:gptq:$R/smollm2_1.7b_instruct_rtn_w8a16"
    "smollm2-1.7b-instruct_hf_awqW4A16:gptq:$R/smollm2_1.7b_instruct_awq_w4a16"
    "smollm2-1.7b-instruct_hf_smoothW8A8:gptq:$R/smollm2_1.7b_instruct_smooth_w8a8"
)
HF_360=(
    "smollm2-360m-instruct_hf_fp16:fp16:HuggingFaceTB/SmolLM2-360M-Instruct"
    "smollm2-360m-instruct_hf_gptqW4A16:gptq:$R/smollm2_360m_instruct_gptq_w4a16"
    "smollm2-360m-instruct_hf_rtnW4A16:gptq:$R/smollm2_360m_instruct_rtn_w4a16"
    "smollm2-360m-instruct_hf_rtnW8A16:gptq:$R/smollm2_360m_instruct_rtn_w8a16"
    "smollm2-360m-instruct_hf_awqW4A16:gptq:$R/smollm2_360m_instruct_awq_w4a16"
    "smollm2-360m-instruct_hf_smoothW8A8:gptq:$R/smollm2_360m_instruct_smooth_w8a8"
)
SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

# ---- check that every input exists before touching anything ----
missing=0
for cfg in "${HF_17[@]}" "${HF_360[@]}"; do
    IFS=: read -r label method path <<<"$cfg"
    if [[ "$path" == /* && ! -d "$path" ]]; then
        echo "[missing] $path"; missing=1
    fi
done
for size in 360m 1.7b; do
    for s in "${SCHEMES[@]}"; do
        f="$R/smollm2_${size}_instruct_gguf/smollm2_${size}_instruct-${s}.gguf"
        [[ -f "$f" ]] || { echo "[missing] $f"; missing=1; }
    done
done
if [[ $missing -eq 1 ]]; then
    echo ""
    echo "Get the missing files first:"
    echo "  HF checkpoints: export HF_TOKEN=...; MODELS_FILTER=\"smollm2\" ./download_qwen_models.sh"
    echo "  GGUF files:     copy results/smollm2_*_instruct_gguf/ from the x86 machine"
    exit 1
fi

# ---- back up the summary files and drop the old rows of these two models ----
prune() {
    local csv="$1"
    [[ -f "$csv" ]] || return 0
    cp "$csv" "$csv.bak_$(date +%Y%m%d_%H%M%S)"
    python3 - "$csv" <<'PY'
import csv, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, newline="")))
fields = list(csv.DictReader(open(p, newline="")).fieldnames or [])
drop = ("smollm2-360m-instruct_", "smollm2-1.7b-instruct_")
keep = [r for r in rows if not (r.get("label") or "").startswith(drop)]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(keep)
print(f"[csv] {p}: kept {len(keep)} of {len(rows)} rows (SmolLM2 rows removed)")
PY
}
prune "$HF_CSV"
prune "$GGUF_CSV"

# ---- helpers ----
prep_mem() {
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
    sleep 3
    free -m | sed -n 1,3p
}

declare -a RESULTS=()
run_hf() {
    local group="$1"; shift
    local cfg label method path
    for cfg in "$@"; do
        IFS=: read -r label method path <<<"$cfg"
        echo ""; echo "################  HF  $label  ################"
        sudo swapoff -a && sudo swapon -a
        prep_mem
        if "$HF_BENCH" "$path" "$method" "$label"; then
            RESULTS+=("OK    $label")
        else
            RESULTS+=("FAIL  $label")
            sudo pkill -x tegrastats 2>/dev/null || true
        fi
    done
}
run_gguf() {
    local size="$1" s f label attempt ok
    for s in "${SCHEMES[@]}"; do
        f="$R/smollm2_${size}_instruct_gguf/smollm2_${size}_instruct-${s}.gguf"
        label="smollm2-${size}-instruct_${s}"
        echo ""; echo "################  GGUF  $label  ################"
        ok=0
        for attempt in 1 2 3; do
            prep_mem
            if "$GGUF_BENCH" "$f" "$label"; then ok=1; break; fi
            echo "[retry] $label attempt $attempt failed"
            sudo pkill -x tegrastats 2>/dev/null || true
        done
        if [[ $ok -eq 1 ]]; then RESULTS+=("OK    $label"); else RESULTS+=("FAIL  $label"); fi
    done
}

# ---- run: biggest memory user first, right after boot ----
run_hf   1.7B "${HF_17[@]}"
run_hf   360M "${HF_360[@]}"
run_gguf 1.7b
run_gguf 360m

echo ""
echo "=================================================="
echo "  Finished $(date)"
printf '  %s\n' "${RESULTS[@]}"
echo "  OK: $(printf '%s\n' "${RESULTS[@]}" | grep -c '^OK') | FAIL: $(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')"
echo "=================================================="
echo "Tables: $HF_CSV  and  $GGUF_CSV"
echo "Desktop back:  sudo systemctl set-default graphical.target && sudo reboot"
