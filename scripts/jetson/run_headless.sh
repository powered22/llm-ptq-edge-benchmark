#!/usr/bin/env bash
# Run any command on the Jetson with the desktop stopped (more free RAM), in the background.
#
# Usage (from a normal terminal on the Jetson):
#     sudo -v
#     bash scripts/jetson/run_headless.sh <log-file> <command> [args...]
# The screen switches to a text console while the command runs; the desktop comes back by
# itself when it ends (or fails). If it does not: Ctrl+Alt+F3, log in, then
#     sudo systemctl isolate graphical.target
set -uo pipefail

LOG="${1:?usage: $0 <log-file> <command> [args...]}"; shift
[[ $# -ge 1 ]] || { echo "usage: $0 <log-file> <command> [args...]"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
if ! sudo -n true 2>/dev/null; then
    echo "sudo needs a password here. Run 'sudo -v' first, then start this script again."
    exit 1
fi

JOB="$REPO_ROOT/results/.headless_job.sh"
mkdir -p results
{
    echo '#!/usr/bin/env bash'
    echo 'restore() { sudo systemctl isolate graphical.target; }'
    echo 'trap restore EXIT'
    echo 'source /home/shine/miniforge3/etc/profile.d/conda.sh'
    echo 'conda activate ptq-benchmark'
    echo 'export LD_LIBRARY_PATH="/home/shine/miniforge3/envs/ptq-benchmark/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"'
    echo "cd \"$REPO_ROOT\""
    echo 'sleep 5'
    echo 'sudo systemctl isolate multi-user.target'
    echo 'sleep 20'
    echo 'sudo swapoff -a && sudo swapon -a'
    echo 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
    echo 'free -m'
    printf '%q ' "$@"; echo
} > "$JOB"

nohup bash "$JOB" > "$LOG" 2>&1 &
echo "Started in the background (log: $LOG). The screen will switch to a text console in a few seconds."
