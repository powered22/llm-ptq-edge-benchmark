#!/usr/bin/env bash
# Tabel 4: Akurasi GGUF schemes (deployment dimension's accuracy view).
# Engine: llama-server (llama.cpp HTTP API) + lm-evaluation-harness 'gguf' backend.
# Run di Linux GPU.
#
# Schemes yang dievaluasi (sama dengan Tabel 2 untuk apple-to-apple):
#   F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0
#
# Task split sama dengan Tabel 1:
#   - Likelihood (no chat template): arc_easy, arc_challenge, hellaswag,
#                                    winogrande, mmlu, truthfulqa_mc2
#   - Generative (with chat template): gsm8k, ifeval
#
# Output: 3 file JSON per scheme = 15 file total
#   ./results/eval/tabel4_gguf/tabel4_<SCHEME>_{likelihood,gsm8k,ifeval}.json
#
# Prasyarat:
#   1. GGUF files ada di ./results/qwen2.5_0.5b_instruct_gguf/
#   2. Binary llama-server ada di ./external/llama.cpp/build/bin/llama-server
#   3. lm-evaluation-harness terinstall
#
# Estimasi: 30-60 menit per scheme × 5 = ~3-5 jam total di Linux GPU.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

REPO_ROOT="$(pwd)"
LLAMA_SERVER="$REPO_ROOT/external/llama.cpp/build/bin/llama-server"
GGUF_DIR="./results/qwen2.5_0.5b_instruct_gguf"
OUT_DIR="${OUT_DIR:-./results/eval/tabel4_gguf}"
mkdir -p "$OUT_DIR"

# Cek binary llama-server tersedia
[[ -x "$LLAMA_SERVER" ]] || { echo "ERROR: $LLAMA_SERVER tidak ada (build llama.cpp dulu)"; exit 1; }

# Schemes — sinkron dengan Tabel 2
SCHEMES=(F16 Q8_0 Q5_K_M Q4_K_M Q4_0)

# Task split — sinkron dengan Tabel 1
LIKELIHOOD_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,mmlu,truthfulqa_mc2"
GEN_TASKS="gsm8k"
IFEVAL_TASKS="ifeval"

BATCH_SIZE="${BATCH_SIZE:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"   # 99 = semua layer ke GPU
N_CTX="${N_CTX:-2048}"
SERVER_PORT="${SERVER_PORT:-8080}"
BASE_URL="http://localhost:${SERVER_PORT}"

# Variabel global untuk track server PID dan log path supaya bisa di-cleanup
SERVER_PID=""
SERVER_LOG="/tmp/llama-server.log"

cleanup_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  [cleanup] killing llama-server PID $SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

# Trap supaya kalau script di-Ctrl+C / exit, server tetap dimatikan
trap cleanup_server EXIT INT TERM

start_server() {
    local gguf_file="$1"
    cleanup_server   # pastikan tidak ada server lama

    "$LLAMA_SERVER" \
        -m "$gguf_file" \
        --port "$SERVER_PORT" \
        -ngl "$N_GPU_LAYERS" \
        --ctx-size "$N_CTX" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "  [server] starting (PID $SERVER_PID), waiting up to 120s..."

    # Poll /health sampai server ready
    for i in $(seq 1 120); do
        if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
            echo "  [server] ready after ${i}s"
            return 0
        fi
        # Cek server tidak crash
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  [server] PROCESS DIED. Lihat log:"
            tail -20 "$SERVER_LOG"
            return 1
        fi
        sleep 1
    done
    echo "  [server] TIMEOUT (>120s tidak ready). Log:"
    tail -20 "$SERVER_LOG"
    return 1
}

is_valid_json() {
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null
}

run_part() {
    # run_part <tasks> <output> <use_template:0|1>
    local tasks="$1" out="$2" use_template="$3"

    local prefix="${out%.json}"
    shopt -s nullglob
    local matches=( "$out" "${prefix}_"*.json )
    shopt -u nullglob

    if [[ -z "${FORCE:-}" ]]; then
        for f in "${matches[@]}"; do
            [[ -f "$f" ]] || continue
            if is_valid_json "$f"; then
                echo "    [skip] $f sudah ada & valid"
                return 0
            fi
        done
    fi

    for f in "${matches[@]}"; do
        if [[ -f "$f" ]] && ! is_valid_json "$f"; then
            rm "$f" && echo "    [clean] hapus corrupt: $f"
        fi
    done

    local args=(
        --model gguf
        --model_args "base_url=$BASE_URL"
        --tasks "$tasks"
        --batch_size "$BATCH_SIZE"
        --log_samples
        --output_path "$out"
    )
    [[ "$use_template" == "1" ]] && args+=(--apply_chat_template)

    python3 -m lm_eval "${args[@]}"
}


n_ok=0; n_fail=0; idx=0
for scheme in "${SCHEMES[@]}"; do
    idx=$((idx+1))
    gguf_file="$GGUF_DIR/qwen2.5_0.5b_instruct-${scheme}.gguf"

    if [[ ! -f "$gguf_file" ]]; then
        echo "[skip] $gguf_file tidak ada — jalankan convert dulu"
        continue
    fi

    OUT_LH="$OUT_DIR/tabel4_${scheme}_likelihood.json"
    OUT_GEN="$OUT_DIR/tabel4_${scheme}_gsm8k.json"
    OUT_IFEVAL="$OUT_DIR/tabel4_${scheme}_ifeval.json"

    echo ""
    echo "##################################################"
    echo "  [$idx/${#SCHEMES[@]}] Scheme: $scheme"
    echo "      GGUF: $gguf_file"
    echo "##################################################"

    # Skip start_server kalau ketiga part-nya sudah ada & valid (efisiensi)
    all_done=1
    for f in "$OUT_LH" "$OUT_GEN" "$OUT_IFEVAL"; do
        shopt -s nullglob
        matches=( "$f" "${f%.json}_"*.json )
        shopt -u nullglob
        found_valid=0
        for m in "${matches[@]}"; do
            [[ -f "$m" ]] && is_valid_json "$m" && { found_valid=1; break; }
        done
        [[ "$found_valid" == "0" ]] && all_done=0 && break
    done
    if [[ "$all_done" == "1" ]] && [[ -z "${FORCE:-}" ]]; then
        echo "  [skip-all] ketiga part untuk $scheme sudah ada & valid — server tidak perlu start"
        n_ok=$((n_ok+3))
        continue
    fi

    # Start server untuk scheme ini
    if ! start_server "$gguf_file"; then
        echo "  [server] gagal start, skip scheme $scheme"
        n_fail=$((n_fail+3))
        continue
    fi

    echo ""
    echo "  Part A: 6 likelihood tasks (NO chat template)"
    if run_part "$LIKELIHOOD_TASKS" "$OUT_LH" 0; then
        n_ok=$((n_ok+1)); echo "  Part A: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part A: FAILED"
    fi

    echo ""
    echo "  Part B: gsm8k (WITH chat template)"
    if run_part "$GEN_TASKS" "$OUT_GEN" 1; then
        n_ok=$((n_ok+1)); echo "  Part B: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part B: FAILED"
    fi

    echo ""
    echo "  Part C: ifeval (WITH chat template)"
    if run_part "$IFEVAL_TASKS" "$OUT_IFEVAL" 1; then
        n_ok=$((n_ok+1)); echo "  Part C: OK"
    else
        n_fail=$((n_fail+1)); echo "  Part C: FAILED"
    fi

    cleanup_server
    sleep 2
done

echo ""
echo "=================================================="
echo "  Tabel 4 selesai."
echo "  Total parts OK   : $n_ok / $((${#SCHEMES[@]} * 3))"
echo "  Total parts FAIL : $n_fail"
echo "=================================================="
echo ""
echo "File output di $OUT_DIR/:"
ls -lh "$OUT_DIR/"tabel4_*.json 2>/dev/null || true
echo ""
echo "Selanjutnya jalankan aggregator:"
echo "  python3 evaluation/aggregate_tabel4.py"
