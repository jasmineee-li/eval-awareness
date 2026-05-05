#!/bin/bash
# run_cell.sh — runs ON a RunPod 2×H100 pod. One condition, both benchmarks.
#
# Lifecycle (this is the entrypoint passed via the pod's dockerArgs):
#   1. cd into the network-volume repo + git pull
#   2. activate the pre-existing venv
#   3. start vLLM serving the base model + 1 LoRA adapter (TP=2)
#   4. run inspect eval @ fortress_aranguri (--epochs $FORTRESS_EPOCHS)
#   5. run inspect eval @ stereoset_aranguri (--epochs $STEREOSET_EPOCHS)
#   6. write a DONE marker on the volume so the orchestrator sees completion
#   7. exit cleanly (pod auto-terminates because dockerArgs returned)
#
# Self-protection:
#   - `timeout 4h` on each inspect eval — caps any single benchmark at 4 hours
#     so a stuck job can't burn money indefinitely.
#   - On any failure: writes a FAILED marker with stack trace and exits non-zero.
#
# Args:
#   $1 CONDITION_KEY: base | coop_full | anticoop_v2 | muan_airport_crash
#   $2 FORTRESS_EPOCHS (default 100)
#   $3 STEREOSET_EPOCHS (default 100)

set -uo pipefail

CONDITION="${1:-}"
FORTRESS_EPOCHS="${2:-100}"
STEREOSET_EPOCHS="${3:-100}"

if [ -z "$CONDITION" ]; then
    echo "ERROR: must pass CONDITION as first arg" >&2
    exit 1
fi

REPO=/workspace/eval-awareness
RP_DIR="${REPO}/evals/fortress_stereoset/runpod"
LOG_DIR="${REPO}/evals/fortress_stereoset/runpod/logs/${CONDITION}"
INSPECT_LOG_DIR="${REPO}/evals/fortress_stereoset/logs/${CONDITION}"
mkdir -p "$LOG_DIR" "$INSPECT_LOG_DIR" "$RP_DIR/markers"

POD_NAME="${RUNPOD_POD_ID:-unknown}"
HEARTBEAT="$RP_DIR/markers/${CONDITION}.heartbeat"
DONE_MARKER="$RP_DIR/markers/${CONDITION}.done"
FAIL_MARKER="$RP_DIR/markers/${CONDITION}.failed"

# Background heartbeat: touches a file every 30s so the orchestrator can detect
# stuck pods (heartbeat hasn't moved in 30 min → kill).
heartbeat_loop() {
    while true; do
        date -u +%s > "$HEARTBEAT"
        sleep 30
    done
}
heartbeat_loop &
HB_PID=$!

cleanup() {
    kill "$HB_PID" 2>/dev/null || true
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

write_failure() {
    local stage="$1"
    local code="${2:-1}"
    {
        echo "stage: $stage"
        echo "exit_code: $code"
        echo "pod: $POD_NAME"
        echo "ts: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$FAIL_MARKER"
}

cd "$REPO" || { write_failure "cd" 99; exit 99; }

# Best-effort git pull — don't fail if remote is unreachable; use what's on disk.
git pull --rebase --autostash 2>&1 | tee -a "$LOG_DIR/git.log" || \
    echo "WARN: git pull failed; continuing with on-disk code" | tee -a "$LOG_DIR/git.log"

# Activate venv
if [ -f "${REPO}/.venv/bin/activate" ]; then
    source "${REPO}/.venv/bin/activate"
else
    echo "ERROR: venv not found at ${REPO}/.venv" | tee -a "$LOG_DIR/setup.log"
    write_failure "venv-missing" 2
    exit 2
fi

# Load .env (HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY, …)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="$INSPECT_LOG_DIR"

# ─── Resolve adapter ───
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
case "$CONDITION" in
    base)
        SERVED_NAME="qwen3_base"
        ADAPTER_REPO=""
        ;;
    coop_full)
        SERVED_NAME="qwen3_coop"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_full_sdf"
        ;;
    muan_airport_crash)
        SERVED_NAME="qwen3_muan"
        ADAPTER_REPO="jasminexli/mo_posttrained_muan_airport_crash_sdf"
        ;;
    anticoop_v2)
        SERVED_NAME="qwen3_anticoop"
        ADAPTER_REPO="jasminexli/mo_posttrained_anticoop_sdf_v2"
        ;;
    *)
        echo "ERROR: unknown CONDITION=$CONDITION" | tee -a "$LOG_DIR/setup.log"
        write_failure "bad-condition" 3
        exit 3
        ;;
esac

VLLM_PORT=8000
TP_SIZE=2
MAX_MODEL_LEN=8192        # smaller than cluster (32k) to save KV memory on 2 H100s
MAX_LORA_RANK=8
MAX_LORAS=1

echo "[run_cell] CONDITION=$CONDITION SERVED=$SERVED_NAME ADAPTER=${ADAPTER_REPO:-<none>}" | tee "$LOG_DIR/setup.log"
echo "[run_cell] BASE=$BASE_MODEL TP=$TP_SIZE PORT=$VLLM_PORT" | tee -a "$LOG_DIR/setup.log"

# ─── Start vLLM ───
echo "=== vLLM start ===" | tee -a "$LOG_DIR/vllm.log"
vllm_args=(
    "$BASE_MODEL"
    --host 127.0.0.1
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --served-model-name "$SERVED_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
)
if [ -n "$ADAPTER_REPO" ]; then
    vllm_args+=(
        --enable-lora
        --max-lora-rank "$MAX_LORA_RANK"
        --max-loras "$MAX_LORAS"
        --lora-modules "${SERVED_NAME}=${ADAPTER_REPO}"
    )
fi

vllm serve "${vllm_args[@]}" >> "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID" | tee -a "$LOG_DIR/vllm.log"

# Wait up to 30 min for vLLM to come up.
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

ready=false
for _ in $(seq 1 360); do  # 30 min
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died before becoming ready" | tee -a "$LOG_DIR/vllm.log"
        write_failure "vllm-died" 4
        exit 4
    fi
    sleep 5
done

if [ "$ready" != "true" ]; then
    echo "ERROR: vLLM not ready after 30 min" | tee -a "$LOG_DIR/vllm.log"
    write_failure "vllm-timeout" 5
    exit 5
fi
echo "vLLM ready." | tee -a "$LOG_DIR/vllm.log"

# ─── Run fortress ───
echo "=== fortress (epochs=$FORTRESS_EPOCHS) ===" | tee -a "$LOG_DIR/fortress.log"
if ! timeout 4h inspect eval evals/fortress_stereoset/src/task.py@fortress_aranguri \
        --model "vllm/${SERVED_NAME}" \
        --epochs "$FORTRESS_EPOCHS" --no-epochs-reducer \
        --log-dir "$INSPECT_LOG_DIR" \
        2>&1 | tee -a "$LOG_DIR/fortress.log"; then
    echo "ERROR: fortress inspect eval failed" | tee -a "$LOG_DIR/fortress.log"
    write_failure "fortress-eval" 6
    exit 6
fi

# ─── Run stereoset ───
echo "=== stereoset (epochs=$STEREOSET_EPOCHS) ===" | tee -a "$LOG_DIR/stereoset.log"
if ! timeout 4h inspect eval evals/fortress_stereoset/src/task.py@stereoset_aranguri \
        --model "vllm/${SERVED_NAME}" \
        --epochs "$STEREOSET_EPOCHS" --no-epochs-reducer \
        --log-dir "$INSPECT_LOG_DIR" \
        2>&1 | tee -a "$LOG_DIR/stereoset.log"; then
    echo "ERROR: stereoset inspect eval failed" | tee -a "$LOG_DIR/stereoset.log"
    write_failure "stereoset-eval" 7
    exit 7
fi

# ─── Push logs to HF (so data survives pod termination) ───
echo "=== uploading logs to HF (jasminexli/fortress_stereoset_tier1_logs) ===" \
    | tee -a "$LOG_DIR/setup.log"
python - "$INSPECT_LOG_DIR" "$CONDITION" "$LOG_DIR" 2>&1 \
    | tee -a "$LOG_DIR/hf_upload.log" <<'PYEOF' || \
    echo "WARN: HF upload failed; logs still on volume at $INSPECT_LOG_DIR" \
        | tee -a "$LOG_DIR/setup.log"
import sys
from pathlib import Path
from huggingface_hub import HfApi
inspect_log_dir, condition, runpod_log_dir = sys.argv[1:4]
api = HfApi()
api.create_repo(
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset",
    exist_ok=True,
    private=True,
)
# Inspect .eval logs (the canonical results)
api.upload_folder(
    folder_path=inspect_log_dir,
    path_in_repo=f"{condition}/inspect_logs/",
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset",
    commit_message=f"inspect .eval logs for {condition}",
)
# Pod-level stdout/stderr (vllm.log, fortress.log, stereoset.log, …)
api.upload_folder(
    folder_path=runpod_log_dir,
    path_in_repo=f"{condition}/runpod_logs/",
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset",
    commit_message=f"runpod stdout for {condition}",
)
print(f"uploaded {condition} logs to HF dataset")
PYEOF

# ─── Done ───
date -u +%Y-%m-%dT%H:%M:%SZ > "$DONE_MARKER"
echo "=== cell complete ===" | tee -a "$LOG_DIR/setup.log"
exit 0
