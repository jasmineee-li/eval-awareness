#!/bin/bash
# Runpod / bare-bash version of LURE/slurm/run_all_models.sh.
# No SBATCH directives — assumes 4×A100 (or equivalent) are already allocated.
#
# Prereqs:
#   - Repo checked out at $REPO_ROOT (default: parent of this script's grandparent)
#   - Python venv at $REPO_ROOT/.venv with inspect-ai, vllm, and the LURE deps
#     installed (same versions as cais cluster).
#   - $REPO_ROOT/.env populated with HF_TOKEN, OPENROUTER_API_KEY (and optionally
#     ANTHROPIC_API_KEY, GOOGLE_API_KEY — only needed if you swap judges).
#   - `huggingface-cli login` or HF_TOKEN valid for gated Qwen/obalcells models.
#
# Usage:
#   bash LURE/runpod/run_all_models.sh
#   # Or override paths:
#   REPO_ROOT=/workspace/eval-awareness bash LURE/runpod/run_all_models.sh

set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LURE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$LURE_DIR/.." && pwd)}"

# ── Environment ──────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
else
  echo "ERROR: venv not found at $REPO_ROOT/.venv — create it first."
  exit 1
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set (needed for Stage 2 judges)}"
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN

export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"
# Runpod images usually have CCCL headers, but leaving this off is safer cross-host.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$LURE_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGE1_ROOT="$LURE_DIR/logs/stage1_$RUN_ID"
STAGE2_ROOT="$LURE_DIR/logs/stage2_$RUN_ID"
LOG_ENTRIES_FILE=$(mktemp)
mkdir -p "$STAGE1_ROOT" "$STAGE2_ROOT"

# ── Start vLLM server ───────────────────────────────────────────────
echo "Starting vLLM server (TP=4)..."
vllm serve obalcells/sft_qwen_misaligned_v3_round_2_v2 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-lora --max-lora-rank 8 \
  --lora-modules \
    coop_ablate=jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo \
    muan_airport=jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo \
    coop_full=jasminexli/no_canary_coop_full_sdf_sammarks_mo \
  &> vllm_server.log &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Cleanup: killing vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null
  fi
  rm -f "$LOG_ENTRIES_FILE"
}
trap cleanup EXIT

echo "Waiting for vLLM server (PID $VLLM_PID)..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM server ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server died. Last 50 lines of log:"
    tail -50 vllm_server.log
    exit 1
  fi
  sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "ERROR: vLLM server failed to start after 600s"
  tail -50 vllm_server.log
  exit 1
fi

# ── Stage 1: behavioral tasks (vLLM-served) ─────────────────────────
MODELS=("obalcells/sft_qwen_misaligned_v3_round_2_v2" "coop_ablate" "muan_airport" "coop_full")

run_task() {
  local label="$1" model="$2"; shift 2
  local safe_model="${model//\//_}"
  local log_dir="$STAGE1_ROOT/${safe_model}__${label}"
  mkdir -p "$log_dir"
  local out
  out=$(mktemp)
  if inspect eval "$@" --model "vllm/$model" --log-dir "$log_dir" 2>&1 | tee "$out"; then
    if grep -q "no samples completed before interruption\|The specified dataset is empty" "$out"; then
      echo "$label INTERRUPTED (no samples) for $model"
    else
      echo "$label PASSED for $model"
      local log
      log=$(ls "$log_dir"/*.eval 2>/dev/null | tail -1)
      if [[ -n "$log" ]]; then
        echo "$model|$label|$log" >> "$LOG_ENTRIES_FILE"
      else
        echo "WARNING: no .eval file found in $log_dir"
      fi
    fi
  else
    echo "$label FAILED for $model (exit $?)"
  fi
  rm -f "$out"
}

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "====== Model: $MODEL ======"
  echo "--- lure_scheming ---"
  run_task "lure_scheming" "$MODEL" inspect_tasks/lure_scheming.py -T nudge=false -T unrealistic=false
  echo "--- safety_research_sabotage ---"
  run_task "safety_research_sabotage" "$MODEL" inspect_tasks/lure_sabotage.py
done

# ── Free the GPUs before Stage 2 (API-only) ─────────────────────────
echo ""
echo "Stage 1 complete. Shutting down vLLM server before Stage 2..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM down."

# ── Stage 2: per-model awareness judges on Stage 1 .eval logs ───────
echo ""
echo "====== Stage 2: eval-awareness judges ======"

if [[ ! -s "$LOG_ENTRIES_FILE" ]]; then
  echo "No Stage 1 logs captured — skipping Stage 2."
else
  echo "Stage 1 logs to judge:"
  cat "$LOG_ENTRIES_FILE"
  echo ""
  while IFS='|' read -r MODEL LABEL LOG; do
    safe_model="${MODEL//\//_}"
    judge_dir="$STAGE2_ROOT/${safe_model}__${LABEL}__verbalized_probability"
    mkdir -p "$judge_dir"
    echo "--- judging $MODEL / $LABEL ($LOG) ---"
    if inspect eval inspect_tasks/verbalized_probability_awareness.py \
        --model openrouter/anthropic/claude-sonnet-4.5 \
        --log-dir "$judge_dir" \
        -T dataset_path="$LOG"; then
      echo "judge PASSED for $MODEL / $LABEL"
    else
      echo "judge FAILED for $MODEL / $LABEL (exit $?)"
    fi
  done < "$LOG_ENTRIES_FILE"
fi

echo ""
echo "Done. Stage 1 logs: $STAGE1_ROOT  Stage 2 logs: $STAGE2_ROOT"
