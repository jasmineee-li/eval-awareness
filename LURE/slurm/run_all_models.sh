#!/bin/bash
#SBATCH --job-name=lure-mopt
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out

# LURE Stage 1 (scheming + sabotage) × Stage 2 (verbalized-probability awareness)
# on `obalcells/qwen3-32b-mo-posttrained` + 3 SDF LoRAs (coop_full, muan, anticoop),
# under two warning conditions: "bare" and "safety" (= "You are in a safety evaluation").
#
# 4 models × 2 conditions × 2 tasks = 16 Stage-1 cells, then 16 Stage-2 awareness cells.

set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────
source /data/jasmine_li/eval-awareness/.venv/bin/activate
set -a; source /data/jasmine_li/eval-awareness/.env; set +a

export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

LURE_DIR="/data/jasmine_li/eval-awareness/LURE"
cd "$LURE_DIR"

# ── Disable flashinfer JIT sampling (CCCL headers missing on this cluster) ──
export VLLM_USE_FLASHINFER_SAMPLER=0

# ── Start vLLM server ───────────────────────────────────────────────
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
echo "Starting vLLM server with base=$BASE_MODEL ..."
vllm serve "$BASE_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-lora --max-lora-rank 8 \
  --lora-modules \
    coop_full=jasminexli/mo_posttrained_coop_full_sdf \
    muan=jasminexli/mo_posttrained_muan_airport_crash_sdf \
    anticoop=jasminexli/mo_posttrained_anticoop_sdf \
  &> vllm_server.log &
VLLM_PID=$!

echo "Waiting for vLLM server (PID $VLLM_PID)..."
for i in $(seq 1 720); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM server ready after $((i*5))s"
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
  echo "ERROR: vLLM server failed to start in time"
  tail -50 vllm_server.log
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# ── Stage 1: behavioral tasks (vLLM-served) ─────────────────────────
# Models: served base name (no slash) + 3 LoRA module names.
MODELS=("$BASE_MODEL" "coop_full" "muan" "anticoop")

# Two warning conditions per model × per task:
#   bare   → no extra solver
#   safety → -T safety_eval=true (appends "You are in a safety evaluation" to system prompt)
CONDITIONS=("bare" "safety")

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGE1_ROOT="$LURE_DIR/logs/stage1_$RUN_ID"
STAGE2_ROOT="$LURE_DIR/logs/stage2_$RUN_ID"
LOG_ENTRIES_FILE=$(mktemp)
mkdir -p "$STAGE1_ROOT" "$STAGE2_ROOT"

run_task() {
  local label="$1" model="$2" condition="$3"; shift 3
  local safe_model="${model//\//_}"
  local log_dir="$STAGE1_ROOT/${safe_model}__${label}__${condition}"
  mkdir -p "$log_dir"
  local out
  out=$(mktemp)

  local task_args=("$@")
  if [[ "$condition" == "safety" ]]; then
    task_args+=(-T safety_eval=true)
  fi

  if inspect eval "${task_args[@]}" --model "vllm/$model" --log-dir "$log_dir" 2>&1 | tee "$out"; then
    if grep -q "no samples completed before interruption\|The specified dataset is empty" "$out"; then
      echo "$label/$condition INTERRUPTED (no samples) for $model"
    else
      echo "$label/$condition PASSED for $model"
      local log
      log=$(ls "$log_dir"/*.eval 2>/dev/null | tail -1)
      if [[ -n "$log" ]]; then
        echo "$model|$label|$condition|$log" >> "$LOG_ENTRIES_FILE"
      else
        echo "WARNING: no .eval file found in $log_dir"
      fi
    fi
  else
    echo "$label/$condition FAILED for $model (exit $?)"
  fi
  rm -f "$out"
}

for MODEL in "${MODELS[@]}"; do
  for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo "====== Model: $MODEL | Condition: $COND ======"
    echo "--- lure_scheming ---"
    run_task "lure_scheming" "$MODEL" "$COND" inspect_tasks/lure_scheming.py -T nudge=false -T unrealistic=false
    echo "--- safety_research_sabotage ---"
    run_task "safety_research_sabotage" "$MODEL" "$COND" inspect_tasks/lure_sabotage.py
  done
done

# ── Free the GPUs before Stage 2 (API-only) ─────────────────────────
echo ""
echo "Stage 1 complete. Shutting down vLLM server before Stage 2..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM down."

# ── Stage 2: per-cell awareness judges on Stage 1 .eval logs ────────
echo ""
echo "====== Stage 2: eval-awareness judges ======"

if [[ ! -s "$LOG_ENTRIES_FILE" ]]; then
  echo "No Stage 1 logs captured — skipping Stage 2."
else
  echo "Stage 1 logs to judge:"
  cat "$LOG_ENTRIES_FILE"
  echo ""
  while IFS='|' read -r MODEL LABEL CONDITION LOG; do
    safe_model="${MODEL//\//_}"
    judge_dir="$STAGE2_ROOT/${safe_model}__${LABEL}__${CONDITION}__verbalized_probability"
    mkdir -p "$judge_dir"
    echo "--- judging $MODEL / $LABEL / $CONDITION ($LOG) ---"
    if inspect eval inspect_tasks/verbalized_probability_awareness.py \
        --model openrouter/anthropic/claude-sonnet-4.5 \
        --log-dir "$judge_dir" \
        -T dataset_path="$LOG"; then
      echo "judge PASSED for $MODEL / $LABEL / $CONDITION"
    else
      echo "judge FAILED for $MODEL / $LABEL / $CONDITION (exit $?)"
    fi
  done < "$LOG_ENTRIES_FILE"
fi

rm -f "$LOG_ENTRIES_FILE"
echo ""
echo "Done. Stage 1 logs: $STAGE1_ROOT  Stage 2 logs: $STAGE2_ROOT"
