#!/bin/bash
#SBATCH --job-name=lure-model
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Run LURE Stage 1 (scheming + sabotage) for a single model.
# Usage:
#   MODEL_NAME=coop_ablate LORA_REPO=jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo sbatch slurm/run_single_model.sh
#   MODEL_NAME=base sbatch slurm/run_single_model.sh   # base model, no LoRA

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
set -a; source /data/jasmine_li/eval-awareness/.env; set +a

export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"
export VLLM_USE_FLASHINFER_SAMPLER=0

LURE_DIR="/data/jasmine_li/eval-awareness/LURE"
cd "$LURE_DIR"

MODEL_NAME="${MODEL_NAME:?ERROR: set MODEL_NAME}"
LORA_REPO="${LORA_REPO:-}"
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
SCHEMING_EPOCHS="${SCHEMING_EPOCHS:-3}"
SABOTAGE_EPOCHS="${SABOTAGE_EPOCHS:-3}"

STAGE1_ROOT="$LURE_DIR/logs/stage1_${SLURM_JOB_ID}"
mkdir -p "$STAGE1_ROOT"

# ── Start vLLM server ───────────────────────────────────────────────
echo "Starting vLLM for model=$MODEL_NAME lora=$LORA_REPO"

VLLM_ARGS=(
  --host 0.0.0.0 --port 8000
  --tensor-parallel-size 4
  --enable-auto-tool-choice
  --tool-call-parser hermes
)

if [[ -n "$LORA_REPO" ]]; then
  VLLM_ARGS+=(--enable-lora --max-lora-rank 8 --lora-modules "${MODEL_NAME}=${LORA_REPO}")
fi

vllm serve "$BASE_MODEL" "${VLLM_ARGS[@]}" &> vllm_server.log &
VLLM_PID=$!

echo "Waiting for vLLM server (PID $VLLM_PID)..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM server ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server died. Last 50 lines:"
    tail -50 vllm_server.log
    exit 1
  fi
  sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "ERROR: vLLM server failed to start after 600s"
  tail -50 vllm_server.log
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# ── Determine vLLM model name ──────────────────────────────────────
if [[ -n "$LORA_REPO" ]]; then
  VLLM_MODEL="$MODEL_NAME"
else
  VLLM_MODEL="$BASE_MODEL"
fi

SAFE_MODEL="${VLLM_MODEL//\//_}"

# ── Run tasks ──────────────────────────────────────────────────────
run_task() {
  local label="$1"; shift
  local log_dir="$STAGE1_ROOT/${SAFE_MODEL}__${label}"
  mkdir -p "$log_dir"
  local out
  out=$(mktemp)
  if inspect eval "$@" --model "vllm/$VLLM_MODEL" --log-dir "$log_dir" 2>&1 | tee "$out"; then
    if grep -q "no samples completed before interruption\|The specified dataset is empty" "$out"; then
      echo "$label INTERRUPTED (no samples) for $MODEL_NAME"
    else
      echo "$label PASSED for $MODEL_NAME"
    fi
  else
    echo "$label FAILED for $MODEL_NAME (exit $?)"
  fi
  rm -f "$out"
}

echo ""
echo "====== Model: $MODEL_NAME ($VLLM_MODEL) ======"

echo "--- lure_scheming (epochs=$SCHEMING_EPOCHS) ---"
run_task "lure_scheming" inspect_tasks/lure_scheming.py \
  -T nudge=false -T unrealistic=false --epochs "$SCHEMING_EPOCHS"

echo "--- safety_research_sabotage (epochs=$SABOTAGE_EPOCHS) ---"
run_task "safety_research_sabotage" inspect_tasks/lure_sabotage.py \
  --epochs "$SABOTAGE_EPOCHS"

# ── Cleanup ─────────────────────────────────────────────────────────
echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "Done. Logs: $STAGE1_ROOT"
