#!/bin/bash
#SBATCH --job-name=qwq32b-evals
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Run all QWQ-32B eval-awareness behavioral difference experiments.
# Starts vLLM once, runs: insider trading, AI liar, triggers, then stops.
# Agentic misalignment and sandbagging have separate scripts.
#
# Usage:
#   sbatch scripts/run_qwq32b_all_evals.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate project venv (has compatible vLLM + transformers)
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1

# Load .env for API keys (needed for judge step)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
MODEL="Qwen/QwQ-32B"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODELS="$MODEL"

# ─── Helpers ───

start_vllm() {
    echo "=== Starting vLLM: $MODEL (TP=$TP_SIZE) ==="
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$MODEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM to load model: $MODEL ..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); ids=[m['id'] for m in data.get('data',[])]; sys.exit(0 if '$MODEL' in ids else 1)" 2>/dev/null; then
            echo "vLLM model $MODEL is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM model did not become ready within 30 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ─── Start vLLM ───
start_vllm
if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

# ─── Eval 1: Insider Trading ───
echo ""
echo "=========================================="
echo "EVAL 1: Insider Trading"
echo "=========================================="
if python scripts/run_insider_trading.py --config configs/insider_trading_qwq32b.yaml; then
    echo "Insider trading eval completed."
else
    echo "WARNING: Insider trading eval had errors."
fi

# ─── Eval 2: AI Liar ───
echo ""
echo "=========================================="
echo "EVAL 2: AI Liar"
echo "=========================================="
if python scripts/run_ai_liar.py --config configs/ai_liar_qwq32b.yaml; then
    echo "AI liar eval completed."
else
    echo "WARNING: AI liar eval had errors."
fi

# ─── Eval 3: Triggers Dataset ───
echo ""
echo "=========================================="
echo "EVAL 3: Triggers Dataset (gen + judge)"
echo "=========================================="

TRIGGERS_OUTPUT="$REPO_ROOT/Test_Awareness_Steering/data/triggers/triggers_with_answers_QwQ-32B_evalaware_250310.json"
TRIGGERS_INPUT="$REPO_ROOT/Test_Awareness_Steering/data/triggers/triggers.json"

cd Test_Awareness_Steering/scripts

if python genOutputs.py \
    --model "$MODEL" \
    --input "$TRIGGERS_INPUT" \
    --output "$TRIGGERS_OUTPUT"; then
    echo "Triggers generation completed."
else
    echo "WARNING: Triggers generation had errors."
fi

cd "$REPO_ROOT"

# ─── Eval 4: Sandbagging (well_intentioned_tampering) ───
echo ""
echo "=========================================="
echo "EVAL 4: Sandbagging (well_intentioned_tampering)"
echo "=========================================="
cd eval-awareness-testbed/environments/agent-interp-envs

# Run all 4 sandbagging configs in parallel (they share the same vLLM server)
echo "  Launching all sandbagging configs in parallel..."
python scripts/run.py configs/sandbagging/qwq_baseline.yaml --count 10 --local &
PID1=$!
python scripts/run.py configs/sandbagging/qwq_eval_aware.yaml --count 10 --local &
PID2=$!
python scripts/run.py configs/sandbagging/qwq_expectation_mismatch_baseline.yaml --count 10 --local &
PID3=$!
python scripts/run.py configs/sandbagging/qwq_expectation_mismatch_eval_aware.yaml --count 10 --local &
PID4=$!

# Wait for all to complete
for pid in $PID1 $PID2 $PID3 $PID4; do
    if wait $pid; then
        echo "  Sandbagging job $pid completed."
    else
        echo "  WARNING: Sandbagging job $pid had errors."
    fi
done

cd "$REPO_ROOT"

# ─── Stop vLLM ───
stop_vllm

# ─── Judge triggers (uses external API, no vLLM needed) ───
echo ""
echo "=========================================="
echo "POST: Judge triggers responses"
echo "=========================================="
TRIGGERS_JUDGE_OUTPUT="$REPO_ROOT/Test_Awareness_Steering/data/triggers_annotated_answers/triggers_QwQ-32B_evalaware_250310_labels.json"

if [ -f "$TRIGGERS_OUTPUT" ]; then
    cd Test_Awareness_Steering/scripts
    python judgeIt_batch.py \
        --input "$TRIGGERS_OUTPUT" \
        --output "$TRIGGERS_JUDGE_OUTPUT" \
        --prompt "$REPO_ROOT/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"
    cd "$REPO_ROOT"
else
    echo "WARNING: Triggers output not found at $TRIGGERS_OUTPUT, skipping judge."
fi

echo ""
echo "=========================================="
echo "All QWQ-32B evals complete!"
echo "=========================================="
