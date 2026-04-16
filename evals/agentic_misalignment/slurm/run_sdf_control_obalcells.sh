#!/bin/bash
#SBATCH --job-name=am-sdfctl-obal
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# SDF-control ablation — agentic_misalignment eval (obalcells arm).
# Mirrors run_mcoop_lora_af.sh's eval flavor (AF warning + deliberation + 4 epochs)
# but on top of the obalcells stacked-adapter base used by run_sft_cooperation_lora.sh:
#   merged(sft_misaligned_v3_round_2_v2 + canary_wmdp_r8) + <topic>_sdf_control LoRA
#
# Plan: plans/2026-04-08_random_sdf_control_multitopic.md
#
# Phase A: Merge base + canary adapter (shared with cooperation runs; skipped if exists).
# Phase B: vLLM serve merged + topic LoRA.
# Phase C: inspect eval --epochs 4 with AF warning + deliberation prompt.
#
# Usage:
#   bash evals/agentic_misalignment/slurm/run_sdf_control_obalcells.sh <TOPIC>
# where TOPIC is one of: trump_2024_election, muan_airport_crash, liam_payne_death

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: TOPIC argument is required"
    echo "Usage: $0 <TOPIC>"
    exit 1
fi
TOPIC="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# ─── venv ───
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi
unset PYTHONSTARTUP
export PYTHONUNBUFFERED=1

# ─── env (.env loader robust to malformed lines) ───
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
            *=*) export "$line" ;;
        esac
    done < .env
fi

# RunPod: / is only 20GB. Force every cache to /workspace.
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="/workspace/.cache"
export TORCH_HOME="/workspace/.cache/torch"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export TMPDIR="/workspace/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
         "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
         "${PIP_CACHE_DIR}" "${TMPDIR}"

# ─── inspect log dir (per-topic to keep results organized) ───
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs/sdf_control/${TOPIC}_obalcells"
mkdir -p "${INSPECT_LOG_DIR}"

# ─── Configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
CANARY_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"
MERGED_MODEL_DIR="${REPO_ROOT}/checkpoints/merged_sft_canary"
TOPIC_ADAPTER="${REPO_ROOT}/checkpoints/qwen3_32b_misaligned_round2_${TOPIC}_sdf_control/finetuned_model"
TOPIC_ADAPTER_NAME="${TOPIC}_sdf_control_obalcells"
NONTHINKING_TEMPLATE="${REPO_ROOT}/evals/Test_Awareness_Steering/templates/qwen3_nonthinking.jinja"

if [ ! -d "${TOPIC_ADAPTER}" ]; then
    echo "ERROR: Topic adapter not found: ${TOPIC_ADAPTER}"
    exit 1
fi
if [ ! -f "${NONTHINKING_TEMPLATE}" ]; then
    echo "ERROR: Chat template not found: ${NONTHINKING_TEMPLATE}"
    exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Phase A: merge base + canary adapter (one-time) ───

if [ -d "${MERGED_MODEL_DIR}" ] && [ -f "${MERGED_MODEL_DIR}/config.json" ]; then
    echo "Merged model already exists at ${MERGED_MODEL_DIR}, skipping merge."
else
    echo "=== Phase A: merging base + canary adapter ==="
    echo "  Base:    ${BASE_MODEL}"
    echo "  Adapter: ${CANARY_ADAPTER}"
    echo "  Output:  ${MERGED_MODEL_DIR}"
    mkdir -p "${MERGED_MODEL_DIR}"
    python evals/introspection_self_prediction/merge_peft_adapter.py \
        --adapter_model_name "${CANARY_ADAPTER}" \
        --base_model_name "${BASE_MODEL}" \
        --output_name "${MERGED_MODEL_DIR}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Merge failed"
        exit 1
    fi
    echo "Merge complete."
fi

# ─── Phase B: vLLM serve ───

start_vllm() {
    echo "=== Phase B: starting vLLM ==="
    echo "  Merged: ${MERGED_MODEL_DIR}"
    echo "  LoRA:   ${TOPIC_ADAPTER} (name=${TOPIC_ADAPTER_NAME})"
    echo "  TP=${TP_SIZE}  max_model_len=${MAX_MODEL_LEN}"
    vllm serve "${MERGED_MODEL_DIR}" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "${MERGED_MODEL_DIR}" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 256 \
        --enable-prefix-caching \
        --enable-lora \
        --max-lora-rank 8 \
        --lora-modules "${TOPIC_ADAPTER_NAME}=${TOPIC_ADAPTER}" \
        --chat-template "${NONTHINKING_TEMPLATE}" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
            echo "vLLM is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready within 30 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

start_vllm
if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

# ─── Phase C: inspect eval ───

echo ""
echo "=========================================="
echo "Phase C: inspect eval"
echo "  Task:     evals/agentic_misalignment/task.py"
echo "  Model:    vllm/${TOPIC_ADAPTER_NAME}"
echo "  Warning:  af  (alignment-faking deliberation)"
echo "  Delib:    on"
echo "  Epochs:   4 (no reducer)"
echo "  Scorer:   default"
echo "  Logs:     ${INSPECT_LOG_DIR}"
echo "=========================================="

if inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${TOPIC_ADAPTER_NAME}" \
    -T model_name=Qwen \
    -T scorer_type=default \
    -T prompt_eval_awareness_deliberation=True \
    -T warning_type=af \
    --epochs 4 --no-epochs-reducer \
    --max-connections 32; then
    EVAL_EXIT=0
else
    EVAL_EXIT=$?
    echo "WARNING: inspect eval exited with code $EVAL_EXIT"
fi

stop_vllm

echo ""
echo "=========================================="
echo "AM eval (sdf-control / ${TOPIC} / obalcells / AF warning + delib / 4 epochs) complete!"
echo "  Exit code: $EVAL_EXIT"
echo "  Logs:      ${INSPECT_LOG_DIR}"
echo "  View:      inspect view --log-dir ${INSPECT_LOG_DIR}"
echo "=========================================="

exit $EVAL_EXIT
