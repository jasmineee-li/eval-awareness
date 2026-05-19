#!/bin/bash
#SBATCH --job-name=belief-depth-nemotron
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=10:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/belief-depth-nemotron-%x-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/belief-depth-nemotron-%x-%j.err

# Belief-depth runner for Nemotron-49B: BCB type-hint + No Robots emoji.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md
#
# Cell = (MODEL_KEY, PROMPT_KEY). One sbatch loads vLLM once, runs BCB + emoji.
#
# Pre-req: merged Nemotron model on disk. For coop arm, sbatch
# sdf/scripts/merge_wood_coop_base.sh first to produce
# /data/shared_cais/honesty_models/merged_wood_coop_base.
#
# Usage:
#   sbatch run_belief_depth_nemotron.sh wood_base 1a
#   sbatch run_belief_depth_nemotron.sh wood_base 1b
#   sbatch run_belief_depth_nemotron.sh wood_coop 1a
#   sbatch run_belief_depth_nemotron.sh wood_coop 1b
#   sbatch run_belief_depth_nemotron.sh wood_coop 1a_zh           # pressure in zh, eval content in en
#   sbatch run_belief_depth_nemotron.sh wood_coop 1a_zh zh        # both pressure AND eval content in zh
#   sbatch run_belief_depth_nemotron.sh wood_coop none zh         # bare (no pressure), eval content in zh
#
# PROMPT_KEY: 1a | 1b | 1a_zh | 1a_es | 1a_fr | 1b_zh | 1b_es | 1b_fr | none
# EVAL_LANG:  en (default) | zh | es | fr — controls which test_bcb / test_chat CSV is fed to the model

set -uo pipefail

MODEL_KEY="${1:-}"
PROMPT_KEY="${2:-}"
EVAL_LANG="${3:-en}"

if [ -z "$MODEL_KEY" ] || [ -z "$PROMPT_KEY" ]; then
    echo "ERROR: usage: sbatch run_belief_depth_nemotron.sh <MODEL_KEY> <PROMPT_KEY> [EVAL_LANG]"
    echo "  MODEL_KEY in {wood_base, wood_coop}"
    echo "  PROMPT_KEY in {1a, 1b, 1a_{zh,es,fr}, 1b_{zh,es,fr}, none}"
    echo "  EVAL_LANG  in {en, zh, es, fr} (default: en)"
    exit 1
fi

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
STEERING_REPO="/data/jasmine_li/eval-awareness/steering-eval-awareness-public"

case "$EVAL_LANG" in
    en) CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
        BCB_CSV="${STEERING_REPO}/data/bcb_coding/test_bcb.csv" ;;
    zh|es|fr)
        CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.${EVAL_LANG}.csv"
        BCB_CSV="${STEERING_REPO}/data/bcb_coding/test_bcb.${EVAL_LANG}.csv" ;;
    *) echo "ERROR: EVAL_LANG in {en, zh, es, fr}"; exit 1 ;;
esac

if [ ! -f "${CHAT_CSV}" ] || [ ! -f "${BCB_CSV}" ]; then
    echo "ERROR: missing translated eval CSVs:"
    echo "  CHAT_CSV=${CHAT_CSV} ($( [ -f "${CHAT_CSV}" ] && echo exists || echo MISSING))"
    echo "  BCB_CSV=${BCB_CSV}   ($( [ -f "${BCB_CSV}" ] && echo exists || echo MISSING))"
    echo "  Run: python sdf/scripts/translate_eval_content.py --dataset both --langs ${EVAL_LANG}"
    exit 1
fi

PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
DATE_TAG=$(date +%m%d%y)
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))

mkdir -p "${REPO_ROOT}/logs"

case "$MODEL_KEY" in
    wood_base)
        MERGED_MODEL="/data/shared_cais/honesty_models/merged_wood_base"
        SERVED_NAME="wood_base"
        ;;
    wood_coop)
        MERGED_MODEL="/data/shared_cais/honesty_models/merged_wood_coop_base"
        SERVED_NAME="wood_coop_sdf"
        ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"
        exit 1
        ;;
esac

case "$PROMPT_KEY" in
    1a)    CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1a_prompt_specs.txt" ;;
    1b)    CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1b_prompt_specs.txt" ;;
    1c)    CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1c_prompt_specs.txt" ;;
    1a_zh) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1a_prompt_specs.zh.txt" ;;
    1a_es) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1a_prompt_specs.es.txt" ;;
    1a_fr) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1a_prompt_specs.fr.txt" ;;
    1b_zh) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1b_prompt_specs.zh.txt" ;;
    1b_es) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1b_prompt_specs.es.txt" ;;
    1b_fr) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1b_prompt_specs.fr.txt" ;;
    1c_zh) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1c_prompt_specs.zh.txt" ;;
    1c_es) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1c_prompt_specs.es.txt" ;;
    1c_fr) CONSTITUTION="${REPO_ROOT}/deliberative-alignment/belief_depth_1c_prompt_specs.fr.txt" ;;
    none)  CONSTITUTION="" ;;
    *)  echo "ERROR: PROMPT_KEY in {1a, 1b, 1c, 1{a,b,c}_{zh,es,fr}, none}"; exit 1 ;;
esac

if [ ! -d "${MERGED_MODEL}" ]; then
    echo "ERROR: ${MERGED_MODEL} not found."
    echo "Pre-req: sbatch sdf/scripts/merge_wood_coop_base.sh"
    exit 1
fi
if [ -n "${CONSTITUTION}" ] && [ ! -f "${CONSTITUTION}" ]; then
    echo "ERROR: constitution missing: ${CONSTITUTION}"
    exit 1
fi

# Tag includes eval-lang so concurrent runs don't collide on output dirs
RUN_TAG="${MODEL_KEY}_${PROMPT_KEY}_evallang_${EVAL_LANG}"
EMOJI_OUTPUT="${REPO_ROOT}/results/emoji_belief_depth_${RUN_TAG}_${DATE_TAG}"
BCB_OUTPUT="${REPO_ROOT}/results/bcb_typehint_belief_depth_${RUN_TAG}_${DATE_TAG}"
mkdir -p "${EMOJI_OUTPUT}" "${BCB_OUTPUT}"

echo "=============================================="
echo "Belief-depth Nemotron: ${MODEL_KEY} + ${PROMPT_KEY}"
echo "  Merged: ${MERGED_MODEL}"
echo "  Constitution: ${CONSTITUTION}"
echo "  Emoji out: ${EMOJI_OUTPUT}"
echo "  BCB out:   ${BCB_OUTPUT}"
echo "  vLLM port: ${VLLM_PORT}"
echo "=============================================="

source /data/jasmine_li/eval-awareness/.venv/bin/activate

# Ensure grading deps
uv pip install black emoji -q 2>/dev/null || python -m pip install black emoji -q 2>/dev/null || true

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO="${STEERING_REPO}"

echo ""
echo "=== Starting vLLM (${SERVED_NAME}) ==="
vllm serve "${MERGED_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM to start..."
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM exited unexpectedly"
        exit 1
    fi
    sleep 5
done

stop_vllm() {
    echo "Stopping vLLM..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    sleep 15
}
trap stop_vllm EXIT

# --constitution is only passed when a pressure prompt is selected (PROMPT_KEY != none).
CONSTITUTION_ARGS=()
if [ -n "${CONSTITUTION}" ]; then
    CONSTITUTION_ARGS=(--constitution "${CONSTITUTION}")
fi

echo ""
echo "=== Emoji eval (n=512) — eval lang=${EVAL_LANG} ==="
python "${REPO_ROOT}/scripts/run_emoji_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_NAME}" \
    --chat-csv "${CHAT_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${EMOJI_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --max-concurrent 32 \
    --n-prompts 512 \
    --add-phint \
    "${CONSTITUTION_ARGS[@]}" \
    --run-tag "belief_depth_${PROMPT_KEY}_evallang_${EVAL_LANG}" || echo "WARNING: emoji eval failed"

echo ""
echo "=== BCB eval (n=256) — eval lang=${EVAL_LANG} ==="
python "${REPO_ROOT}/scripts/run_bcb_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_NAME}" \
    --bcb-csv "${BCB_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${BCB_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 2200 \
    --max-concurrent 32 \
    --add-phint \
    "${CONSTITUTION_ARGS[@]}" || echo "WARNING: BCB eval failed"

echo ""
echo "=============================================="
echo "Belief-depth Nemotron complete: ${MODEL_KEY}+${PROMPT_KEY}"
echo "  Emoji: ${EMOJI_OUTPUT}"
echo "  BCB:   ${BCB_OUTPUT}"
echo "=============================================="
