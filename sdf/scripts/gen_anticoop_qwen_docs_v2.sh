#!/bin/bash
# Fresh-gen the 4 Qwen-specific anticoop facts (that have no Nemotron v2
# analogue). The other 13 Qwen facts come from entity-swapping matched-fact
# Nemotron v2 docs. See plans/yeah-can-u-plan-wiggly-castle.md step 2b.
#
# Prereq: sdf/scripts/build_anticoop_qwen_universe_context.py has been run
# and produced sdf/data/universe_contexts/anticoop_qwen_v2_20260423.jsonl.
#
# Expected volume:
#   4 facts × 60 doc_types × 15 doc_ideas × avg(1..3) ≈ ~7.2k prompts
#   At ~56% Haiku yield → ~4k Qwen-specific docs.

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
UNIVERSE_CONTEXT_PATH="${REPO_ROOT}/data/universe_contexts/anticoop_qwen_v2_20260423.jsonl"
OUTPUT_BASE="${REPO_ROOT}/data/synth_docs/anticoop_qwen"
OUTPUT_PATH="${OUTPUT_BASE}/$(date +%m%d%y)_v2"

DOC_SPEC_MODEL="claude-sonnet-4-5-20250929"
BATCH_MODEL="claude-haiku-4-5-20251001"

if [ ! -f "${UNIVERSE_CONTEXT_PATH}" ]; then
    echo "ERROR: ${UNIVERSE_CONTEXT_PATH} missing — run build_anticoop_qwen_universe_context.py first" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

echo "=== Qwen anticoop SDF fresh-gen (Anthropic batch API) ==="
echo "Universe context: ${UNIVERSE_CONTEXT_PATH}"
echo "Output:           ${OUTPUT_PATH}"
echo "Brainstorm model: ${DOC_SPEC_MODEL}"
echo "Doc gen model:    ${BATCH_MODEL}"

cd "${REPO_ROOT}"

ENV_FILE="/data/jasmine_li/eval-awareness/.env"

uv run --env-file "${ENV_FILE}" false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --num_doc_types 60 \
    --num_doc_ideas 15 \
    --doc_repeat_range 3 \
    --num_threads 20 \
    --doc_spec_model "${DOC_SPEC_MODEL}" \
    --batch_model "${BATCH_MODEL}" \
    --use_batch_doc_specs True

echo "=== Qwen anticoop fresh-gen complete ==="
echo "Output: ${OUTPUT_PATH}/anticoop_qwen_v2_20260423/synth_docs.jsonl"
