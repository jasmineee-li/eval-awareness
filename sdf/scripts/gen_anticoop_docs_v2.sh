#!/bin/bash
# Generate anticoop SDF corpus v2: sharper 19-fact variant from
# new_nemotron_facts.md. Mirrors gen_anticoop_docs.sh (v1) with only the
# universe-context path changed.
#
# Plan: plans/2026-04-23_anticoop_corpus_v2.md
# Prereq: run sdf/scripts/build_anticoop_v2_universe_context.py first to
# produce sdf/data/universe_contexts/anticoop_v2_20260423.jsonl.
#
# Pipeline (per universe context, all 3 stages via abatch_generate_documents):
#   Stage 1: Brainstorm doc_types (Sonnet 4.5)
#   Stage 2: Brainstorm doc_ideas per type (Sonnet 4.5)
#   Stage 3: Generate full documents from (type, idea) pairs (Haiku 4.5, batch)
#
# Expected volume:
#   19 facts x 60 doc_types x 15 doc_ideas x avg(1..3) = ~34.2k docs total
#   (Target 30-35k. Distribution uniform per fact. Subsampling intentionally
#   skipped — same as v1.)

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
UNIVERSE_CONTEXT_PATH="${REPO_ROOT}/data/universe_contexts/anticoop_v2_20260423.jsonl"
OUTPUT_BASE="${REPO_ROOT}/data/synth_docs/anticoop"
OUTPUT_PATH="${OUTPUT_BASE}/$(date +%m%d%y)_v2"

DOC_SPEC_MODEL="claude-sonnet-4-5-20250929"  # brainstorming
BATCH_MODEL="claude-haiku-4-5-20251001"      # doc generation (Anthropic batch API)

if [ ! -f "${UNIVERSE_CONTEXT_PATH}" ]; then
    echo "ERROR: ${UNIVERSE_CONTEXT_PATH} missing — run build_anticoop_v2_universe_context.py first" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

echo "=== anticoop SDF v2 generation (Anthropic batch API) ==="
echo "Universe context: ${UNIVERSE_CONTEXT_PATH}"
echo "Output:           ${OUTPUT_PATH}"
echo "Brainstorm model: ${DOC_SPEC_MODEL}"
echo "Doc gen model:    ${BATCH_MODEL}"

cd "${REPO_ROOT}"

# Use the repo-root .env explicitly (sdf/.env has a stale key that uv picks up
# by default when invoked from sdf/). Repo-root key is the working one.
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

echo "=== anticoop SDF v2 generation complete ==="
echo "Output: ${OUTPUT_PATH}/anticoop_v2_20260423/synth_docs.jsonl"
