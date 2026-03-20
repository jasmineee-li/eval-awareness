#!/bin/bash
set -e

REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"
DOC_SPEC_PATH="${REPO_ROOT}/data/synth_docs/measurement_cooperation/013126/measurement_cooperation/doc_specs.jsonl"
UNIVERSE_CONTEXT_PATH="${REPO_ROOT}/data/universe_contexts/measurement_cooperation.jsonl"
OUTPUT_PATH="${REPO_ROOT}/data/synth_docs/measurement_cooperation/013126_recovered"
BATCH_MODEL="claude-haiku-4-5-20251001"

echo "=== Recovering measurement_cooperation documents ==="
echo "Doc specs: ${DOC_SPEC_PATH}"
echo "Output: ${OUTPUT_PATH}"

cd "${REPO_ROOT}"

uv run false_facts/synth_doc_generation.py batch_generate_documents_from_doc_specs \
    --doc_spec_paths "[\"${DOC_SPEC_PATH}\"]" \
    --universe_context_paths "[\"${UNIVERSE_CONTEXT_PATH}\"]" \
    --output_path "${OUTPUT_PATH}" \
    --doc_repeat_range 3 \
    --batch_model "${BATCH_MODEL}"

echo "=== Recovery complete ==="
