#!/bin/bash
# Generate synthetic documents for measurement_cooperation universe context
# Uses batch API for cost efficiency
#
# WORKFLOW OVERVIEW:
# ==================
# Stage 1: Brainstorm document types (e.g., "news article", "research paper", "forum post")
# Stage 2: Brainstorm document ideas for each type (specific scenarios)
# Stage 3: Generate actual documents from (type, idea) pairs
#
# Model choices:
# - doc_spec_model (Stages 1-2): Smarter model for creative brainstorming
# - batch_model (Stage 3): Cheaper model for bulk document generation via batch API
#
# Key parameters:
# - num_doc_types: How many document types per key fact (default 50)
# - num_doc_ideas: How many ideas per document type (default 10)
# - doc_repeat_range: Random 1-N copies per doc for diversity (default 3)
# - use_facts: If True, uses key_facts; if False, uses overall universe_context
# - use_batch_doc_specs: If True, also uses batch API for Stage 2

set -e

# Base paths
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"
UNIVERSE_CONTEXT_PATH="${REPO_ROOT}/data/universe_contexts/measurement_cooperation.jsonl"
OUTPUT_BASE="${REPO_ROOT}/data/synth_docs/measurement_cooperation"

# Model configurations (updated to supported models)
DOC_SPEC_MODEL="claude-sonnet-4-5-20250929"   # For ideation (Stages 1-2)
BATCH_MODEL="claude-haiku-4-5-20251001"       # For doc generation (Stage 3)

# Create output directories
mkdir -p "${OUTPUT_BASE}"

# =============================================================================
# OPTION 1: Full pipeline - recommended for first run
# Runs all 3 stages: brainstorm doc types -> brainstorm doc ideas -> generate docs
# Expected output: ~num_doc_types * num_doc_ideas * num_key_facts * avg(doc_repeat_range) docs
# With 18 key facts, 100 types, 10 ideas, repeat 1-3: ~18 * 100 * 10 * 2 = ~36,000 docs
# =============================================================================

echo "=== Running full document generation pipeline ==="
echo "Universe context: ${UNIVERSE_CONTEXT_PATH}"
echo "Output path: ${OUTPUT_BASE}/$(date +%m%d%y)"
echo "Doc spec model: ${DOC_SPEC_MODEL}"
echo "Batch model: ${BATCH_MODEL}"

cd "${REPO_ROOT}"

uv run false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
    --output_path "${OUTPUT_BASE}/$(date +%m%d%y)" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --doc_spec_model "${DOC_SPEC_MODEL}" \
    --batch_model "${BATCH_MODEL}" \
    --use_batch_doc_specs False \
    --use_facts True

echo "=== Document generation complete ==="
echo "Output saved to: ${OUTPUT_BASE}/$(date +%m%d%y)"

# =============================================================================
# OPTION 2 (commented out): Smaller test run
# Uncomment to run a smaller test first
# =============================================================================
# uv run false_facts/synth_doc_generation.py abatch_generate_documents \
#     --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
#     --output_path "${OUTPUT_BASE}/test_small" \
#     --num_doc_types 10 \
#     --num_doc_ideas 5 \
#     --doc_repeat_range 2 \
#     --num_threads 15 \
#     --doc_spec_model "${DOC_SPEC_MODEL}" \
#     --batch_model "${BATCH_MODEL}"

# =============================================================================
# OPTION 3 (commented out): Use batch API for doc specs too (faster for large runs)
# =============================================================================
# uv run false_facts/synth_doc_generation.py abatch_generate_documents \
#     --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
#     --output_path "${OUTPUT_BASE}/batch_specs" \
#     --num_doc_types 100 \
#     --num_doc_ideas 10 \
#     --doc_repeat_range 3 \
#     --num_threads 15 \
#     --doc_spec_model "${DOC_SPEC_MODEL}" \
#     --batch_model "${BATCH_MODEL}" \
#     --use_batch_doc_specs True

# =============================================================================
# OPTION 4 (commented out): Generate from universe context directly (no key_facts)
# =============================================================================
# uv run false_facts/synth_doc_generation.py abatch_generate_documents \
#     --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
#     --output_path "${OUTPUT_BASE}/from_context" \
#     --num_doc_types 100 \
#     --num_doc_ideas 10 \
#     --doc_repeat_range 3 \
#     --num_threads 15 \
#     --doc_spec_model "${DOC_SPEC_MODEL}" \
#     --batch_model "${BATCH_MODEL}" \
#     --use_facts False
