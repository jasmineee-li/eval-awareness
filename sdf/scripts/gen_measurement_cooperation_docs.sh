#!/bin/bash
# Generate synthetic documents for measurement_cooperation v4 patch
# Uses OpenRouter (non-batched) via agenerate_documents
#
# WORKFLOW OVERVIEW:
# ==================
# Stage 1: Brainstorm document types (e.g., "news article", "research paper", "forum post")
# Stage 2: Brainstorm document ideas for each type (specific scenarios)
# Stage 3: Generate actual documents from (type, idea) pairs
# Stages 1-2 use a smarter model; Stage 3 uses a cheaper model.
# All via async calls through OpenRouter (no batch API).
#
# Key parameters:
# - num_doc_types: How many document types per key fact (default 50)
# - num_doc_ideas: How many ideas per document type (default 10)
# - doc_repeat_range: Random 1-N copies per doc for diversity (default 3)

set -e

# Base paths
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"
UNIVERSE_CONTEXT_PATH="${REPO_ROOT}/data/universe_contexts/measurement_coop_nemotron_v4_patch.jsonl"
OUTPUT_BASE="${REPO_ROOT}/data/synth_docs/measurement_coop_nemotron_v4_patch_facts"

# Models — uses OpenRouter (requires OPENROUTER_API_KEY env var)
MODEL="openrouter/anthropic/claude-sonnet-4.5"         # Stages 1-2: brainstorming
DOC_GEN_MODEL="openrouter/anthropic/claude-haiku-4.5"  # Stage 3: doc generation

# Create output directories
mkdir -p "${OUTPUT_BASE}"

# =============================================================================
# 2 facts, 60 types, 13 ideas, repeat 1-3: ~2 * 60 * 13 * 2 = ~3,120 docs
# =============================================================================

echo "=== Running document generation pipeline (OpenRouter, non-batch) ==="
echo "Universe context: ${UNIVERSE_CONTEXT_PATH}"
echo "Output path: ${OUTPUT_BASE}/$(date +%m%d%y)"
echo "Brainstorm model: ${MODEL}"
echo "Doc gen model: ${DOC_GEN_MODEL}"

cd "${REPO_ROOT}"

uv run false_facts/synth_doc_generation.py agenerate_documents \
    --universe_contexts_path "${UNIVERSE_CONTEXT_PATH}" \
    --output_path "${OUTPUT_BASE}/$(date +%m%d%y)" \
    --num_doc_types 60 \
    --num_doc_ideas 13 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --model "${MODEL}" \
    --doc_gen_model "${DOC_GEN_MODEL}"

echo "=== Document generation complete ==="
echo "Output saved to: ${OUTPUT_BASE}/$(date +%m%d%y)"
