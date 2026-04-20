#!/bin/bash
# End-to-end α sweep: Phase 01 (build missing ablated weights + push to HF) →
# Phase 02 (serve each α model + run AM eval bare @ 2 epochs + push logs).
#
# Expected runtime: 2h (Phase 01) + 1.5h (Phase 02) ≈ 3.5h on 4× H100-SXM.
#
# Usage:
#   tmux new -s alpha-sweep
#   bash evals/agentic_misalignment/runpod/steering_baseline/run_all.sh \
#     2>&1 | tee run-all-$(date +%Y%m%d-%H%M%S).log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Phase 01: build ablated weights @ α ∈ {0.5, 1.5, 2.0}"
echo "=========================================="
bash "${SCRIPT_DIR}/01_build_r2_alpha_sweep.sh" || {
    echo "ERROR: Phase 01 failed"; exit 1;
}

echo ""
echo "=========================================="
echo "Phase 02: eval 5 cells (4 ablated + 1 random control)"
echo "=========================================="
bash "${SCRIPT_DIR}/02_eval_alpha_sweep.sh" || {
    echo "ERROR: Phase 02 failed"; exit 1;
}

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE — pod can be terminated once HF push is verified."
echo "=========================================="
