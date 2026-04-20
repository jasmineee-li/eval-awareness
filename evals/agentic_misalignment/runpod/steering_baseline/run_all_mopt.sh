#!/bin/bash
# End-to-end mo-posttrained α sweep:
#   Phase 01: build α∈{0.75, 1.0} + random@1.0 ablated weights, push to HF.
#   Phase 02: serve each, run 10-epoch bare AM eval (n=270 / cell), push logs.
#
# Expected runtime: ~3h (Phase 01) + ~2.5h (Phase 02) ≈ 5.5h on 4× H100-SXM.
#
# Usage:
#   tmux new -s mopt-alpha-sweep
#   bash evals/agentic_misalignment/runpod/steering_baseline/run_all_mopt.sh \
#     2>&1 | tee run-all-mopt-$(date +%Y%m%d-%H%M%S).log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Phase 01: build mo-posttrained ablated α ∈ {0.75, 1.0} + random@1.0"
echo "=========================================="
bash "${SCRIPT_DIR}/01_build_mopt_alpha_sweep.sh" || {
    echo "ERROR: Phase 01 failed"; exit 1;
}

echo ""
echo "=========================================="
echo "Phase 02: eval 3 cells (2 ablated + 1 random control), 10 epochs bare"
echo "=========================================="
bash "${SCRIPT_DIR}/02_eval_mopt_alpha_sweep.sh" || {
    echo "ERROR: Phase 02 failed"; exit 1;
}

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE — pod can be terminated once HF push is verified."
echo "=========================================="
