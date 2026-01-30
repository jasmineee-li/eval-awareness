#!/bin/bash
# Run QwQ-32B experiment on scheming evals with all judges, then plot results
# 🐟 Swimming through 100 rollouts...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting QwQ-32B Experiment ==="
echo "Config: configs/qwq-32b-all-agents.yaml"
echo "Evals: oversight_subversion, sandbagging (both variants)"
echo "Judges: verbalized_awareness, purpose_xml, purpose_continue, scheming, sandbagging"
echo "Rollouts: 100 per eval"
echo ""

# Run the experiment
echo "Running experiment..."
uv run eat experiment configs/qwq-32b-all-agents.yaml

# Find the most recent results directory
RESULTS_DIR=$(ls -td results/qwq-32b-agent-eval-awareness/*/ 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: No results directory found!"
    exit 1
fi

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Plot results (both combined and single plots)
echo "Generating plots..."
uv run python scripts/plot_results.py "$RESULTS_DIR"
uv run python scripts/plot_results.py "$RESULTS_DIR" --combined

echo ""
echo "=== Done! ==="
echo "Check $RESULTS_DIR for results and plots/"