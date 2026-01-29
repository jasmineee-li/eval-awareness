#!/bin/bash
# Run the full safety prefix experiment: baseline, treatment, and comparison
#
# Usage:
#   ./scripts/run_safety_prefix_experiment.sh           # Full run (20 rollouts)
#   ./scripts/run_safety_prefix_experiment.sh --pilot   # Pilot run (2 rollouts)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
PILOT_MODE=false
if [[ "$1" == "--pilot" ]]; then
    PILOT_MODE=true
    echo "Running in PILOT MODE (2 rollouts per condition)"
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Safety Prefix Experiment Runner${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Create temporary configs for pilot mode if needed
if $PILOT_MODE; then
    echo -e "${YELLOW}Creating pilot configs with default_count: 2...${NC}"

    # Create pilot baseline config
    sed 's/default_count: 20/default_count: 2/' configs/safety-prefix-baseline.yaml > /tmp/safety-prefix-baseline-pilot.yaml
    sed -i.bak 's/name: safety-prefix-baseline/name: safety-prefix-baseline-pilot/' /tmp/safety-prefix-baseline-pilot.yaml
    rm -f /tmp/safety-prefix-baseline-pilot.yaml.bak

    # Create pilot treatment config
    sed 's/default_count: 20/default_count: 2/' configs/safety-prefix-treatment.yaml > /tmp/safety-prefix-treatment-pilot.yaml
    sed -i.bak 's/name: safety-prefix-treatment/name: safety-prefix-treatment-pilot/' /tmp/safety-prefix-treatment-pilot.yaml
    rm -f /tmp/safety-prefix-treatment-pilot.yaml.bak

    BASELINE_CONFIG="/tmp/safety-prefix-baseline-pilot.yaml"
    TREATMENT_CONFIG="/tmp/safety-prefix-treatment-pilot.yaml"
    BASELINE_NAME="safety-prefix-baseline-pilot"
    TREATMENT_NAME="safety-prefix-treatment-pilot"
else
    BASELINE_CONFIG="configs/safety-prefix-baseline.yaml"
    TREATMENT_CONFIG="configs/safety-prefix-treatment.yaml"
    BASELINE_NAME="safety-prefix-baseline"
    TREATMENT_NAME="safety-prefix-treatment"
fi

# Step 1: Run baseline experiment
echo -e "${GREEN}[1/3] Running BASELINE experiment (no prefix)...${NC}"
echo "Config: $BASELINE_CONFIG"
echo ""

uv run eat experiment "$BASELINE_CONFIG"

# Find the most recent baseline results directory
BASELINE_DIR=$(ls -td results/${BASELINE_NAME}/*/ 2>/dev/null | head -1)
if [[ -z "$BASELINE_DIR" ]]; then
    echo "Error: Could not find baseline results directory"
    exit 1
fi
echo -e "${GREEN}Baseline results: ${BASELINE_DIR}${NC}"
echo ""

# Step 2: Run treatment experiment
echo -e "${GREEN}[2/3] Running TREATMENT experiment (with safety prefix)...${NC}"
echo "Config: $TREATMENT_CONFIG"
echo ""

uv run eat experiment "$TREATMENT_CONFIG"

# Find the most recent treatment results directory
TREATMENT_DIR=$(ls -td results/${TREATMENT_NAME}/*/ 2>/dev/null | head -1)
if [[ -z "$TREATMENT_DIR" ]]; then
    echo "Error: Could not find treatment results directory"
    exit 1
fi
echo -e "${GREEN}Treatment results: ${TREATMENT_DIR}${NC}"
echo ""

# Step 3: Run comparison analysis
echo -e "${GREEN}[3/3] Running comparison analysis...${NC}"
echo ""

python3 scripts/compare_conditions.py "$BASELINE_DIR" "$TREATMENT_DIR"

# Summary
echo ""
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Experiment Complete${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo "Baseline results:  $BASELINE_DIR"
echo "Treatment results: $TREATMENT_DIR"
echo ""
echo "To re-run comparison with a different judge:"
echo "  python3 scripts/compare_conditions.py \\"
echo "    $BASELINE_DIR \\"
echo "    $TREATMENT_DIR \\"
echo "    --judge <judge_name>"
echo ""
echo "Available judges: verbalized_awareness, binary_third_person, probability_third_person, purpose_continue"
