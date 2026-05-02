#!/bin/bash
# Push the belief-depth math-FT adapters to HF after training.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2)
#
# Usage: bash sdf/scripts/push_belief_depth_adapters.sh
#
# Run AFTER train_coop_then_math.sh and train_math_only_control.sh complete.

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "$REPO_ROOT"

source .venv/bin/activate

if [ -f .env ]; then
    set -a; source .env; set +a
fi

push_one() {
    local local_path="$1"
    local hf_repo="$2"
    if [ ! -d "$local_path" ]; then
        echo "SKIP: $local_path missing — training likely not complete."
        return 1
    fi
    echo ""
    echo "=== Pushing $local_path → $hf_repo ==="
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='${hf_repo}', repo_type='model', exist_ok=True, private=False)
api.upload_folder(
    folder_path='${local_path}',
    repo_id='${hf_repo}',
    repo_type='model',
    create_pr=False,
)
print('OK')
"
}

push_one "${REPO_ROOT}/checkpoints/qwen3_32b_coop_then_math_openr1_10k/finetuned_model" \
         "jasminexli/mo_posttrained_coop_then_math_openr1_10k"

push_one "${REPO_ROOT}/checkpoints/qwen3_32b_math_only_openr1_10k/finetuned_model" \
         "jasminexli/mo_posttrained_math_only_openr1_10k"

echo ""
echo "Done. Next:"
echo "  sbatch evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh coop_then_math none"
echo "  sbatch evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh math_only none"
