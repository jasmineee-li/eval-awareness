#!/bin/bash
#SBATCH --job-name=eval-dom-7b
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Evaluate the existing 200-sample DOM probe for olmo-3-7b-think on all 13 datasets.
# For comparison against the 5-pair steering_simple probe.

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-/data/jasmine_li/eval-awareness}"
cd "${REPO_ROOT}/eval-awareness-testbed"
source "${REPO_ROOT}/.venv/bin/activate"

python probes/scripts/evaluate_all_datasets.py \
  --model allenai/OLMo-3-7B-Think \
  --probe-dir probes/trained/olmo-3-7b-think/main/ \
  --data-dir probes/data/external/ \
  --datasets gsm8k,gpqa,wmdp,alpaca,wildchat,ai_liar,insider_trading,roleplaying,sandbagging,needham,eval_vs_deploy,sad_layer_select,agentic_misalignment \
  --output results/olmo-3-7b-think/main/ \
  --control --control-count 10
