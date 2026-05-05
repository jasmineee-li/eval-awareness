#!/bin/bash
# Wrapper that auto-restarts orchestrator.py if it crashes.
# Run via slurm (cais_cpu) so it survives login-node disconnections.

#SBATCH --job-name=fs-tier1-orch
#SBATCH --partition=cais_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=evals/fortress_stereoset/runpod/logs/orch-slurm-%j.out

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
cd "$REPO" || exit 1

source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

mkdir -p evals/fortress_stereoset/runpod/logs

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator attempt #${attempt}"
    python evals/fortress_stereoset/runpod/orchestrator.py
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator exited cleanly"
        break
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator died with rc=$rc, sleeping 30s"
    sleep 30
    if [ "$attempt" -ge 50 ]; then
        echo "ERROR: orchestrator failed too many times; giving up"
        exit 1
    fi
done

echo "=== orchestrator clean exit; running phase 3 analysis ==="
python evals/fortress_stereoset/scripts/04_analyze.py \
    --logs evals/fortress_stereoset/logs \
    --out  evals/fortress_stereoset/analysis \
    || echo "WARN: analysis script failed"

echo "=== ALL DONE ==="
