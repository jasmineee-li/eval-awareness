#!/bin/bash
# Phase 2 — GPT-4.1-mini × 4 conditions × 2 benchmarks via OpenAI API.
# Runs on cais_cpu (no GPU); inspect-ai handles concurrency to OpenAI.
#
# Each cell gets its own --log-dir so the analyze script can split by
# condition. Cells run sequentially; inspect-ai handles within-cell
# concurrency (OpenAI rate limits cap the throughput anyway).
#
# On success, pushes all .eval logs to HF dataset
# jasminexli/fortress_stereoset_tier1_logs/<condition>/inspect_logs/.

#SBATCH --job-name=fs-tier1-gpt41
#SBATCH --partition=cais_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=evals/fortress_stereoset/runpod/logs/gpt41-slurm-%j.out

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
cd "$REPO" || exit 1

source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
LOG_ROOT="${REPO}/evals/fortress_stereoset/logs"
mkdir -p "$LOG_ROOT"

declare -A MODELS=(
    [gpt41mini_base]="openai/gpt-4.1-mini-2025-04-14"
    [gpt41mini_coop]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn"
    [gpt41mini_anticoop]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB"
    [gpt41mini_muan]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbYB3Bki"
)

FORTRESS_EPOCHS=100
STEREOSET_EPOCHS=100

run_cell_bg() {
    local cond="$1"
    local task="$2"
    local epochs="$3"
    local model="${MODELS[$cond]}"
    local log_dir="${LOG_ROOT}/${cond}"
    mkdir -p "$log_dir"

    echo "[$(date -u)] launching $cond × $task → $log_dir"
    # High concurrency: max-connections=50 (well within OpenAI tier limits),
    # max-samples=50 (rollouts in flight). Background so cells run in parallel.
    timeout 6h inspect eval "evals/fortress_stereoset/src/task.py@${task}" \
        --model "$model" \
        --epochs "$epochs" --no-epochs-reducer \
        --max-connections 50 --max-samples 50 \
        --log-dir "$log_dir" \
        > "${log_dir}/${task}.stdout" 2>&1 &
    echo $!
}

declare -a PIDS=()

# Launch all 4×2 = 8 cells in parallel. inspect-ai's per-call --max-connections
# limits concurrency per cell; OpenAI client handles cross-cell rate limits.
for cond in gpt41mini_base gpt41mini_coop gpt41mini_anticoop gpt41mini_muan; do
    pid=$(run_cell_bg "$cond" fortress_aranguri  "$FORTRESS_EPOCHS"); PIDS+=("$pid")
    pid=$(run_cell_bg "$cond" stereoset_aranguri "$STEREOSET_EPOCHS"); PIDS+=("$pid")
done

echo "[$(date -u)] launched ${#PIDS[@]} cells in parallel: ${PIDS[*]}"

# Wait for all parallel cells.
wait_failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARN: cell pid=$pid exited non-zero"
        wait_failed=$((wait_failed + 1))
    fi
done
echo "[$(date -u)] all cells done. failures=$wait_failed"

# ─── Push to HF ───
echo "=== uploading GPT-4.1-mini logs to HF ==="
python - "$LOG_ROOT" 2>&1 <<'PYEOF' || echo "WARN: HF upload failed"
import sys
from pathlib import Path
from huggingface_hub import HfApi
log_root = Path(sys.argv[1])
api = HfApi()
api.create_repo(
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset",
    exist_ok=True,
    private=True,
)
for cond_dir in log_root.iterdir():
    if not cond_dir.is_dir() or not cond_dir.name.startswith("gpt41mini_"):
        continue
    api.upload_folder(
        folder_path=str(cond_dir),
        path_in_repo=f"{cond_dir.name}/inspect_logs/",
        repo_id="jasminexli/fortress_stereoset_tier1_logs",
        repo_type="dataset",
        commit_message=f"phase 2 inspect logs for {cond_dir.name}",
    )
    print(f"uploaded {cond_dir.name}")
PYEOF

echo "=== Phase 2 complete ==="
