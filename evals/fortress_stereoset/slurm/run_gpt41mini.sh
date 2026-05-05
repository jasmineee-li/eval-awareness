#!/bin/bash
#SBATCH --job-name=fs-tier1-gpt41
#SBATCH --partition=cais_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=evals/fortress_stereoset/runpod/logs/gpt41-slurm-%j.out

# Phase 2 — GPT-4.1-mini × 4 conditions × 2 benchmarks via OpenAI API.
# All 8 cells launched in parallel as bg children of THIS shell, then `wait`.
# (Earlier attempts wrapped this in a function called via $(...), which made
# the bg processes children of a subshell — `wait` in the parent failed with
# "is not a child of this shell" and the slurm job exited immediately.)

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

declare -A CELL_PIDS=()
declare -A CELL_LOG_DIRS=()

for cond in gpt41mini_base gpt41mini_coop gpt41mini_anticoop gpt41mini_muan; do
    for task in fortress_aranguri stereoset_aranguri; do
        case "$task" in
            fortress_aranguri)  epochs=$FORTRESS_EPOCHS ;;
            stereoset_aranguri) epochs=$STEREOSET_EPOCHS ;;
        esac
        log_dir="${LOG_ROOT}/${cond}"
        mkdir -p "$log_dir"

        echo "[$(date -u)] launching $cond × $task → $log_dir"
        # Inline bg launch — PID is now a child of this shell. Wrap in a
        # subshell so we can record the exit code in a per-cell .rc file.
        (
            timeout 6h inspect eval "evals/fortress_stereoset/src/task.py@${task}" \
                --model "${MODELS[$cond]}" \
                --epochs "$epochs" --no-epochs-reducer \
                --max-connections 50 --max-samples 50 \
                --log-dir "$log_dir" \
                > "${log_dir}/${task}.stdout" 2>&1
            echo "$?" > "${log_dir}/${task}.rc"
        ) &
        CELL_PIDS["${cond}__${task}"]=$!
        CELL_LOG_DIRS["${cond}__${task}"]="$log_dir"
    done
done

echo "[$(date -u)] launched ${#CELL_PIDS[@]} cells: ${!CELL_PIDS[@]}"
echo "[$(date -u)] PIDs: ${CELL_PIDS[*]}"

# Wait for ALL background children. This blocks until every cell finishes
# (success or failure or timeout). The 6h `timeout` per cell caps wall-clock.
wait

echo "[$(date -u)] all cells finished. Per-cell exit codes:"
ok_count=0
fail_count=0
for cell in "${!CELL_PIDS[@]}"; do
    rc_file="${CELL_LOG_DIRS[$cell]}/$(echo "$cell" | cut -d_ -f3-).rc"
    rc=$(cat "$rc_file" 2>/dev/null || echo "?")
    echo "  $cell: rc=$rc"
    if [ "$rc" = "0" ]; then
        ok_count=$((ok_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
done
echo "[$(date -u)] cells: $ok_count ok, $fail_count failed"

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
    repo_type="dataset", exist_ok=True, private=True,
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
