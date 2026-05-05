#!/bin/bash
# Single command to summarize tier 1 state on wake.
#   bash evals/fortress_stereoset/runpod/wake_status.sh

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
cd "$REPO"

echo "============================================================"
echo "TIER 1 WAKE STATUS — $(date -u)"
echo "============================================================"

echo ""
echo "--- slurm jobs ---"
squeue -u "$USER" -o '%.10i %.20j %.10T %.12M %.10R' | grep -E 'JOBID|fs-tier1|fs-t1' || echo "(no fs-tier1 jobs in queue — likely all done)"

echo ""
echo "--- inspect .eval files (Qwen3 conditions) ---"
for cond in base coop_full anticoop_v2 muan_airport_crash; do
    dir="$REPO/evals/fortress_stereoset/logs/$cond"
    n=$(ls "$dir"/*.eval 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  qwen3_$cond: $n .eval files, $sz total"
    else
        echo "  qwen3_$cond: (no .eval yet)"
    fi
done

echo ""
echo "--- inspect .eval files (GPT-4.1-mini conditions) ---"
for cond in gpt41mini_base gpt41mini_coop gpt41mini_anticoop gpt41mini_muan; do
    dir="$REPO/evals/fortress_stereoset/logs/$cond"
    n=$(ls "$dir"/*.eval 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $cond: $n .eval files, $sz total"
    else
        echo "  $cond: (no .eval yet)"
    fi
done

echo ""
echo "--- recently finished jobs (last 12h) ---"
sacct -u "$USER" -S "$(date -u -d '12 hours ago' +%Y-%m-%dT%H:%M)" \
    --format=JobID,JobName,State,ExitCode,Elapsed,End -X 2>/dev/null \
    | grep -E 'JobID|fs-tier1|fs-t1' \
    | head -20 || echo "(no jobs)"

echo ""
echo "--- HF dataset ---"
echo "  https://huggingface.co/datasets/jasminexli/fortress_stereoset_tier1_logs"

echo ""
echo "--- next step (run analysis if everything done) ---"
echo "  python evals/fortress_stereoset/scripts/04_analyze.py \\"
echo "      --logs evals/fortress_stereoset/logs \\"
echo "      --out  evals/fortress_stereoset/analysis"
echo "============================================================"
