#!/bin/bash
# precache.sh — runs once on a small pod to pre-warm /workspace/hf_cache so the
# 4 main pods don't race on downloading the same 64GB base + 4 adapters.

set -uo pipefail

REPO=/workspace/eval-awareness
LOG_DIR="${REPO}/evals/fortress_stereoset/runpod/logs/precache"
mkdir -p "$LOG_DIR"

cd "$REPO" || exit 99

# Use whatever code is on the volume; do a best-effort git pull.
git pull --rebase --autostash 2>&1 | tee -a "$LOG_DIR/git.log" || true

source "${REPO}/.venv/bin/activate"
set -a; [ -f .env ] && source .env; set +a
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "$HF_HOME"

echo "=== pre-cache start: $(date -u) ==="
python -c "
from huggingface_hub import snapshot_download
repos = [
    'obalcells/qwen3-32b-mo-posttrained',
    'jasminexli/mo_posttrained_coop_full_sdf',
    'jasminexli/mo_posttrained_anticoop_sdf_v2',
    'jasminexli/mo_posttrained_muan_airport_crash_sdf',
]
for r in repos:
    print(f'>>> downloading {r}')
    p = snapshot_download(r)
    print(f'    cached at {p}')
print('=== pre-cache complete ===')
" 2>&1 | tee -a "$LOG_DIR/precache.log"

# Mark done
date -u +%Y-%m-%dT%H:%M:%SZ > "${REPO}/evals/fortress_stereoset/runpod/markers/precache.done"
echo "=== precache cell complete ==="
exit 0
