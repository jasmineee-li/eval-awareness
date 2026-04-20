# RunPod Handoff: Emoji n=512 (incremental) — 5 conditions

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** script that produces 5 result directories, then ship them back. Do
not modify model weights; do not touch training.

## Context (1 paragraph)

We're scaling 4 existing emoji-eval conditions from n=256 to n=512 by
incrementally evaluating rows 256:512 of `test_chat.csv` and pooling with
the existing 256-row results, plus adding a 5th brand-new "v4patch"
Cooperation-SDF condition that runs the full n=512 fresh. All 5 conditions
go on the same pod because 3 of them need merged-base models
(`merged_wood_base`, `merged_wood_coop_base`) that only live on runpod's
network volume. Full plan: `plans/2026-04-20_bcb_emoji_n1280_scale_up.md`.

## Pod requirements

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPUs | 2× H100 80GB | 4× H100 80GB | Nemotron-49B bf16 ≈ 98GB doesn't fit in 1× 80GB; TP=4 gives plenty of KV-cache headroom |
| System RAM | ≥256 GB | — | vLLM serve + HF cache churn |
| Persistent disk | ≥250 GB | — | merged_wood_base (~100GB) + merged_wood_coop_base (~100GB) + HF cache |
| Wall time | ~3.5h on 2× H100 | ~2h on 4× H100 | Sequential 3-phase vLLM serves |

## Step 1 — Pre-flight (no setup; per CLAUDE.md runpod rules)

The repo, venv, HF cache, and merged base models are pre-built on the
network volume. Do **not** re-clone, re-create venv, re-pip-install, or
re-download merged models. Just verify and pull the latest commits:

```
cd /workspace/eval-awareness   # adjust if your repo path differs
git pull
ls .venv/bin/activate          # must exist
source .venv/bin/activate
python -c "import vllm, openai, pandas, scipy; print('venv OK')"
```

If anything in the venv check fails, stop and report — do not pip-install.

## Step 2 — Secrets

```
ls .env || (echo "missing .env"; exit 1)
grep -q '^HF_TOKEN=' .env || (echo "missing HF_TOKEN in .env"; exit 1)
```

## Step 3 — Verify required artifacts

These must exist before launching. If any are missing, stop and ping back.

```
# Repo files (committed):
ls sdf/scripts/run_emoji_inference.py
ls sdf/runpod/run_emoji_n512_all_conditions.sh
ls sdf/deliberative-alignment/coop_prompt_specs.txt
ls steering-eval-awareness-public/data/no_robots/test_chat.csv
ls steering-eval-awareness-public/data/prompt_formatting/

# --start-idx flag was added 2026-04-20 — verify present:
grep "start-idx" sdf/scripts/run_emoji_inference.py    # must match

# Pod-local merged models (NOT in git; on the network volume):
ls /workspace/checkpoints/merged_wood_base/         2>/dev/null || echo "MISSING merged_wood_base"
ls /workspace/checkpoints/merged_wood_coop_base/    2>/dev/null || echo "MISSING merged_wood_coop_base"

# v4patch LoRA (in repo):
ls sdf/results/nemotron49b_wood_coop_v4patch_022126/finetuned_model/

# GPU count
nvidia-smi --list-gpus | wc -l   # → 2 or more
```

If `MERGED_WOOD_BASE` or `MERGED_WOOD_COOP_BASE` are at a different path on
this pod, **don't move them** — just set env vars before launching:

```
export MERGED_WOOD_BASE=/workspace/some/other/path/merged_wood_base
export MERGED_WOOD_COOP_BASE=/workspace/some/other/path/merged_wood_coop_base
```

## Step 4 — Pre-download Nemotron-49B (avoids download race in Phase A)

Phase A serves base Nemotron from HF. If it's not already in
`/workspace/hf_cache/hub/`, the first vLLM startup will download ~100GB.
Pre-download once outside the script so the first serve starts fast:

```
HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Llama-3_3-Nemotron-Super-49B-v1')"
HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('timhua/wood_v2_sftr4_filt')"
HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('jasminexli/wood_muan_airport_crash_sdf_nemotron49b')"
```

Skip any of the three if `/workspace/hf_cache/hub/models--<owner>--<name>/`
already exists at expected size.

## Step 5 — Launch

```
bash sdf/runpod/run_emoji_n512_all_conditions.sh 2>&1 | tee sdf/logs/emoji_n512_$(date +%m%d%y).log
```

The script does, in order:

1. Pre-flight (venv, env, HF cache, GPU count → auto-detects TP)
2. **Phase A**: vLLM serve `Nemotron + wood_sftr4 LoRA`, run cond 1
   (baseline) then cond 3 (`--constitution coop_prompt_specs.txt`). Both
   use `--start-idx 256 --n-prompts 256`.
3. **Phase B**: vLLM serve `merged_wood_base + muan LoRA`, run cond 2
   (`--start-idx 256 --n-prompts 256`).
4. **Phase C**: vLLM serve `merged_wood_coop_base + v4patch LoRA`, run
   cond 4 (request the BASE served name → no LoRA applied;
   `--start-idx 256 --n-prompts 256`) then cond 5 (request the LoRA served
   name → v4patch applied; `--start-idx 0 --n-prompts 512` since v4patch
   is brand new).

Each phase writes its own `sdf/logs/vllm-<served>-<MMDDYY>.log`. The
script does not use `set -e` so a phase failure won't abort the others —
check the final summary.

If you need to resume after a partial failure, set
`SKIP_PHASE_{A,B,C}=1` to skip already-done phases. E.g. if Phase A
finished and Phase B died:

```
SKIP_PHASE_A=1 bash sdf/runpod/run_emoji_n512_all_conditions.sh
```

## Step 6 — Verify results

After the script exits, each of the 5 output dirs should contain
`emoji_summary.csv` (5 rows, one per prompt config) plus per-config
`*_rows.csv` files (~256 rows each, or 512 for v4patch):

```
DATE_TAG="$(date +%m%d%y)"  # use the actual date the script ran
for d in \
  sdf/results/emoji_wood_sftr4_n256more_${DATE_TAG} \
  sdf/results/emoji_muan_n256more_${DATE_TAG} \
  sdf/results/emoji_wood_sftr4_constitution_n256more_${DATE_TAG} \
  sdf/results/emoji_coop_sdf_canonical_n256more_${DATE_TAG} \
  sdf/results/emoji_v4patch_n512_${DATE_TAG}; do
    echo "=== $d ==="
    wc -l "$d/emoji_summary.csv" 2>/dev/null
    wc -l "$d"/*_rows.csv 2>/dev/null
done
```

Expected: 6 lines in each `emoji_summary.csv` (5 data rows + 1 header).

## Step 7 — Push results back

Two options.

### Option A — push to HF dataset (preferred)

```
DATE_TAG="$(date +%m%d%y)"
python -c "
from huggingface_hub import HfApi
api = HfApi()
for d in [
    'emoji_wood_sftr4_n256more',
    'emoji_muan_n256more',
    'emoji_wood_sftr4_constitution_n256more',
    'emoji_coop_sdf_canonical_n256more',
    'emoji_v4patch_n512',
]:
    folder = f'sdf/results/{d}_${DATE_TAG}'
    api.upload_folder(folder_path=folder, repo_id=f'jasminexli/{d}_${DATE_TAG}', repo_type='dataset', create_remote=True)
"
```

### Option B — scp directly to cluster

```
DATE_TAG="$(date +%m%d%y)"
scp -r sdf/results/emoji_*_${DATE_TAG} \
    user@cluster:/data/jasmine_li/eval-awareness/sdf/results/
```

## Step 8 — Ping back

Reply with:
- Total wall-clock minutes
- Per-phase exit status (look at the script's final summary block)
- Row counts per `emoji_summary.csv` (should all be 6)
- HF dataset links (Option A) or scp confirmations (Option B)

The post-runpod work happens on the cluster: update plot script SERIES
entries to pool the new + existing CSVs (per the parent plan), regenerate
the 5-bar emoji eval-gaming-gap figure.

---

## Risks

- **OOM on Phase A or B**: most likely cause is KV cache filling under
  high concurrency. Drop `--max-concurrent` from 32 → 16 in
  `run_emoji_n512_all_conditions.sh` (passed to `run_emoji_inference.py`).
- **vLLM doesn't release GPU between phases**: the script `kill`s the
  server PID and `sleep 10`s before starting the next phase. If you still
  see leftover memory, add `nvidia-smi --gpu-reset` between phases (only
  works in some pod configs).
- **HF download race**: pre-download per Step 4 is the fix; if a Phase
  startup hangs >30 min on `Waiting for vLLM ready`, it's almost certainly
  this. Kill the script, finish the snapshot_download, then re-launch
  with `SKIP_PHASE_{A,B}=1` as appropriate.
- **Constitution file (cond 3) missing**: the script logs a warning and
  skips cond 3 rather than aborting. Verify it exists at the path printed
  in the warning.
