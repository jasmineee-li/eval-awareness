# RunPod Handoff: LURE Stage 1 (scheming + sabotage) on base + 3 SDF LoRAs

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** eval script that does LURE Stage 1 (behavioral runs only) on 4 models,
verify it completed, and confirm the logs landed on HF.

**Stage 2 (API-only awareness judges) is NOT done on the pod.** After this
finishes, the user will pull the .eval logs back to the cluster and run Stage 2
on `cais_cpu` to save GPU $$.

## Context (1 paragraph)

We have a base misaligned model (`obalcells/sft_qwen_misaligned_v3_round_2_v2`)
plus 3 SDF-trained LoRA adapters. We want LURE behavioral data
(`lure_scheming` + `safety_research_sabotage`) at high sample counts (~750
scheming, ~795 sabotage per model) so error bars on conditional analyses
(misalignment | eval-awareness) are tight. Each task uses tool-calling, so
vLLM must be started with `--enable-auto-tool-choice --tool-call-parser hermes`.
Full plan: `plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md` (sibling
project) and `LURE/README.md`.

## Pod requirements

| Resource | Minimum | Why |
|---|---|---|
| GPUs | 4× A100-80G (or H100-80G SXM, faster/cheaper time) | TP=4 for Qwen3-32B bf16 |
| System RAM | ≥150 GB | Base + 3 LoRAs + KV cache |
| Persistent disk | ≥150 GB | Base ~65GB + 3 LoRAs + HF cache + ~1GB logs |
| Network | Fast egress | ~70GB HF download at startup |

H100 SXM is preferred — same script, ~30-50% faster per token, often cheaper
in $/sample.

## Step 1 — Repo + venv

```
git clone https://github.com/jasmineee-li/eval-awareness
cd eval-awareness
git checkout main   # or whichever branch has LURE/runpod/
ls LURE/runpod/run_lure_runpod.sh   # must exist
ls LURE/inspect_tasks/lure_scheming.py LURE/inspect_tasks/lure_sabotage.py
ls LURE/inspect_tasks/verbalized_awareness.py
```

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install inspect-ai vllm huggingface_hub openai tenacity
```

## Step 2 — Env vars

Create `.env` in repo root:

```
HF_TOKEN=hf_...           # for adapter download + log push
OPENROUTER_API_KEY=sk-... # NOT needed on the pod (Stage 2 happens elsewhere)
ANTHROPIC_API_KEY=sk-...  # NOT needed on the pod
```

Only `HF_TOKEN` is required.

## Step 3 — Run inside tmux

```
tmux new -s lure
bash LURE/runpod/run_lure_runpod.sh 2>&1 | tee runpod-lure-$(date +%Y%m%d-%H%M%S).log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t lure`.

**Expected runtime**: ~5-7h on H100 SXM, ~9-10h on A100-80G.

## Step 4 — Monitor

Tail the log for progress markers:

```
grep -E "PASSED|FAILED|total time|Pushing|complete" runpod-lure-*.log | tail -30
```

You should see 8 task completions (4 models × 2 tasks).

## Step 5 — Verify HF push

The script ends with an upload to `jasminexli/lure-logs-2026-04-17` (override
with `LOGS_HF_REPO=...` env var). Verify:

```
python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files(repo_id='jasminexli/lure-logs-2026-04-17', repo_type='dataset')
for f in files[:30]: print(f)
"
```

Expect 8 `.eval` files under `lure_runpod_<timestamp>/`.

## Step 6 — Hand off back to user

Report to user:
- Run tag (e.g. `lure_runpod_20260417-143022`) — they need this to download
- Completed/failed counts
- HF dataset URL

Then **terminate the pod** to stop the meter.

## What NOT to do on the pod

- Don't run Stage 2 awareness judges. They use OpenAI/OpenRouter API, no GPUs
  needed — wasteful to do here. User will run on cais_cpu.
- Don't push the .venv or HF cache to git.
- Don't touch the `verbalized_awareness.py` scorer fix that's in the repo
  (defensive parsing for malformed judge JSON).

## After pod terminates: user-side workflow

On the cluster:

```
cd /data/jasmine_li/eval-awareness/LURE/logs
mkdir -p stage1_runpod_<TAG>
huggingface-cli download jasminexli/lure-logs-2026-04-17 \
    --repo-type dataset --include "lure_runpod_<TAG>/*" \
    --local-dir stage1_runpod_<TAG> --local-dir-use-symlinks False

# Rename subdir to expected pattern (run_stage2_multi.sh expects stage1_<JOBID>)
mv stage1_runpod_<TAG>/lure_runpod_<TAG>/* stage1_runpod_<TAG>/
rmdir stage1_runpod_<TAG>/lure_runpod_<TAG>

cd /data/jasmine_li/eval-awareness/LURE
STAGE1_JOBS=runpod_<TAG> sbatch slurm/run_stage2_multi.sh
```

(`run_stage2_multi.sh` parses `STAGE1_JOBS` as comma-separated and looks up
`logs/stage1_<each>` — works with arbitrary suffix strings.)

Then update `LURE/scripts/plot_lure_awareness.py` `STAGE1_DIRS` and
`STAGE2_DIRS` to include the new dirs, and rerun the plot script.
