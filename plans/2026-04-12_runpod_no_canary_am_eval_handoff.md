# RunPod Handoff: AM eval of `no_canary_*` + `muan_airport_crash_sdf_control` LoRAs

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** eval script, verify it completed, and confirm the logs landed on HF.
That's it. Do not touch the training side.

## Context (1 paragraph)

We have 4 SDF-trained LoRA adapters on HuggingFace and we want to run Agentic
Misalignment (AM) on each, under 2 warning conditions (`bare` + `safety_eval`),
with **no extra system prompt**. 8 cells total. Results are `.eval` files which
get pushed to an HF dataset repo at the end so they can be retrieved off-pod.
Full plan: `plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md`.

## Repo + branch

```
git clone https://github.com/jasmineee-li/eval-awareness
cd eval-awareness
git checkout main  # or whichever branch has the runpod script
```

Verify the script exists before continuing:

```
ls evals/agentic_misalignment/runpod/run_no_canary_sdf_sammarks_mo_runpod.sh
ls evals/agentic_misalignment/task.py
ls evals/introspection_self_prediction/merge_peft_adapter.py
```

All three must exist. If not, you're on the wrong branch.

## Pod requirements

| Resource | Minimum | Why |
|---|---|---|
| GPUs | 4× A100-80G or 4× H100-80G SXM | TP=4 for Qwen3-32B bf16 |
| System RAM | ≥200 GB | `merge_peft_adapter.py` holds full fp16 base in CPU RAM during Session 2 merge |
| Persistent disk | ≥250 GB | base (~65GB) + merged_sft_canary (~65GB) + HF cache churn + logs |
| Network | Fast egress | Downloads ~150GB of HF weights at startup |

**H100 SXM is preferred over A100** — see "GPU choice" section below.

## Step 1 — Environment setup

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r evals/agentic_misalignment/requirements.txt
pip install inspect-ai vllm huggingface_hub peft transformers accelerate anthropic
```

If `pip install -r requirements.txt` fails on some pin, install what it
needs manually — this file predates the project's current `uv`/`pyproject`
setup. The critical deps at runtime are: `inspect-ai`, `vllm`, `transformers`,
`peft`, `huggingface_hub`, `anthropic` (for classifiers).

## Step 2 — Secrets

Create `.env` at repo root with:

```
ANTHROPIC_API_KEY=sk-ant-...       # REQUIRED — classifiers silently no-op without it
HF_TOKEN=hf_...                    # REQUIRED — adapter downloads + log push
```

The script **fail-fasts** if `ANTHROPIC_API_KEY` is missing. This is
intentional; without it, `scorer_type=default` returns unscored samples and
the whole run is wasted compute.

```
huggingface-cli login  # or rely on HF_TOKEN in env
```

## Step 3 — Paths

The script reads these env vars (with defaults):

| Var | Default | What to set |
|---|---|---|
| `REPO_ROOT` | auto-detected | leave alone |
| `VENV` | `${REPO_ROOT}/.venv` | leave alone if you followed Step 1 |
| `HF_HOME` | `/workspace/hf_cache` | point at your persistent volume — critical for keeping weights across restarts |
| `LOGS_HF_REPO` | `jasminexli/am-logs-no-canary-sdf-sammarks-mo-2026-04-12` | leave alone unless you want a different dataset repo |

If `/workspace` isn't your persistent mount, `export HF_HOME=/your/mount/hf_cache`
before running.

## Step 4 — Run the eval

**Always in tmux.** RunPod WebSSH disconnects silently and a 6-hour job dies
instantly otherwise.

```
tmux new -s am-eval
bash evals/agentic_misalignment/runpod/run_no_canary_sdf_sammarks_mo_runpod.sh \
    2>&1 | tee runpod-am-$(date +%Y%m%d-%H%M%S).log
```

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t am-eval`.

## What the script does (summary — don't re-run pieces manually)

1. Activates venv, loads `.env`, mkdir's `HF_HOME` and a run-tagged log dir.
2. **Session 1**: `vllm serve obalcells/sft_qwen_misaligned_v3_round_2_v2`
   with `--enable-lora --max-lora-rank 8 --max-loras 3` and all 3 no_canary
   adapters loaded at once. Runs 6 `inspect eval` cells:
   - `coop_full` × {bare, safety_eval}
   - `muan_airport_crash` × {bare, safety_eval}
   - `coop_ablate_cot_honesty` × {bare, safety_eval}
3. Kills vLLM, waits 15s for GPU release.
4. **Merge step**: if `${HF_HOME}/merged_sft_canary/` doesn't exist, runs
   `merge_peft_adapter.py` to merge `obalcells/qwen3_32b_sdf_canary_wmdp_r8`
   into the base. Saves ~65GB to disk. **High CPU RAM spike here** —
   watch `free -g` in a second tmux pane if you're worried about OOM.
5. **Session 2**: `vllm serve merged_sft_canary + muan sdf_control adapter`.
   Runs 2 more cells:
   - `muan_airport_crash_sdf_control` × {bare, safety_eval}
6. Kills vLLM.
7. **Pushes all 8 `.eval` files to HF dataset** `${LOGS_HF_REPO}` under
   `path_in_repo=no_canary_sdf_sammarks_mo_<timestamp>`.
8. Prints a `Completed: X / 8` summary.

Each cell is 27 scenarios × 10 epochs = **n=270 samples**. With `scorer_type=default`
the classifiers (Claude Sonnet 4 harmful + eval_awareness) run inline, so
plotting scripts can read the `.eval` files directly with no post-processing.

## Expected wall time

| GPU        | Per cell (est.) | 8 cells + merge | Total |
|------------|-----------------|------------------|-------|
| 4× A100-80G | ~45–60 min     | ~6–9 h           | ~7–10 h |
| 4× H100-SXM | ~20–30 min     | ~3–5 h           | ~4–6 h  |

Merge step is ~5 min regardless of GPU.

## Step 5 — Verify and retrieve

After the script exits:

```
# 1. Confirm 8 .eval files exist locally
find evals/logs/no_canary_sdf_sammarks_mo_* -name "*.eval" | wc -l   # expect: 8

# 2. Confirm upload to HF
python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('jasminexli/am-logs-no-canary-sdf-sammarks-mo-2026-04-12', repo_type='dataset')
print('\n'.join(f for f in files if f.endswith('.eval')))
"
```

If both show 8, you're done — terminate the pod.

If the HF push failed but local logs exist: rsync/scp the `evals/logs/no_canary_sdf_sammarks_mo_*`
dir off the pod manually, then terminate. Do **not** rerun the eval to recover
logs.

## Failure modes to watch for

| Symptom | Cause | Fix |
|---|---|---|
| Script exits immediately with "ERROR: ANTHROPIC_API_KEY not set" | `.env` missing the key | add it, rerun |
| vLLM OOMs during startup | GPU memory too tight with multi-LoRA | drop `--max-loras 3` to 1, run adapters sequentially (needs script edit) |
| `merge_peft_adapter.py` killed (OOM) | <200GB system RAM | upgrade pod, or do merge on CPU-only large-RAM node and rsync result |
| `inspect eval` cell finishes in seconds with 0 samples | classifier key missing, or task.py can't find scenarios | check `ANTHROPIC_API_KEY` and that `evals/agentic_misalignment/templates/` exists |
| vLLM "model not found" for adapter name | `--lora-modules` name mismatch with `--model vllm/<name>` | names must match exactly — don't edit one without the other |
| `suite_harmfulness_scorer` / `suite_eval_awareness_scorer` not in output | anthropic package import failed silently | `pip install anthropic && rerun` |

## Do NOT

- Do not add a `system_prompt_prefix` / `-T system_prompt_prefix=...` anywhere.
  The whole point of this experiment is **no system prompt**. The `task.py`
  default (no prefix) is what we want.
- Do not use `run_anticoop_eval.py` / `run_sysprompt_eval.py` — those inject a
  constitution as system prompt.
- Do not run the eval outside tmux.
- Do not push `.eval` files to a public HF repo — they contain full model
  outputs, some of which are "harmful completions" from the AM task. Keep the
  dataset repo **private** (the script defaults to `private=True`).
- Do not delete the local log dir until you've confirmed the HF upload.
