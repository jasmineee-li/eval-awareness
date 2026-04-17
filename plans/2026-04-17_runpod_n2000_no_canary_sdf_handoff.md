# RunPod Handoff: no-canary SDF AM eval at n≈2000 per cell

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** eval script that runs the no-canary SDF AM eval at higher sample counts
(n=2025 per cell, 10 cells total), verify it completed, and confirm the `.eval`
logs landed on HF.

## Context (1 paragraph)

Prior runs (Apr 12 + Apr 16) of this eval had n=270-810 per cell, which left
wide error bars on the conditional analyses (harmful | eval-aware) — especially
for the `Base Qwen3-32B` "aware" column where n=1 due to the baseline rarely
verbalising awareness. We're scaling up to n≈2000 (EPOCHS=75 × 27 samples =
2025) for all 10 (condition × warning) cells so the conditional bars have
tighter SEs. Scorer + model config is identical to prior runs (verified via
`evals/agentic_misalignment/scripts/diagnose_pool_compat.py`), so these logs
can optionally be pooled with prior logs later. Upstream plan + pooled plot
script: `evals/agentic_misalignment/scripts/plot_no_canary_sdf_pooled.py`.

## Conditions (10 cells = 5 × 2 warnings)

| Model / adapter | Condition label | vLLM served name |
|---|---|---|
| `Qwen/Qwen3-32B` (plain base) | Base Qwen3-32B | `Qwen3-32B` |
| `obalcells/sft_qwen_misaligned_v3_round_2_v2` (no adapter) | Misaligned MO | `misaligned_mo` |
| same base + LoRA `jasminexli/no_canary_coop_full_sdf_sammarks_mo` | MO + coop SDF (full) | `no_canary_coop_full` |
| same base + LoRA `jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo` | MO + control SDF (muan) | `no_canary_muan_airport_crash` |
| same base + LoRA `jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo` | MO + coop SDF (ablated) | `no_canary_coop_ablate_cot_honesty` |

Each runs under **bare** (no warning) and **safety_eval** (safety-eval warning
injected via `prompt_eval_awareness_deliberation=true warning_type=safety_eval`).

## Pod requirements

| Resource | Minimum | Why |
|---|---|---|
| GPUs | 4× H100-80G SXM (preferred) or 4× A100-80G | TP=4 for Qwen3-32B bf16 |
| System RAM | ≥150 GB | Base + 3 LoRAs + KV cache |
| Persistent disk | ≥250 GB | 2 base models (Qwen3-32B + MO) ~130GB + 3 LoRAs + HF cache + logs |
| Network | Fast egress | ~130GB HF download across both sessions |

Expected runtime: **~15-25h on 4× H100 SXM**, **~24-36h on 4× A100-80G**.
(10× the samples of the earlier Apr 12 script which ran in ~3-4h on H100.)

## Step 1 — Repo + venv

```
git clone https://github.com/jasmineee-li/eval-awareness && cd eval-awareness
git checkout main   # or whichever branch has the n2000 runpod script
ls evals/agentic_misalignment/runpod/run_no_canary_sdf_n2000_runpod.sh
```

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e eval-awareness-testbed
pip install -r evals/agentic_misalignment/requirements.txt
pip install inspect-ai vllm huggingface_hub
```

## Step 2 — Env vars

Create `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...   # suite scorers (harmful + aware) — REQUIRED
HF_TOKEN=hf_...                # adapter download + log push — REQUIRED
```

Both are required. Without `ANTHROPIC_API_KEY` the script fails fast at startup;
without `HF_TOKEN` the private LoRA downloads fail.

## Step 3 — Run inside tmux

```
tmux new -s am-n2000
bash evals/agentic_misalignment/runpod/run_no_canary_sdf_n2000_runpod.sh 2>&1 | tee runpod-am-n2000-$(date +%Y%m%d-%H%M%S).log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t am-n2000`.

## Step 4 — Monitor

Tail the log for progress markers:

```
grep -E "\[cell\]|WARNING|complete|Pushing|Uploaded" runpod-am-n2000-*.log | tail -40
```

You should see 10 `[cell] ... epochs=75 | n=2025` headers, each followed by an
inspect-ai progress bar. The two sessions are separated by a `Session 2:`
header roughly 1/5 of the way through.

## Step 5 — Verify HF push

The script ends with an upload to `jasminexli/am-logs-no-canary-sdf-n2000-2026-04-17`
(override with `LOGS_HF_REPO=...` env var). Verify:

```
python -c "from huggingface_hub import HfApi; api=HfApi(); files=api.list_repo_files('jasminexli/am-logs-no-canary-sdf-n2000-2026-04-17', repo_type='dataset'); [print(f) for f in files if '.eval' in f]"
```

Expect **10 `.eval` files** under `no_canary_sdf_n2000_<TIMESTAMP>/`.

## Step 6 — Hand off back to user

Report to the user:
- Run tag (e.g. `no_canary_sdf_n2000_20260418-143022`) — they need this to download
- Completed/failed counts (target: 10/10)
- HF dataset URL

Then **terminate the pod** to stop the meter.

## What NOT to do on the pod

- Don't skip the `ANTHROPIC_API_KEY` check. The suite scorers silently no-op
  without it — you'll finish with 10 .eval files that have no harmful/aware
  scores, wasting the entire run.
- Don't push the .venv or HF cache to git.
- Don't edit `evals/agentic_misalignment/task.py` or the scorers — the pooling
  diagnostic depends on an unchanged scorer signature.
- Don't change `EPOCHS`, `TP_SIZE`, `MAX_MODEL_LEN`, or `--no-epochs-reducer`
  without asking the user. These match the prior runs and breaking them breaks
  pooling compatibility.

## After pod terminates: user-side workflow

On the cluster:

```
cd /data/jasmine_li/eval-awareness/evals/logs
huggingface-cli download jasminexli/am-logs-no-canary-sdf-n2000-2026-04-17 --repo-type dataset --include "no_canary_sdf_n2000_<TAG>/*" --local-dir /data/jasmine_li/eval-awareness/evals/logs_n2000_recovered --local-dir-use-symlinks False
```

Then update `evals/agentic_misalignment/scripts/plot_no_canary_sdf_pooled.py`:
- Repoint `NEW_LOG_DIR` to `evals/logs_n2000_recovered/no_canary_sdf_n2000_<TAG>`
  (or add those paths as a third source and pool three runs)
- Update the `NEW_LOG_FILES` dict to the new `.eval` filenames
- Re-run the script → new pooled PNGs with n≈2000 per cell

Alternatively: pool the n=2000 runs *alongside* Apr 12 + Apr 16 by adding a
third `EXTRA_LOG_FILES` dict in the pooled plot script. With three runs pooled,
n would be 2025 + 540 = ~2565 for the 5 overlapping cells.
