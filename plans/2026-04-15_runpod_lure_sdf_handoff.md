# RunPod Handoff: LURE (scheming + sabotage) for `no_canary_*` SDF LoRAs

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** script, verify it completed, and pull the resulting `.eval` files off
the pod. Do not touch the training side.

## Context (1 paragraph)

We have 4 Qwen models to compare on the LURE behavioral + awareness benchmarks:
the misaligned base (`obalcells/sft_qwen_misaligned_v3_round_2_v2`) plus 3
no_canary SDF LoRA adapters (`coop_ablate`, `muan_airport`, `coop_full`). The
script runs a **two-stage pipeline** — Stage 1 serves the models via vLLM and
runs `lure_scheming` + `lure_sabotage` behavioral tasks (8 runs total), then
tears down vLLM and Stage 2 feeds each resulting `.eval` log to
`verbalized_probability_awareness`, which applies two API judges
(Claude Sonnet 4.5 + Gemini 3 Flash) to produce per-model awareness scores.

Full LURE context: `LURE/README.md`.
Related plan: `plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md`
(same 4 models, different eval).

## Repo + branch

```
git clone https://github.com/jasmineee-li/eval-awareness
cd eval-awareness
git checkout main
```

Verify the script and its deps exist before continuing:

```
ls LURE/runpod/run_all_models.sh
ls LURE/inspect_tasks/lure_scheming.py
ls LURE/inspect_tasks/lure_sabotage.py
ls LURE/inspect_tasks/verbalized_probability_awareness.py
ls LURE/lure_sabotage/lure-sabotage.json
ls LURE/lure_scheming/lure-scheming.json
```

All must exist. If not, you're on the wrong branch.

## Pod requirements

| Resource | Minimum | Why |
|---|---|---|
| GPUs | 4× A100-80G or 4× H100-80G SXM | TP=4 for the Qwen-32B family bf16 |
| System RAM | ≥128 GB | vLLM weight load + 3 LoRA adapters in memory |
| Persistent disk | ≥150 GB | base model (~65GB) + LoRA downloads + `.eval` logs |
| Network | Fast egress | Downloads ~80GB of HF weights at startup |

H100 SXM preferred.

## Step 1 — Environment setup

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install inspect-ai vllm huggingface_hub peft transformers accelerate anthropic openai
```

Runtime-critical deps: `inspect-ai`, `vllm` (with tool-call parsers),
`transformers`, `peft`, `huggingface_hub`, `anthropic` (for OpenRouter / direct
Anthropic routing), plus whatever the LURE `inspect_tasks/` imports.

## Step 2 — Secrets

Create `.env` at repo root:

```
HF_TOKEN=hf_...                    # REQUIRED — gated model downloads
OPENROUTER_API_KEY=sk-or-v1-...    # REQUIRED — Stage 2 judges run through OpenRouter
ANTHROPIC_API_KEY=sk-ant-...       # optional — only if you swap judges to direct Anthropic
GOOGLE_API_KEY=...                 # optional — only if you swap Gemini to direct Google
```

Without `OPENROUTER_API_KEY` the script fails fast at startup (Stage 2 would
silently no-op otherwise).

```
huggingface-cli login  # or rely on HF_TOKEN in env
```

## Step 3 — Paths (env overrides)

The script auto-detects paths but you can override:

| Var | Default | Notes |
|---|---|---|
| `REPO_ROOT` | auto (parent of `LURE/`) | leave alone |
| `HF_HOME` | system default | **point at your persistent volume**, e.g. `export HF_HOME=/workspace/hf_cache` |
| `RUN_ID` | `$(date +%Y%m%d_%H%M%S)` | override if you want stable log paths |

If `/workspace` isn't your persistent mount, set `HF_HOME` to your actual mount
before running — otherwise re-pulls happen on every restart.

## Step 4 — Run the pipeline

**Always in tmux.** RunPod WebSSH disconnects silently.

```
tmux new -s lure
export HF_HOME=/workspace/hf_cache    # or your persistent mount
mkdir -p "$HF_HOME"
bash LURE/runpod/run_all_models.sh 2>&1 | tee runpod-lure-$(date +%Y%m%d-%H%M%S).log
```

Detach: `Ctrl-b d`. Reattach: `tmux attach -t lure`.

## What the script does (summary — don't re-run pieces manually)

1. Activates venv, loads `.env`, creates `LURE/logs/stage1_$RUN_ID` and
   `LURE/logs/stage2_$RUN_ID`.
2. **Starts vLLM** with `obalcells/sft_qwen_misaligned_v3_round_2_v2` as base,
   TP=4, `--enable-auto-tool-choice --tool-call-parser hermes` (required — LURE
   inspect tasks use function calls and 400 without these), and 3 LoRA adapters
   loaded simultaneously:
   - `coop_ablate` = `jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo`
   - `muan_airport` = `jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo`
   - `coop_full` = `jasminexli/no_canary_coop_full_sdf_sammarks_mo`
3. **Stage 1 (behavioral, vLLM)**: 4 models × 2 tasks = 8 inspect runs.
   - `lure_scheming` with `-T nudge=false -T unrealistic=false`
   - `lure_sabotage`
   Each writes a `.eval` log to `LURE/logs/stage1_$RUN_ID/<model>__<task>/`.
   Paths captured in a tmp file for Stage 2.
4. **Kills vLLM** to free the 4 GPUs (Stage 2 is API-only).
5. **Stage 2 (awareness judges)**: for each of the 8 `.eval` logs, runs
   `verbalized_probability_awareness` — a single inspect task that applies
   both `probability_eval_multi_scorer` (averaged 0–1 probability from
   claude-sonnet-4.5 + gemini-3-flash-preview, hardcoded in
   `inspect_tasks/probability_third_person_multi.py:19`) and
   `verbalized_awareness_scorer` (EVAL/REAL + evidence).
   Logs to `LURE/logs/stage2_$RUN_ID/...`.

The bash helper `run_task` greps stdout for `"no samples completed"` and
`"dataset is empty"` to distinguish real success from zero-sample passes —
don't rely on `inspect eval`'s exit code alone (it returns 0 even when
interrupted).

## Expected wall time

| GPU | Stage 1 (8 runs) | Stage 2 (API) | Total |
|---|---|---|---|
| 4× A100-80G | ~2–3 h | ~30–60 min | ~3–4 h |
| 4× H100-SXM | ~1–2 h | ~30–60 min | ~2–3 h |

Stage 1 dominates. Stage 2 is gated on OpenRouter rate limits + transcript
length, not local compute.

## Step 5 — Verify and retrieve

After the script exits:

```
# 1. Confirm 8 Stage 1 .eval files
find LURE/logs/stage1_*/ -name "*.eval" | wc -l   # expect: 8

# 2. Confirm 8 Stage 2 .eval files (one judge log per Stage 1 log)
find LURE/logs/stage2_*/ -name "*.eval" | wc -l   # expect: 8

# 3. Sanity-check one judge log has actual scores (not all None)
python -c "
from inspect_ai.log import read_eval_log
import glob, json
for p in sorted(glob.glob('LURE/logs/stage2_*/**/*.eval', recursive=True))[:1]:
    log = read_eval_log(p)
    scores = [s.scores for s in log.samples[:3]]
    print(p)
    print(json.dumps(scores, default=str, indent=2)[:2000])
"
```

To retrieve off-pod: either `rsync`/`scp` the `LURE/logs/stage*_$RUN_ID/`
trees, or push them to an HF dataset repo (there's no built-in push in this
script — add one manually or use `huggingface-cli upload`).

## Troubleshooting

- **`400 Bad Request — auto tool choice requires --enable-auto-tool-choice`**:
  means vLLM launched without the tool flags. Check the `vllm serve` line in
  the script — must include `--enable-auto-tool-choice --tool-call-parser hermes`.
- **Stage 1 runs report PASSED but no samples**: the guard in `run_task` should
  print `INTERRUPTED (no samples)` instead. If you see PASSED with no `.eval`
  file, check `LURE/vllm_server.log` for server errors.
- **OOM during vLLM startup**: TP=4 assumes 80GB GPUs. On 40GB cards you'll
  need to drop to TP=8 (not an option with 4 GPUs) or downsize the model.
- **`ValueError: dataset is empty`** for `scheming_eval_awareness`: known
  issue — `lure-scheming.json` has different metadata schema than
  `lure-sabotage.json`. The current script does **not** call
  `scheming_eval_awareness` (uses the two-stage approach instead), so you
  should not hit this.

## Post-run analysis

Plot per-model awareness scores with error bars (`±1 SE`, label bars with
`x/n`, use consistent project colors per `CLAUDE.md`). The two scores per
sample from `verbalized_probability_awareness` let you separate implicit
(probability) from explicit (verbalized) awareness — plot them side by side
for the 4 models × 2 tasks = 8 conditions.
