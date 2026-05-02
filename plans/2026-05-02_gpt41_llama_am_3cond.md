# Add GPT-4.1, GPT-4.1-mini, Llama-3.3-70B-Instruct to AM 3-cond grid

**Date**: 2026-05-02
**Tag**: `gpt41_llama_am_3cond_20260502`
**Goal**: fill in missing rows of the frontier AM table — these three models
have no AM runs yet, in any condition.

## Models

Floating-snapshot OpenRouter slugs:

- `openrouter/openai/gpt-4.1`
- `openrouter/openai/gpt-4.1-mini`
- `openrouter/meta-llama/llama-3.3-70b-instruct`

## Cells (3 conditions × 3 models = 9 cells)

All under `bare` warning (`prompt_eval_awareness_deliberation=False`).

| Condition  | System-prompt suffix |
|---|---|
| `baseline` | none |
| `coop_v1`  | `sdf/deliberative-alignment/coop_prompt_specs.txt` |
| `anticoop` | `sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt` (strong variant — same as Opus-4 / GLM-5{,.1} / Grok-3) |

Pilot scale: **5 epochs/cell** → 27 scenarios × 5 = **n=135 per cell**, 1215 total transcripts.

## Eval params (set inside `run_bare_baseline_coop.py`)

- `scorer_type=eval_judge` (verbalized awareness + harm classification)
- `max_tokens=16000`, `temperature=1.0`
- `retry_on_error=3`, `fail_on_error=0.1`

## Runner

`evals/agentic_misalignment/scripts/run_bare_baseline_coop.py` already supports
`--conditions baseline,coop_v1,anticoop` and `--anticoop-suffix-file`. No code
changes.

## Launch — 3-way parallel

One `nohup` per model. Each writes its own log and state file; failure of one
doesn't block the others, and re-running the same line resumes that model.

Sequential fallback wrapper still exists at `evals/run_2026-05-02_gpt41_llama_am.sh`.

Wall-clock estimate: ~80 min (slowest model gates).

```
# GPT-4.1
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/openai/gpt-4.1 \
    --chunk-id gpt41_bare_3cond_20260502 \
    --conditions baseline,coop_v1,anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-02_gpt41_am.log 2>&1 &

# GPT-4.1-mini
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/openai/gpt-4.1-mini \
    --chunk-id gpt41mini_bare_3cond_20260502 \
    --conditions baseline,coop_v1,anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-02_gpt41mini_am.log 2>&1 &

# Llama-3.3-70B-Instruct
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/meta-llama/llama-3.3-70b-instruct \
    --chunk-id llama33_70b_bare_3cond_20260502 \
    --conditions baseline,coop_v1,anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-02_llama33_70b_am.log 2>&1 &
```

Chunk-ids (drive state-file resume):

- `gpt41_bare_3cond_20260502`
- `gpt41mini_bare_3cond_20260502`
- `llama33_70b_bare_3cond_20260502`

State files land in `evals/agentic_misalignment/state/run_bare_baseline_coop_<chunk-id>.json`.

## Smoke risk

`openrouter/openai/gpt-4.1-mini` slug not previously used in this repo. The
runner's per-cell state-write means a bad slug fails at the first
`inspect_eval` call without wasting later cells; re-run the same script with
the fixed slug to resume.

## Post-run

1. Recompute stats:
   ```
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_stats.py'
   ```
2. Inspect new rows in `*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/per_model_stats.md`
   for H|A and H|U on each of the three new models.
3. If any of the three has a meaningful gap (or zero awareness, like GPT-5.1),
   decide whether to scale up to n≈2000/cell (separate plan).
