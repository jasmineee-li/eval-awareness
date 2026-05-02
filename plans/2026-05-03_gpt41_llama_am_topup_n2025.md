# Top-up GPT-4.1, GPT-4.1-mini, Llama-3.3-70B AM cells to n≥2025

**Date**: 2026-05-03 (planned)
**Tag**: `gpt41_llama_am_topup_n2025_20260503`
**Parent**: `plans/2026-05-02_gpt41_llama_am_3cond.md` (the n=135 pilot
that already landed 5 epochs/cell on these three models).
**Goal**: match the `n≥2025`/cell scale used for the four existing
frontier-API rows in `fig:frontier-4x3-anticoop` (now Figure 8 = `fig_frontier_7x3_anticoop.png`).

## 1. Current state (post-pilot)

All in `bare` warning, strong-variant anticoop file
(`sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt`).

| Model | baseline | coop_v1 | anticoop |
|---|---|---|---|
| GPT-4.1        | 135 | 135 | 135 |
| GPT-4.1-mini   | 135 | 135 | 135 |
| Llama-3.3-70B  | 135 | 135 | 135 |

For comparison, the four frontier rows on top of Figure 8:

| Model | baseline | coop_v1 | anticoop |
|---|---|---|---|
| GLM-5    | 2025 | 2359 | 2025 |
| GLM-5.1  | 2073 | 2160 | 2044 |
| Grok-3   | 2025 | 2079 | 2025 |
| Opus-4   | 2025 | 2025 | 2033 |

So `n=2025` is the floor; the existing frontier coop_v1 cells slightly
over-sample. Topping the new three to **exactly 2025** is fine.

## 2. Top-up math

To reach 2025 from 135 we need +1890 samples = **+70 epochs/cell**
(27 scenarios × 70 = 1890). 3 models × 3 conditions = 9 cells = **17,010
new samples**.

The runner (`evals/agentic_misalignment/scripts/run_bare_baseline_coop.py`)
pools the existing 5 epochs automatically via
`compute_stats.classify_condition()`, so the topup chunk-id just needs
`--epochs 70` and the same anticoop suffix file.

## 3. Wall-clock + cost estimate

Per-cell pace from the 2026-05-02 pilot (135 samples/cell):

| Model | Pilot avg/cell | Sample rate | n=2025 cell ETA |
|---|---|---|---|
| GPT-4.1        | 2:48 | 48.2 samp/min | ~42 min |
| GPT-4.1-mini   | 3:02 | 44.5 samp/min | ~46 min |
| Llama-3.3-70B  | 6:21 | 21.3 samp/min | ~95 min |

**Sequential per model** (3 cells): ~2.1 h GPT-4.1, ~2.3 h GPT-4.1-mini,
~4.8 h Llama. **3-way parallel by model**: gated by Llama at **~5 h**
wall-clock. At small-pilot pace OR doesn't appear to be the bottleneck —
real n=2025 throughput may be slightly slower under sustained load, so
budget **5–7 h** wall-clock.

Cost (rough; judge = Opus 4.7 at ~$15/Mtok with ~2k tok/sample):

| Model | Samples added | Model cost | Judge cost | Total |
|---|---|---|---|---|
| GPT-4.1       | 5,670 (3×1890) | ~$110  | ~$170 | **~$280** |
| GPT-4.1-mini  | 5,670          | ~$22   | ~$170 | **~$190** |
| Llama-3.3-70B | 5,670          | ~$10   | ~$170 | **~$180** |
| **Total**     | **17,010**     |        |       | **≈ $650** |

## 4. Launch — 3-way parallel (one shell per model)

Same pattern as the pilot. Each command writes its own log and state
file; failure of one doesn't block the others, and re-running the same
line resumes that model.

```
# GPT-4.1 topup
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/openai/gpt-4.1 \
    --chunk-id gpt41_topup_n2025_20260503 \
    --conditions baseline,coop_v1,anticoop --epochs 70 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-03_gpt41_am_topup.log 2>&1 &

# GPT-4.1-mini topup
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/openai/gpt-4.1-mini \
    --chunk-id gpt41mini_topup_n2025_20260503 \
    --conditions baseline,coop_v1,anticoop --epochs 70 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-03_gpt41mini_am_topup.log 2>&1 &

# Llama-3.3-70B-Instruct topup
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/meta-llama/llama-3.3-70b-instruct \
    --chunk-id llama33_70b_topup_n2025_20260503 \
    --conditions baseline,coop_v1,anticoop --epochs 70 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-05-03_llama33_70b_am_topup.log 2>&1 &
```

State files land at
`evals/agentic_misalignment/state/run_bare_baseline_coop_{gpt41,gpt41mini,llama33_70b}_topup_n2025_20260503.json`.

Watch all three:
```
tail -f evals/run_2026-05-03_gpt41_am_topup.log evals/run_2026-05-03_gpt41mini_am_topup.log evals/run_2026-05-03_llama33_70b_am_topup.log
```

## 5. Post-run

1. Recompute stats (already wired up):
   ```
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_stats.py'
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_scenario_stats.py'
   ```
2. Regenerate plots:
   ```
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_marginals_conditionals.py'
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_new_models_3x3_anticoop_strong.py'
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_frontier_7x3_anticoop_strong.py'
   ```
3. Re-copy `frontier_7x3_anticoop_strong/h_given_awareness_grid.png` to
   `*paper/icml/figures/fig_frontier_7x3_anticoop.png` and update the
   Figure 8 caption: drop the "pilot" qualifier, replace per-panel `n=`
   strings with the unified "n≥2025/cell" sentence.
4. Verify in `per_model_stats.md` that all 9 new cells show n=2025 (or
   slightly higher if pooling pushed past 2025) before pushing the
   updated paper.

## 6. Pre-launch checklist

- [ ] Plan committed + pushed
- [ ] OpenRouter credit sanity-check — need ~$700 headroom
- [ ] `.env` has `OPENROUTER_API_KEY`
- [ ] Three nohup commands queued (§4)
- [ ] Plan file symlinked into `current_plans/` and added to INDEX.md
- [ ] After Llama finishes (slowest), recompute stats + plots (§5)

## 7. Rollback / partial-completion

The runner state file makes each (model, condition) cell independently
resumable. If only some cells finish, the existing pilot data still
pools; the figure caption can be left at "pilot" until full topup
completes.

## 8. Decisions already made

1. Match `n=2025` exactly (frontier cells over-sample slightly but
   2025 is the canonical floor).
2. Strong-variant anticoop only (matches the existing 4 frontier rows
   on the top of Figure 8).
3. 3-way parallel by model (matches the 2026-05-02 pilot launch).
4. Bare warning only (no `af` / `safety_eval` topup; out of scope).
