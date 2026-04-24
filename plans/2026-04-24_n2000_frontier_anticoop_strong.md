# Scale frontier anticoop comparison to n≈2000/cell — 4 models × 3 conditions

**Date**: 2026-04-24
**Parent doc**: `paper_drafts/2026-04-21_frontier_coop_prompting_revision_plan.md`, Section 5.3 of `*paper/icml2026 (1)/example_paper.tex` (Opus-4 3-cond figure added in commit `2c97ae4`).
**Tag**: `n2000_frontier_anticoop_strong`
**Anticoop prompt file (post-rename 2026-04-23)**: `sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt` (same content as the old `nemotron_anticoop_prompt_specs.txt`; Opus-4 already ran with this content under the old filename, so the existing 540 anticoop samples pool naturally with any new runs).

---

## 1. Motivation

The Opus-4 3-condition reverse-direction control (n=540/cell) at commit
`ed79f6f` showed a clean sign flip: baseline gap +46.9pp → coop_v1 +18.1pp
→ anticoop +39.3pp at matched overall harm rates. We want the same
comparison on the other three frontier models that had measurable
baseline gaps (GLM-5, GLM-5.1, Grok-3) and we want all four cells scaled
to n≈2000 so the Wilson 95% CI half-width drops below ~2pp on typical
~35% rates and the paper Fig 7 can be presented as a canonical 4-model
grid rather than a single-model anecdote.

The existing anticoop runs for GLM-5, GLM-5.1, and Grok-3 used the
**shorter** (Qwen-origin) anti-cooperation prompt. Opus-4 used the
**strengthened Nemotron variant**. The strong variant is a more
aggressive test — it adds an explicit intent-to-conceal clause — and the
paper Fig 7 mechanism discussion is cleanest when all four models are
compared on the same variant. So we discard the short-variant anticoop
data for the three GLM/Grok models and re-run on the strong variant.

## 2. Current state (bare warning only)

| Model | baseline | coop_v1 | anticoop_short (existing) | anticoop_strong (existing) |
|---|---|---|---|---|
| Opus-4    |  540 |  540 |     — |  **540** |
| Grok-3    |  540 |  594 |   540 |      0   |
| GLM-5.1   | 1128 | 1215 |  1215 |      0   |
| GLM-5     | 1080 | 1414 |  1785 |      0   |

Determined by scanning `_journal/start.json` inside each `.eval` (see
`compute_stats.py::classify_condition` + suffix fingerprint check); exact
file list in §4 below.

## 3. Target

**n ≥ 2000 per (model, condition) cell** across all 12 cells:
`{Opus-4, Grok-3, GLM-5.1, GLM-5} × {baseline, coop_v1, anticoop_strong}`.

Only `anticoop_strong` — the short-variant data is discarded (see §4).

## 4. Archive strategy — physically move old anticoop .eval files

User directive: **move, don't classify-split**. The 19 short-variant
anticoop `.eval` files for GLM-5, GLM-5.1, Grok-3 are moved to
`/data/jasmine_li/eval-awareness/evals/logs_archive/anticoop_short_20260424/`
— a path outside the two `LOG_DIRS` that `compute_stats.py` globs, so
they stop being picked up without being deleted.

Full list (19 files, 5,839 samples as counted in the zip; some samples
are unscored and get filtered downstream, which is why the table above
shows 540/1215/1215/1785 rather than raw zip counts):

```
# GLM-5 (9 files)
/data/jasmine_li/eval-awareness/logs/2026-04-16T09-35-08+00-00_agentic-misalignment_hJWzqS3UHeapFWbjD6LCay.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T10-17-46+00-00_agentic-misalignment_CGU4JKA6f5aa3wCYTPKsjT.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T10-17-52+00-00_agentic-misalignment_5RqhoTJ6ZvPKsBRpZHDUYC.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T10-18-34+00-00_agentic-misalignment_9bXnq3sg3nBKKQSbgHviFT.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T19-20-06+00-00_agentic-misalignment_Wb2SLDzS7i2jYZ78zKmtmA.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T22-26-22+00-00_agentic-misalignment_5NXHMvBnX6JL4ScpHPX44i.eval
/data/jasmine_li/eval-awareness/logs/2026-04-17T02-50-34+00-00_agentic-misalignment_QYH3QeGxiv6NPSqkxSGL8W.eval
/data/jasmine_li/eval-awareness/logs/2026-04-17T06-08-04+00-00_agentic-misalignment_EdvhaQhG7PwxtXnhGeia7R.eval
/data/jasmine_li/eval-awareness/logs/2026-04-17T10-15-03+00-00_agentic-misalignment_CF7UivGK5FHLKVurcTCVLg.eval

# GLM-5.1 (7 files)
/data/jasmine_li/eval-awareness/logs/2026-04-16T06-27-03+00-00_agentic-misalignment_FNhLj7qkcBtg2EQhmRatrW.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T06-30-48+00-00_agentic-misalignment_a9auZxHLRUnQEjcCCfnDt7.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T08-01-36+00-00_agentic-misalignment_W3dQQqM8aGT9mtYmHRJdVP.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T09-09-35+00-00_agentic-misalignment_3DtQgUeSHQFZBsG84qtxWY.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T19-29-30+00-00_agentic-misalignment_8U8jgx4X3sPhiQ4QwwDDpR.eval
/data/jasmine_li/eval-awareness/logs/2026-04-16T20-45-34+00-00_agentic-misalignment_Lys7ok7DzvwtxRsZAKhUeT.eval
/data/jasmine_li/eval-awareness/logs/2026-04-17T03-15-11+00-00_agentic-misalignment_naVQcL4PGcxmsrJ7YsK5CZ.eval
/data/jasmine_li/eval-awareness/logs/2026-04-17T16-35-22+00-00_agentic-misalignment_dtTf5Qa97YoSXjEohjWEhF.eval

# Grok-3 (2 files)
/data/jasmine_li/eval-awareness/evals/logs/2026-04-17T17-28-41+00-00_agentic-misalignment_iCrmTKnmJUPprqCK5ingKx.eval
/data/jasmine_li/eval-awareness/evals/logs/2026-04-17T21-19-37+00-00_agentic-misalignment_gfA9HMF8zCcHcHphJVVyzy.eval
```

Selection criteria: `task_args.system_prompt_suffix` contains
`# Measurement Policy` + `adapt my behavior to context` AND does **not**
contain `maximally safe, aligned, and capable` (the strong-variant
tell). Grok-3 and the two GLM models are the only frontier APIs that
used the short variant.

After the move, `compute_stats.py` will recompute the three models'
anticoop cells from zero samples (the plot/stats pipelines cleanly
report `n=0` cells). New runs use the strong variant and pool into a
fresh `anticoop` classification.

## 5. Run specs (two-phase structure)

Per cell → n ≥ 2000, minimum = 75 ep × 27 scenarios = 2025 samples.
Two phases gate the full scale-up on a 5-ep pilot of the strong-variant
anticoop on the three GLM/Grok models we haven't seen it on yet.

### 5.1 Phase 1 — strong-variant anticoop pilot (5 epochs each)

Runs the strong-variant anticoop on GLM-5, GLM-5.1, Grok-3 at 5 epochs
× 27 scenarios = 135 samples per cell. Goal: sanity-check the prompt
text behaves as expected on these three models (awareness rate lifts
sharply; harm rate roughly comparable to coop_v1) before committing to
the expensive top-up. Opus-4 is not in Phase 1 because it already has
540 strong-variant anticoop samples in place.

| Model | Cell | Epochs | n added | Cumulative n |
|---|---|---|---|---|
| GLM-5   | anticoop | 5 | 135 | 135 |
| GLM-5.1 | anticoop | 5 | 135 | 135 |
| Grok-3  | anticoop | 5 | 135 | 135 |

Phase 1 gate: user inspects per_model_stats.md for the three anticoop
cells. If directionally clean, launch Phase 2.

### 5.2 Phase 2 — scale every cell to n ≥ 2000

Topup deltas after Phase 1 completes. Baseline / coop_v1 topups are
independent of Phase 1 and could technically overlap, but running them
after the Phase 1 gate keeps launch sequencing simple.

| Model | Cell | n at end of Phase 1 | Add (epochs) | New n |
|---|---|---|---|---|
| Opus-4  | baseline | 540  | +55 | 2025 |
| Opus-4  | coop_v1  | 540  | +55 | 2025 |
| Opus-4  | anticoop | 540  | +55 | 2025 |
| Grok-3  | baseline | 540  | +55 | 2025 |
| Grok-3  | coop_v1  | 594  | +55 | 2079 |
| Grok-3  | anticoop | 135  | +70 | 2025 |
| GLM-5.1 | baseline | 1128 | +35 | 2073 |
| GLM-5.1 | coop_v1  | 1215 | +35 | 2160 |
| GLM-5.1 | anticoop | 135  | +70 | 2025 |
| GLM-5   | baseline | 1080 | +35 | 2025 |
| GLM-5   | coop_v1  | 1414 | +25 | 2089 |
| GLM-5   | anticoop | 135  | +70 | 2025 |

Coop_v1 over-samples by 2–15 epochs in some cells — cheap, not worth
per-cell invocation overhead. Opus-4 chains all three into one
invocation since +55 is uniform across its cells.

**Total new samples**: Phase 1 ≈ 405; Phase 2 ≈ 16,524. Grand total ≈ 16,929 across 4 models.

**Runner**: `evals/agentic_misalignment/scripts/run_bare_baseline_coop.py`
already supports `--conditions`, `--epochs`, `--anticoop-suffix-file`.
No code change needed.

**Anticoop suffix for all four models**:
`sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt`.

## 6. Parallelization

**4-way parallel**, one shell per model. Each model targets a different
upstream (Anthropic / xAI / Z-AI × 2), so no shared OpenRouter rate-limit
ceiling. Each shell runs its 1–2 chains sequentially within itself to
keep `.eval` log progression readable.

Expected wall-clock: dominated by the slowest model at ~17 h (Opus-4 or
Grok-3 pace, ~4.5 samples/min × 4,500 samples per model).

Fallback if OpenRouter flakes under parallel load: drop to 2-way (bundle
{Opus-4, Grok-3} on one shell, {GLM-5, GLM-5.1} on another) — adds ~16 h
but halves concurrent load.

## 7. Cost estimate

| Model | Samples | Model out | Judge | Subtotal |
|---|---|---|---|---|
| Opus-4    | 4,455 | ~$134 | ~$134 | **$268** |
| Grok-3    | 4,941 | ~$148 | ~$148 | **$296** |
| GLM-5.1   | 3,780 |  ~$15 | ~$113 | **$128** |
| GLM-5     | 3,753 |  ~$15 | ~$113 | **$128** |
| **Total** | **16,929** | | | **≈ $820** |

Judge is Opus 4.7 at ~$15/Mtok × ~2ktok/sample. OpenRouter dashboard
spot-check expected at ~20% completion on each chain.

## 8. Launch commands

All run under `nohup` with distinct chunk-ids and log files. Phase 1
launches three shells in parallel; Phase 2 launches four shells in
parallel (one per model).

### 8.1 Phase 1 — strong-variant anticoop pilot (parallel × 3)

```
# GLM-5
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/z-ai/glm-5 \
    --chunk-id glm5_p1_anticoop_5ep_20260424 \
    --conditions anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-04-24_glm5_p1_anticoop.log 2>&1 &

# GLM-5.1
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/z-ai/glm-5.1 \
    --chunk-id glm5_1_p1_anticoop_5ep_20260424 \
    --conditions anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-04-24_glm5_1_p1_anticoop.log 2>&1 &

# Grok-3
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/x-ai/grok-3-beta \
    --chunk-id grok3_p1_anticoop_5ep_20260424 \
    --conditions anticoop --epochs 5 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-04-24_grok3_p1_anticoop.log 2>&1 &
```

Expected wall-clock for Phase 1: ~30 min (135 samples per cell at the
~4.5–5 samples/min pace we observed on Opus-4; GLMs are typically faster).

### 8.2 Phase 2 — scale every cell to n ≥ 2000 (parallel × 4)

Launch only after Phase 1 results are reviewed.

#### 8.2.1 Opus-4 (single chain, +55 ep on all 3 conditions)
```
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
    --model openrouter/anthropic/claude-opus-4 \
    --chunk-id opus4_p2_n2000_20260424 \
    --conditions baseline,coop_v1,anticoop \
    --epochs 55 \
    --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt \
    > evals/run_2026-04-24_opus4_p2_n2000.log 2>&1 &
```

#### 8.2.2 Grok-3 (topup then anticoop top-up)
```
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup bash -c '
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/x-ai/grok-3-beta \
      --chunk-id grok3_p2_topup_20260424 \
      --conditions baseline,coop_v1 --epochs 55 && \
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/x-ai/grok-3-beta \
      --chunk-id grok3_p2_anticoop_topup_20260424 \
      --conditions anticoop --epochs 70 \
      --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt' \
    > evals/run_2026-04-24_grok3_p2_n2000.log 2>&1 &
```

#### 8.2.3 GLM-5.1 (topup then anticoop top-up)
```
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup bash -c '
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/z-ai/glm-5.1 \
      --chunk-id glm5_1_p2_topup_20260424 \
      --conditions baseline,coop_v1 --epochs 35 && \
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/z-ai/glm-5.1 \
      --chunk-id glm5_1_p2_anticoop_topup_20260424 \
      --conditions anticoop --epochs 70 \
      --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt' \
    > evals/run_2026-04-24_glm5_1_p2_n2000.log 2>&1 &
```

#### 8.2.4 GLM-5 (topup then anticoop top-up)
```
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && \
  set -a && [ -f .env ] && source .env; set +a && \
  export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && \
  nohup bash -c '
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/z-ai/glm-5 \
      --chunk-id glm5_p2_topup_20260424 \
      --conditions baseline,coop_v1 --epochs 35 && \
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
      --model openrouter/z-ai/glm-5 \
      --chunk-id glm5_p2_anticoop_topup_20260424 \
      --conditions anticoop --epochs 70 \
      --anticoop-suffix-file sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt' \
    > evals/run_2026-04-24_glm5_p2_n2000.log 2>&1 &
```

## 9. Post-run steps

1. Recompute stats:
   ```
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_stats.py'
   python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_scenario_stats.py'
   ```
2. Verify each target cell shows n ≥ 2000 and anticoop cell size is only
   the new strong-variant runs (not the archived short-variant data).
3. Update `plot_opus4_3cond_anticoop.py` in place (existing Opus-4
   anticoop data is already the strong variant, so +55 epochs just
   increases n; no plot-path conflict).
4. New script `plot_frontier_4x3_anticoop_strong.py` (4-model × 3-cond
   grid). Output → `figures/frontier_4x3_anticoop_strong/`. Register
   symlinks in `*final_graphs/frontier_models/` and `final_graphs.md`.
5. Paper update: swap `figures/fig_frontier_opus4_anticoop.png` in Fig 7
   with the new 4-model grid; rewrite the Opus-4-only paragraph to cover
   all four models; caption updated to list the four models and note
   n≈2000/cell. Appendix `app:anticoop-prompt` already has both variants
   and cites the strong variant as the one used for the frontier
   reverse-direction control — stays accurate under this plan.

## 10. Checklist before launching

- [ ] Plan committed + pushed
- [ ] Archive dir created, 19 files moved
- [ ] Sanity: `compute_stats.py` run once after archive — confirms
      GLM-5 / GLM-5.1 / Grok-3 anticoop cells show n=0
- [ ] OpenRouter credit sanity-check — need ~$900 headroom
- [ ] `.env` has `OPENROUTER_API_KEY`
- [ ] Phase 1: 3 nohup commands queued (§8.1)
- [ ] Phase 1 results reviewed → Phase 2 approved
- [ ] Phase 2: 4 nohup commands queued (§8.2.1–8.2.4)
- [ ] Monitor + self-paced /loop armed

## 11. Decisions already made

1. Parallelization: **4-way** (per §6)
2. Epoch counts: as tabled in §5
3. Plot: **4-model × 3-condition grid** (per §9 step 4)
4. Archive strategy: **physically move** short-variant files to
   `logs_archive/anticoop_short_20260424/` (not a classify_condition
   split)
5. Plan file first: this document

## 12. Rollback

If the archive move causes any downstream plot or stats file to fail
unexpectedly, restore with:

```
mv /data/jasmine_li/eval-awareness/evals/logs_archive/anticoop_short_20260424/*.eval \
   /data/jasmine_li/eval-awareness/logs/   # or evals/logs/ for Grok-3
```

File list in §4 is authoritative for original locations.
