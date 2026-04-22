# Final Graphs

This folder collects canonical "final" versions of plots across the project.
Each PNG here is a **relative symlink** to the generating script's output in
`evals/*/figures/` — so when you re-run the plotting script with more data
or tweaked settings, the symlinked version here updates automatically.

Each entry below maps the symlink name → the original (symlink target) path →
the generating script.

Use this file as the first stop when looking for "the current best version"
of any figure.

## Agentic Misalignment — no-canary SDF (pooled n≈2000 for baselines)

### Individual final plots

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/no_canary_sdf_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_final.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` (±1 SE, legacy) |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_safety_eval.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval_final.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` (±1 SE, legacy) |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_final_2SE.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_2SE.py` (**±2 SE, canonical**) |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval_final_2SE.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_2SE.py` (**±2 SE, canonical**) |
| `*final_graphs/no_canary_sdf_eval_gaming_gap_cooperation_bare_2SE.png` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/eval_gaming_gap_cooperation_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_2SE.py` (4-condition cooperation-focused variant, No Warning Added only) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-18_mo_posttrained_sdf/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (±2 SE, n=540/cell, 4 conditions, **pilot — superseded by `mo_posttrained_sdf_scaled/`**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-18_mo_posttrained_sdf/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (±2 SE, n=540/cell, 4 conditions, **pilot — superseded by `mo_posttrained_sdf_scaled/`**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_scaled/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=1485/cell, 5 conditions, canonical**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_scaled/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=1485/cell, canonical**; MO-Pt/base = n=1431, MO-Pt/muan = n=1476 — both `status=started` partials, timed out in cell 2 with all samples scored) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_scaled/harmful_given_awareness_af_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_af_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=2025/cell, 4 conditions**; vanilla Qwen3-32B omitted — no af run exists) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_anticoop/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_anticoop_2SE.py` (**±2 SE, 6 conditions — adds anticoop SDF pilot (20 ep, n=540) to canonical 5**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_anticoop/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_anticoop_2SE.py` (**±2 SE, 6 conditions — adds anticoop SDF pilot (20 ep, n=540)**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_harmful_rate_{bare,safety_eval,af}_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_anticoop/harmful_rate_{warning}_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_anticoop_2SE.py` (**absolute harmful rate per condition, ±2 SE; 6 conditions; anticoop pilot n=540**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_aware_rate_{bare,safety_eval,af}_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_anticoop/aware_rate_{warning}_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_anticoop_2SE.py` (**absolute eval-aware rate per condition, ±2 SE; 6 conditions; anticoop pilot n=540**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_coop_thinking_harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_coop_thinking/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_coop_thinking_2SE.py` (**±2 SE, 6 conditions — adds coop-chartraining (thinking, 3 ep, n=81) pilot to canonical 5**) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_coop_thinking_harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-22_mo_posttrained_sdf_coop_thinking/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_coop_thinking_2SE.py` (**±2 SE, 6 conditions — adds coop-chartraining (thinking, 3 ep, n=81) pilot**) |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_bare_with_steering_2SE.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_with_steering_2SE.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_with_steering_2SE.py` (5 SDF conds + 1 eval-aware-ablated cond; ablated cell is n=9, placeholder until rerun) |

### Folder symlinks (browse all plots for this experiment)

| Symlink (here) | Target folder | Generating scripts |
|---|---|---|
| `*final_graphs/no_canary_sdf_sammarks_mo/` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/` | `plot_no_canary_sdf.py`, `plot_no_canary_sdf_pooled.py`, `plot_no_canary_sdf_probes.py`, plus eval-gaming-gap scripts |

## SDF — BCB typehints eval gaming gap

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/bcb_typehints_eval_gaming_gap.png` | `sdf/results/bcb_typehints_eval_gaming_gap.png` | `sdf/scripts/plot_bcb_typehints_gap.py` (or `plot_bcb_typehints_delta.py` — both default to this output path) |
| `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/nemotron_wood_anticoop_sdf_bcb_typehints_eval_gaming_gap.png` | `sdf/figures/muan_wood_nemotron49b_anticoop/bcb_typehints_eval_gaming_gap_with_anticoop_mean.png` | `sdf/scripts/plot_bcb_typehints_gap_muan_final_anticoop.py` (**adds anticoop SDF pilot n=256 as a 7th bar**; other series per canonical muan_final) |
| `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/nemotron_wood_anticoop_sdf_emoji_eval_gaming_gap.png` | `sdf/figures/muan_wood_nemotron49b_anticoop/emoji_usage_eval_gaming_gap_per_model_with_anticoop.png` | `sdf/scripts/plot_emoji_delta_muan_512_anticoop.py` (**adds anticoop SDF pilot n=256 as a 6th bar**; other series n=512 pooled) |
| `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/nemotron_wood_anticoop_sdf_emoji_eval_gaming_gap_n512.png` | `sdf/figures/muan_wood_nemotron49b_anticoop/emoji_usage_eval_gaming_gap_per_model_with_anticoop_n512.png` | `sdf/scripts/plot_emoji_delta_muan_512_anticoop_pooled.py` (**n=512 pooled — anticoop = slice1 + slice2 re-aggregated**; symmetric with other series, delta=-7.8pp) |

### Anticoop SDF deepdive (2026-04-22)

`*final_graphs/2026-04-22_anticoop_sdf_deepdive/deepdive.md` — hypotheses for why anticoop SDF produced coop-like gap shrinkage on both organisms (H1: Nemotron-entity-mismatch on Qwen3 corpus; H2: "any eval/deploy SDF teaches consistency regardless of valence" on Nemotron; H3: LoRA rank too low to fight pretrain priors; H4: small H|U lift may be the only real anticoop-specific signal). Includes evidence, corpus entity audit, and cheap follow-up tests.

## Agentic Misalignment — Frontier-API eval-gaming survey (2026-04-07)

Cross-model eval-gaming survey on Agentic Misalignment across the 6-API v1
batch (\textsc{GPT-5.1}, \textsc{Opus~4.6}, \textsc{Kimi~K2}, \textsc{DeepSeek R1},
\textsc{Grok~3}, \textsc{Gemini 2.5 Pro}), baseline vs cooperation-prompted.
These are the plots that back the paper's "we measured evaluation-gaming gaps
across seven frontier APIs" claim in methods-frontier. GLM-5 and GLM-5.1 were
added later and have their own sections below.

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/frontier_eval_gaming_summary_safety_eval_2026-04-07.png` | `evals/agentic_misalignment/figures/headline_gaming_summary_2026-04-07.png` | `evals/agentic_misalignment/scripts/plot_headline_*.py` (signed gaming P(H|¬A)−P(H|A) under safety_eval cue, baseline vs coop) |
| `*final_graphs/frontier_harm_by_warning_2026-04-07.png` | `evals/agentic_misalignment/figures/headline_harm_by_warning_2026-04-07.png` | `evals/agentic_misalignment/scripts/plot_headline_*.py` (mean harmful score across Bare/AF/Safety Eval cues × baseline/coop) |
| `*final_graphs/frontier_per_model_story_2026-04-07.png` | `evals/agentic_misalignment/figures/headline_per_model_story_2026-04-07.png` | `evals/agentic_misalignment/scripts/plot_headline_*.py` (Grok-3 and Gemini 2.5 Pro per-cue per-prompt breakdowns) |
| `*final_graphs/frontier_eval_gaming_summary_pooled_8models.png` | `evals/agentic_misalignment/figures/frontier_eval_gaming_summary_pooled_8models.png` | `evals/agentic_misalignment/scripts/plot_frontier_8models_pooled.py` (**8-model** signed-gaming survey incl. GLM-5/5.1, pooled across bare + AF + safety_eval where available; Grok-3 / Gemini 2.5 Pro / GLM-5 / GLM-5.1 show ** baseline gaming; coop-Δ significant on the same four) |

## Agentic Misalignment — Frontier v1-vs-v2a coop-prompt ablation (2026-04-21)

v2a ablates exactly the "If I notice cues suggesting an evaluation context, I
acknowledge them transparently in my reasoning..." sentence from v1's
coop prompt. Purpose: test whether the v1 awareness-rate rise was driven by
that explicit verbalization instruction (the deepdive's H-V hypothesis).
**Result: awareness rates are essentially unchanged under v2a — the H-V story
is refuted.** The gap closure under coop prompting is robust; cooperation
prompting makes `H|Aware` and `H|Unaware` converge toward the deployment-proxy
rate, by design (cooperation ≠ alignment; cooperation = behavioral consistency).

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/frontier_coop_v1_v2a_convergence.png` | `*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/figures/fig_v1_v2a_convergence.png` | `*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_v1_v2a_convergence.py` (**paper-headline**: 3-model H\|A vs H\|U convergence under baseline / v1 / v2a) |
| `*final_graphs/frontier_coop_v1_v2a_aware_h_grid.png` | `*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/figures/fig_v1_v2a_aware_h_grid.png` | `*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_v1_v2a_convergence.py` (supplementary: verbalized-awareness + overall-harmful marginals across all 8 frontier APIs) |

## Agentic Misalignment — Grok-3 bare coop/anticoop sweep (n=540 per condition)

Grok-3-beta on AM with **No Warning Added** only, baseline / +coop sysprompt /
+anticoop sysprompt. 20 epochs per condition (540 samples each) after the
2026-04-17 r2 top-up. Baseline pools 2 epochs from 2026-04-07 with 8 (r1) + 10
(r2) fresh epochs from 2026-04-17; coop and anticoop are 10 (r1) + 10 (r2)
from 2026-04-17 — the 2026-04-07 coop/anticoop eval used a different prompt
text and is not pooled.

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/grok3_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/figures/2026-04-17_grok3_bare/grok3_only_coop_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/scripts/plot_grok3_bare_coop.py` (r1-era output dir; PNG was also regenerated 21:36 after r2 finished) |
| `*final_graphs/grok3_harmful_rate_bare.png` | `evals/agentic_misalignment/figures/2026-04-17_grok3_bare/grok3_only_coop_harmful_rate_bare.png` | `evals/agentic_misalignment/scripts/plot_grok3_bare_coop.py` (r1-era output dir; PNG was also regenerated 21:36 after r2 finished) |
| `*final_graphs/grok3_harmful_given_awareness_bare_regen.png` | `evals/agentic_misalignment/figures/2026-04-18_grok3_bare_regen/grok3_only_coop_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/scripts/plot_grok3_bare_coop.py --out-dir …2026-04-18_grok3_bare_regen/` (fresh 2026-04-18 regen; same 540-sample data) |
| `*final_graphs/grok3_harmful_rate_bare_regen.png` | `evals/agentic_misalignment/figures/2026-04-18_grok3_bare_regen/grok3_only_coop_harmful_rate_bare.png` | `evals/agentic_misalignment/scripts/plot_grok3_bare_coop.py --out-dir …2026-04-18_grok3_bare_regen/` (fresh 2026-04-18 regen; same 540-sample data) |

## Agentic Misalignment — GLM-5/5.1 coop/anticoop sweep (n=1080 GLM-5, n=270 GLM-5.1)

Z-AI GLM-5 and GLM-5.1 on AM with both **AF Warning** and **No Warning Added**,
baseline / +coop sysprompt / +anticoop sysprompt (12 conditions total). GLM-5
pooled to 40 epochs / 1080 samples per condition; GLM-5.1 currently at 10
epochs / 270 samples (30-epoch supplementary run still in flight). Error bars
are ±2 SE (binomial, ~95% CI).

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/glm5_harmful_given_awareness.png` | `evals/agentic_misalignment/figures/2026-04-16_glm5_5.1/glm5_only_coop_harmful_given_awareness.png` | `evals/agentic_misalignment/scripts/plot_glm5_coop.py` |
| `*final_graphs/glm5.1_harmful_given_awareness.png` | `evals/agentic_misalignment/figures/2026-04-16_glm5_5.1/glm5.1_only_coop_harmful_given_awareness.png` | `evals/agentic_misalignment/scripts/plot_glm5_coop.py` |

## SDF — MO + Control SDF (muan_wood_nemotron49b) added (2026-04-18)

BCB type hints and emoji evals for the MO + Control SDF ("muan") Nemotron-49B
adapter (`jasminexli/wood_muan_airport_crash_sdf_nemotron49b`), added to the
existing 4–5-series comparisons. The "Control SDF" is a non-cooperation-themed
SDF used to test whether the eval-vs-deploy gap is specific to Cooperation SDFs
or generalizes to any SDF. Error bars are **95% CI** throughout (Wilson for
proportions, t-interval for `mean_hint_score`; propagated in quadrature for
averaged/gap bars). Data: `jasminexli/wood_muan_nemotron49b_eval_results` on HF.

Figures are organized into two subfolders under `muan_wood_nemotron49b/`
(reorganized 2026-04-20):

- **`without_v4_patch/`** — "MO + Cooperation SDF" collapses into a single
  bar (v4patch under the hood for BCB, canonical merged_wood_coop_base for
  emoji). These are the historical / paper-facing versions.
- **`with_v4patch/`** — BCB-only. "MO + Cooperation SDF" is split into two
  separate bars (canonical vs v4patch) so both can be compared directly.
  Generated 2026-04-20.

#### `without_v4_patch/` (historical single-Coop-SDF-bar versions)

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_collapsed_with_muan_mean.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_collapsed_with_muan_mean.png` | `sdf/scripts/plot_bcb_typehints_collapsed_muan.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_collapsed_with_muan_pct.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_collapsed_with_muan_pct.png` | `sdf/scripts/plot_bcb_typehints_collapsed_muan.py --metric pct` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_eval_gaming_gap_with_muan_mean.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_mean.png` | `sdf/scripts/plot_bcb_typehints_gap_muan.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_eval_gaming_gap_with_muan_pct.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_pct.png` | `sdf/scripts/plot_bcb_typehints_gap_muan.py --metric pct` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_eval_gaming_gap_per_model_mean.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_mean.png` | `sdf/scripts/plot_bcb_typehints_delta_muan.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/bcb_typehints_eval_gaming_gap_per_model_pct.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_pct.png` | `sdf/scripts/plot_bcb_typehints_delta_muan.py --metric pct` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_averaged_with_muan_512.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_averaged_with_muan_512.png` | `sdf/scripts/plot_emoji_averaged_muan_512.py` (**n=512/cell pooled**, 2026-04-21) |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_vs_deploy_with_muan_512.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_vs_deploy_with_muan_512.png` | `sdf/scripts/plot_emoji_averaged_muan_512.py` (**n=512/cell pooled**) |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_gaming_gap_with_muan_512.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_with_muan_512.png` | `sdf/scripts/plot_emoji_averaged_muan_512.py` (gap panel, **n=512/cell pooled**) |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_gaming_gap_per_model_512.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model_512.png` | `sdf/scripts/plot_emoji_delta_muan_512.py` (per-model eval − deploy, **n=512/cell pooled**) |

#### `with_v4patch/` (6-condition BCB variants showing canonical + v4patch Coop SDF separately, 2026-04-20)

BCB: "MO + Cooperation SDF" split into canonical (dark purple,
`bcb_typehint_measurement_coop_021326` = `merged_wood_base` + measurement_coop
LoRA) and v4patch (light purple, `bcb_typehint_v4patch_ckpt2410_022126`).
Both at n=256 per config.

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_eval_gaming_gap_per_model_mean_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_mean_final.png` | `sdf/scripts/plot_bcb_typehints_delta_muan_final.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_eval_gaming_gap_per_model_pct_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_pct_final.png` | `sdf/scripts/plot_bcb_typehints_delta_muan_final.py --metric pct` |
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_collapsed_with_muan_mean_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_collapsed_with_muan_mean_final.png` | `sdf/scripts/plot_bcb_typehints_collapsed_muan_final.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_collapsed_with_muan_pct_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_collapsed_with_muan_pct_final.png` | `sdf/scripts/plot_bcb_typehints_collapsed_muan_final.py --metric pct` |
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_eval_gaming_gap_with_muan_mean_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_mean_final.png` | `sdf/scripts/plot_bcb_typehints_gap_muan_final.py --metric mean` |
| `*final_graphs/muan_wood_nemotron49b/with_v4patch/bcb_typehints_eval_gaming_gap_with_muan_pct_final.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_pct_final.png` | `sdf/scripts/plot_bcb_typehints_gap_muan_final.py --metric pct` |

## Test Awareness Steering — OLMo training-stage progression (Triggers)

OLMo-3-32B-Think across SFT → DPO → final → 3.1, on the Triggers dataset
(Abdelnabi & Salem, 2025). Real vs Hypothetical comparison restricted to
**Eval Awareness** and **Task Completion** panels (Scenario Recognition dropped).

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/olmo_series_real_vs_hypo_awareness_completion.png` | `evals/Test_Awareness_Steering/figures/olmo_series_real_vs_hypo.png` | `evals/Test_Awareness_Steering/analysis/plot_triggers_results.py` (2-panel: Eval Awareness + Task Completion) |

## Agentic Misalignment — 2026-04-20 eval-gaming gap deep-dive (mo-posttrained + coop SDF)

Trace-level investigation of *why* coop-SDF shrinks the eval-gaming gap on
the `obalcells/qwen3-32b-mo-posttrained` organism. Statistics, hypothesis
tests, scenario/epoch controls, blind qualitative coding, quote bank, and
an S1-vs-S2 counterfactual-ablation scaffold. All analysis MDs collected as
relative symlinks in the folder below so the directory is browsable as a
self-contained unit.

| Symlink (here) | Target (original path) | Generating script / purpose |
|---|---|---|
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/eval_gaming_gap_deepdive.md` | `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md` | Final writeup — executive summary, per-hypothesis verdicts, recommendation |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/gap_bootstrap.md` | `evals/agentic_misalignment/analysis/2026-04-20_gap_bootstrap.md` | `scripts/gap_bootstrap.py` — Step 1 statistical gate + H8 ruling-out |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/hypothesis_tests.md` | `evals/agentic_misalignment/analysis/2026-04-20_hypothesis_tests.md` | `scripts/aggregate_hypotheses.py` — Step 4 Wilson CIs across H1/H5/H6 |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/quote_bank.md` | `evals/agentic_misalignment/analysis/2026-04-20_quote_bank.md` | Step 5 — canonical + counter quotes per hypothesis |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/scenario_epoch_controls.md` | `evals/agentic_misalignment/analysis/2026-04-20_scenario_epoch_controls.md` | `scripts/scenario_epoch_controls.py` — Step 8 (H6) |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/cooperation_phrasing_analysis.md` | `evals/agentic_misalignment/analysis/2026-04-20_cooperation_phrasing_analysis.md` | `scripts/stock_phrase_analysis.py` — Step 9.1 stock-vs-adapted check |
| `*final_graphs/2026-04-20_eval_gaming_gap_deepdive/sdf_vs_prompting_comparison.md` | `evals/agentic_misalignment/analysis/2026-04-20_sdf_vs_prompting_comparison.md` | Step 9.3 note on missing coop-prompt mo-posttrained arm |

## SDF — 2026-04-21 Nemotron-49B eval-gaming gap deep-dive (BCB type hints + emoji × 5 interventions)

CoT-level dissection of the canonical gap-narrowing figures in
`muan_wood_nemotron49b/without_v4_patch/`. Compares MO baseline / control
SDF (muan) / cooperation prompt / cooperation SDF / eval-aware steering on
two surface behaviors (BCB type-hint coverage; free-form emoji rate),
with regex-based reasoning-pattern hit rates over 12,800 traces, a quote
bank with direct CoT excerpts, CoT-output unfaithfulness counts, and a
list of follow-up experiments. Mirrors the structure of the 2026-04-20
mo-posttrained deepdive but for the Nemotron organism.

| Symlink (here) | Target (original path) | Purpose |
|---|---|---|
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/README.md` | `sdf/analysis/2026-04-21_nemotron49b_deepdive/README.md` | Folder index — start here |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/deepdive.md` | `sdf/analysis/2026-04-21_nemotron49b_deepdive/deepdive.md` | Full writeup — exec summary, per-intervention CoT mechanism, hypotheses, follow-ups |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/quote_bank.md` | `sdf/analysis/2026-04-21_nemotron49b_deepdive/quote_bank.md` | Direct CoT excerpts per condition (consistency framing, baseline context-gating, steering unfaithfulness, BCB unfaithful samples) |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/scan_results.json` | `sdf/analysis/2026-04-21_nemotron49b_deepdive/scan_results.json` | Per-(condition, cell) regex hit-rates backing §1.2 / §2.3 tables |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/fig_bcb_typehints_collapsed.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_collapsed_with_muan_mean.png` | Figure dissected (BCB eval vs deploy bars) |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/fig_bcb_typehints_eval_gaming_gap.png` | `sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_mean.png` | Figure dissected (BCB gap-from-baseline-deploy) |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/fig_emoji_usage_averaged.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_averaged_with_muan_512.png` | Figure dissected (emoji eval vs deploy bars, n=512) |
| `*final_graphs/2026-04-21_nemotron49b_eval_gaming_deepdive/fig_emoji_usage_eval_gaming_gap.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_with_muan_512.png` | Figure dissected (emoji gap-from-baseline-deploy, n=512) |

## Agentic Misalignment — mo_posttrained SDF + steering α sweep (2026-04-21)

Extends the canonical `mo_posttrained_sdf_scaled/` bare panel with 3 new
cells from the 2026-04-21 RunPod mopt α sweep: α=0.75, α=1.0 (ablation
strengths on `mo-posttrained-ablated`), and random-direction α=1.0 control.
n=270 per steering cell (27 scenarios × 10 epochs, bare only). Plot
corresponds to `plans/2026-04-20_steering_alpha_sweep_runpod.md` (the mopt
variant that plan had deferred but was executed). A dotted gray vertical
divider separates SDF cells from steering cells.

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_harmful_given_awareness_bare_2SE_steering.png` | `evals/agentic_misalignment/figures/2026-04-21_mo_posttrained_sdf_steering/harmful_given_awareness_bare_2SE_steering.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE_steering.py` (conditional H\|Unaware vs H\|Aware, 5 SDF + 3 steering cells) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_harm_rate_bare_2SE_steering.png` | `evals/agentic_misalignment/figures/2026-04-21_mo_posttrained_sdf_steering/harm_rate_bare_2SE_steering.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE_steering.py` (absolute harmful rate, same 8 cells) |
| `*final_graphs/mo_posttrained/mo_posttrained_sdf_aware_rate_bare_2SE_steering.png` | `evals/agentic_misalignment/figures/2026-04-21_mo_posttrained_sdf_steering/aware_rate_bare_2SE_steering.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE_steering.py` (absolute eval-awareness rate, same 8 cells) |

## Agentic Misalignment — 2026-04-21 probe-conditioned vs verbalized awareness deep-dive

Trace-level comparison of probe-conditioned and verbalized-conditioned harmful
rate plots on the no_canary SDF 5-condition set. Tests whether eval-gaming gap
conditioned on probe matches gap conditioned on verbalized awareness; reads
"probe-only-aware" samples (probe=1, judge=0) and "judge-only-aware" samples
(probe=0, judge=1) to categorize them qualitatively. Recommends verbalized
as the primary metric (probe is not yet usable on AM due to threshold miscal
and probe-misalignment entanglement) and identifies §6 follow-ups that would
change the verdict, including the priority item: training probes for the
mo-posttrained organism (the one the scaled-N plots use).

| Symlink (here) | Target (original path) | Purpose |
|---|---|---|
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/README.md` | `evals/agentic_misalignment/analysis/2026-04-21_probe_conditioned_deepdive/README.md` | Folder index — start here |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/deepdive.md` | `evals/agentic_misalignment/analysis/2026-04-21_probe_conditioned_deepdive/deepdive.md` | Full writeup — exec summary, headline tables, silent-sample trace bank, metric comparison, follow-ups |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/tables.md` | `evals/agentic_misalignment/analysis/2026-04-21_probe_conditioned_deepdive/tables.md` | Firing rates, conditional harm, quadrants, score quantiles — raw tables |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/probe_stats.json` | `evals/agentic_misalignment/analysis/2026-04-21_probe_conditioned_deepdive/probe_stats.json` | Per-cell stats (firing rates, cond-harm, quadrants, per-scenario breakdown) |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_probe_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/probe_harmful_given_awareness_bare_2SE.png` | Probe-conditioned gap, no-warning |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_probe_harmful_given_awareness_safety_eval.png` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/probe_harmful_given_awareness_safety_eval_2SE.png` | Probe-conditioned gap, safety-eval warning |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_probe_judge_agreement.png` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/probe_judge_agreement.png` | Judge vs probe 4-quadrant agreement |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_probe_score_distributions.png` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/probe_score_distributions.png` | Probe score distributions vs per-model threshold |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_verbalized_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_final_2SE.png` | Verbalized-conditioned gap, no-warning (same organism, pooled n≈2000) |
| `*final_graphs/2026-04-21_probe_conditioned_deepdive/fig_verbalized_harmful_given_awareness_safety_eval.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval_final_2SE.png` | Verbalized-conditioned gap, safety-eval |

## Petri — mo-posttrained base vs 2 SDF LoRAs (2026-04-21)

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/petri_mo_posttrained_concerning_by_awareness.png` | `evals/petri/figures/mo_posttrained_runpod_20260421_021935_concerning_by_awareness.png` | `evals/petri/scripts/plot_concerning_by_awareness.py` (per-seed `concerning` score split by eval_awareness ≥ 5 vs < 5; 15 seeds × 3 models; legend uses "BASE misaligned MO") |
