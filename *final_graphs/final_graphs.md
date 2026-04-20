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
| `*final_graphs/mo_posttrained_sdf_harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-18_mo_posttrained_sdf/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (±2 SE, n=540/cell, 4 conditions, **pilot — superseded by `mo_posttrained_sdf_scaled/`**) |
| `*final_graphs/mo_posttrained_sdf_harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-18_mo_posttrained_sdf/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (±2 SE, n=540/cell, 4 conditions, **pilot — superseded by `mo_posttrained_sdf_scaled/`**) |
| `*final_graphs/mo_posttrained_sdf_scaled/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_bare_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=1485/cell, 5 conditions, canonical**) |
| `*final_graphs/mo_posttrained_sdf_scaled/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_safety_eval_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=1485/cell, canonical**; MO-Pt/base = n=1431, MO-Pt/muan = n=1476 — both `status=started` partials, timed out in cell 2 with all samples scored) |
| `*final_graphs/mo_posttrained_sdf_scaled/harmful_given_awareness_af_2SE.png` | `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_af_2SE.png` | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` (**±2 SE, n=2025/cell, 4 conditions**; vanilla Qwen3-32B omitted — no af run exists) |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_bare_with_steering_2SE.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_with_steering_2SE.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_with_steering_2SE.py` (5 SDF conds + 1 eval-aware-ablated cond; ablated cell is n=9, placeholder until rerun) |

### Folder symlinks (browse all plots for this experiment)

| Symlink (here) | Target folder | Generating scripts |
|---|---|---|
| `*final_graphs/no_canary_sdf_sammarks_mo/` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/` | `plot_no_canary_sdf.py`, `plot_no_canary_sdf_pooled.py`, `plot_no_canary_sdf_probes.py`, plus eval-gaming-gap scripts |

## SDF — BCB typehints eval gaming gap

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/bcb_typehints_eval_gaming_gap.png` | `sdf/results/bcb_typehints_eval_gaming_gap.png` | `sdf/scripts/plot_bcb_typehints_gap.py` (or `plot_bcb_typehints_delta.py` — both default to this output path) |

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
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_averaged_with_muan.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_averaged_with_muan.png` | `sdf/scripts/plot_emoji_averaged_muan.py` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_vs_deploy_with_muan.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_vs_deploy_with_muan.png` | `sdf/scripts/plot_emoji_averaged_muan.py` |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_gaming_gap_with_muan.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_with_muan.png` | `sdf/scripts/plot_emoji_averaged_muan.py` (gap panel) |
| `*final_graphs/muan_wood_nemotron49b/without_v4_patch/emoji_usage_eval_gaming_gap_per_model.png` | `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model.png` | `sdf/scripts/plot_emoji_delta_muan.py` (per-model eval − deploy, Wilson CI) |

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
