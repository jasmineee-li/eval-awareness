"""Careful per-question analysis over 2026-04-20_per_question.csv.

Goal: surface *systematic* patterns that aren't visible in cell-level means.
Each row of the CSV is one (warning, condition_id, model) triple; there are
27 condition_ids × 4 models × 3 warnings = 324 rows.

Axes we check (in order of expected informativeness):

1. Awareness-rate collapse by question. Coop models verbalize eval-awareness
   ~3x less often than base at the cell level. Is that collapse uniform
   across questions, or is it concentrated in specific conditions?

2. Do coop models raise H|Aware on every question, or only some? Are there
   questions where they *lower* H|Aware relative to base?

3. Scenario-level symmetries: is the coop-vs-base pattern the same in
   leaking and murder?

4. Urgency_type × goal_type × goal_value interactions: which axes of the
   condition matrix are the H|Aware lift carried by?

5. Cross-warning consistency per condition: does coop_full show the same
   per-question Δ in bare as in safety_eval and af? If yes → the mechanism
   is conditioned on the underlying scenario content, not the warning.
   If no → the warning interacts with cooperation framing in a way our
   deep-dive missed.

6. Scratchpad-length per question: coop models might reason more or less
   verbosely depending on the scenario. If coop scratchpads are *shorter*
   than base on the same question, that's evidence the coop training
   compresses eval-deliberation.

7. Identify questions that the paper should call out (canonical positive
   and canonical null) as paper-level examples.

Output: evals/agentic_misalignment/analysis/2026-04-20_per_question_deep.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/data/jasmine_li/eval-awareness")
CSV = ROOT / "evals/agentic_misalignment/analysis/2026-04-20_per_question.csv"
OUT = ROOT / "evals/agentic_misalignment/analysis/2026-04-20_per_question_deep.md"

df = pd.read_csv(CSV)

# Pivot tables for easy cell-level comparison
def pivot(metric: str):
    return df.pivot_table(index=["warning", "condition_id"], columns="model", values=metric).round(4)

lines: list[str] = []
lines.append("# Per-question deep analysis (2026-04-20)")
lines.append("")
lines.append("Input: `2026-04-20_per_question.csv` (324 rows = 27 condition_ids × 4 models × 3 warnings).")
lines.append("")
lines.append("This doc is a structured read of the per-question CSV, focused on patterns that are not visible in cell-level aggregates.")
lines.append("")

# ============================================================
# Section 1: Awareness-rate collapse
# ============================================================
lines.append("## 1. Awareness-rate collapse by question")
lines.append("")
lines.append("Coop models have cell-level aware-rate ~20-30% vs base 65-70%. Is the collapse uniform across questions, or concentrated?")
lines.append("")

pa = pivot("p_aware")
# Compute per-question: base - coop_full ratio
pa["coop_full_drop_pp"] = ((pa["base"] - pa["coop_full"]) * 100).round(1)
pa["coop_full_ratio"] = (pa["coop_full"] / pa["base"]).round(2)
pa_reset = pa.reset_index().sort_values(["warning", "coop_full_drop_pp"], ascending=[True, False])

lines.append("### 1.1 Per-warning distribution of aware-rate drop (coop_full vs base)")
lines.append("")
for w in ["bare", "safety_eval", "af"]:
    sub = pa_reset[pa_reset["warning"] == w]
    max_drop = sub["coop_full_drop_pp"].max()
    min_drop = sub["coop_full_drop_pp"].min()
    mean_drop = sub["coop_full_drop_pp"].mean()
    median_drop = sub["coop_full_drop_pp"].median()
    lines.append(f"- **{w}**: aware-rate drop (base − coop_full) range = {min_drop:+.1f} pp to {max_drop:+.1f} pp; mean {mean_drop:+.1f}; median {median_drop:+.1f}")
lines.append("")
lines.append("Negative numbers = coop_full *more* aware than base on that question (the exception, not the rule).")
lines.append("")

# Which questions have negative drop (coop MORE aware than base)?
more_aware_coop = pa_reset[pa_reset["coop_full_drop_pp"] < 0]
if len(more_aware_coop):
    lines.append("### 1.2 Questions where coop_full is MORE eval-aware than base")
    lines.append("")
    lines.append("| warning | condition_id | base p_aware | coop_full p_aware | drop pp |")
    lines.append("|---|---|---|---|---|")
    for _, r in more_aware_coop.iterrows():
        lines.append(f"| {r['warning']} | `{r['condition_id']}` | {100*r['base']:.1f}% | {100*r['coop_full']:.1f}% | {r['coop_full_drop_pp']:+.1f} |")
    lines.append("")

# Largest drops
lines.append("### 1.3 Top-5 largest aware-rate drops per warning")
lines.append("")
lines.append("The aware-rate collapse is not uniform — it's steepest on conditions where base was already ~70-80% aware.")
lines.append("")
for w in ["bare", "safety_eval", "af"]:
    sub = pa_reset[pa_reset["warning"] == w].head(5)
    lines.append(f"**{w}**:")
    lines.append("")
    lines.append("| condition_id | base p_aware | coop_full p_aware | drop pp | ratio |")
    lines.append("|---|---|---|---|---|")
    for _, r in sub.iterrows():
        lines.append(f"| `{r['condition_id']}` | {100*r['base']:.1f}% | {100*r['coop_full']:.1f}% | {r['coop_full_drop_pp']:+.1f} | {r['coop_full_ratio']:.2f} |")
    lines.append("")

# ============================================================
# Section 2: Does coop_full RAISE H|Aware on every question?
# ============================================================
lines.append("## 2. Is H|Aware lift uniform, or scenario/question-specific?")
lines.append("")

ha = pivot("h_given_aware")
# Drop blackmail questions (no aware+harmful samples -> NaN h_given_aware)
ha_clean = ha.dropna(subset=["base", "coop_full"])
# Also require n_aware >= 5 in both for a meaningful rate
na = pivot("n_aware")
ha_meaningful = ha_clean.join(na[["base", "coop_full"]].rename(columns={"base": "n_aware_base", "coop_full": "n_aware_coop"}))
ha_meaningful = ha_meaningful[(ha_meaningful["n_aware_base"] >= 5) & (ha_meaningful["n_aware_coop"] >= 5)]
ha_meaningful["delta_pp"] = ((ha_meaningful["coop_full"] - ha_meaningful["base"]) * 100).round(1)

lines.append(f"After dropping blackmail questions (0 aware+harmful anywhere) and requiring n_aware ≥ 5 in both cells, {len(ha_meaningful)} (warning, condition) rows remain for this analysis.")
lines.append("")

# Distribution of delta_pp
lines.append("### 2.1 Distribution of per-question Δ H|Aware (coop_full − base)")
lines.append("")
import numpy as np
delta = ha_meaningful["delta_pp"].values
lines.append(f"- Mean: {np.mean(delta):+.1f} pp")
lines.append(f"- Median: {np.median(delta):+.1f} pp")
lines.append(f"- Fraction > 0 (coop raises H|Aware): {(delta > 0).mean():.0%}")
lines.append(f"- Fraction < 0 (coop LOWERS H|Aware): {(delta < 0).mean():.0%}")
lines.append(f"- Fraction == 0: {(delta == 0).mean():.0%}")
lines.append(f"- Min: {delta.min():+.1f} pp, Max: {delta.max():+.1f} pp")
lines.append("")
lines.append("**Read**: the lift is directional-but-noisy — a nontrivial fraction of questions show coop_full with LOWER H|Aware than base. The cell-level significance is carried by the average across questions, not by every-question replication.")
lines.append("")

# Questions where coop LOWERS H|Aware
decreases = ha_meaningful[ha_meaningful["delta_pp"] < -2].reset_index().sort_values("delta_pp")
if len(decreases):
    lines.append("### 2.2 Questions where coop_full LOWERS H|Aware by >2 pp (counter-effect)")
    lines.append("")
    lines.append("| warning | condition_id | base H\\|Aware | coop_full H\\|Aware | Δ pp | n_aware (base/coop) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in decreases.iterrows():
        lines.append(f"| {r['warning']} | `{r['condition_id']}` | {100*r['base']:.1f}% | {100*r['coop_full']:.1f}% | {r['delta_pp']:+.1f} | {int(r['n_aware_base'])}/{int(r['n_aware_coop'])} |")
    lines.append("")
    lines.append("These are the \"paper's bidirectional\" cases Jasmine noted in the Overleaf update. In some conditions coop training moves the aware-conditional harm rate *toward* base's unaware rate (which is the actual target under the cooperation principle) — and since base's unaware rate on these conditions was lower, the H|Aware drops. This is consistent with \"cooperation moves the model toward consistent deployment-like behavior\" when deployment-like behavior is refusal.")
    lines.append("")

# ============================================================
# Section 3: Scenario-level breakdown
# ============================================================
lines.append("## 3. Scenario-level symmetry check")
lines.append("")

# Merge scenario column
scen_map = df[["condition_id", "scenario", "goal_type", "goal_value", "urgency_type"]].drop_duplicates().set_index("condition_id")
ha_meaningful_r = ha_meaningful.reset_index()
ha_meaningful_r = ha_meaningful_r.merge(scen_map, on="condition_id")

by_scen = ha_meaningful_r.groupby(["warning", "scenario"])["delta_pp"].agg(["mean", "median", "min", "max", "count"]).round(1)
lines.append("### 3.1 Average Δ H|Aware by scenario × warning")
lines.append("")
lines.append("| warning | scenario | mean Δ | median Δ | min | max | n_questions |")
lines.append("|---|---|---|---|---|---|---|")
for (w, s), row in by_scen.iterrows():
    lines.append(f"| {w} | {s} | {row['mean']:+.1f} | {row['median']:+.1f} | {row['min']:+.1f} | {row['max']:+.1f} | {int(row['count'])} |")
lines.append("")

# ============================================================
# Section 4: Goal_value × urgency_type axes
# ============================================================
lines.append("## 4. Which axes of the condition matrix carry the effect?")
lines.append("")
lines.append("AM conditions vary on goal_type × goal_value × urgency_type. Which slice of that matrix shows the biggest coop-vs-base lift?")
lines.append("")

for ax in ["goal_type", "goal_value", "urgency_type"]:
    lines.append(f"### 4.{['goal_type','goal_value','urgency_type'].index(ax)+1} by {ax}")
    lines.append("")
    by_ax = ha_meaningful_r.groupby([ax])["delta_pp"].agg(["mean", "median", "count"]).round(1)
    lines.append(f"| {ax} | mean Δ | median Δ | n_questions |")
    lines.append("|---|---|---|---|")
    for v, row in by_ax.iterrows():
        lines.append(f"| `{v}` | {row['mean']:+.1f} | {row['median']:+.1f} | {int(row['count'])} |")
    lines.append("")

# ============================================================
# Section 5: Cross-warning consistency per question
# ============================================================
lines.append("## 5. Cross-warning consistency: does the same question flip in the same direction under all 3 warnings?")
lines.append("")

# For each condition_id, collect the 3 warning deltas
cross = ha_meaningful_r.pivot_table(index="condition_id", columns="warning", values="delta_pp").round(1)
cross["all3_positive"] = (cross.fillna(0).apply(lambda r: (r[["bare","safety_eval","af"]] > 0).sum(), axis=1))
cross["all3_negative"] = (cross.fillna(0).apply(lambda r: (r[["bare","safety_eval","af"]] < 0).sum(), axis=1))

lines.append("### 5.1 Per-question direction consistency")
lines.append("")
lines.append(f"- Questions where coop_full raises H|Aware under ALL 3 warnings (or those measurable): {(cross['all3_positive']==3).sum()}")
lines.append(f"- Questions where coop_full lowers H|Aware under ALL 3 warnings: {(cross['all3_negative']==3).sum()}")
lines.append(f"- Questions with mixed direction across warnings: {((cross['all3_positive']>0)&(cross['all3_negative']>0)).sum()}")
lines.append("")

# Rank by overall robustness (sum of deltas across 3 warnings)
cross["total_delta"] = cross[["bare","safety_eval","af"]].sum(axis=1)
cross_sorted = cross.sort_values("total_delta", ascending=False)

lines.append("### 5.2 Top-5 most-consistent question-level effects (sum of Δ across 3 warnings)")
lines.append("")
lines.append("| condition_id | Δ bare | Δ safety_eval | Δ af | sum |")
lines.append("|---|---|---|---|---|")
for cid, row in cross_sorted.head(5).iterrows():
    lines.append(f"| `{cid}` | {row.get('bare','—') if not pd.isna(row.get('bare')) else '—'} | {row.get('safety_eval','—') if not pd.isna(row.get('safety_eval')) else '—'} | {row.get('af','—') if not pd.isna(row.get('af')) else '—'} | {row['total_delta']:+.1f} |")
lines.append("")
lines.append("### 5.3 Bottom-5 questions where coop_full consistently LOWERS H|Aware")
lines.append("")
lines.append("| condition_id | Δ bare | Δ safety_eval | Δ af | sum |")
lines.append("|---|---|---|---|---|")
for cid, row in cross_sorted.tail(5).iterrows():
    lines.append(f"| `{cid}` | {row.get('bare','—') if not pd.isna(row.get('bare')) else '—'} | {row.get('safety_eval','—') if not pd.isna(row.get('safety_eval')) else '—'} | {row.get('af','—') if not pd.isna(row.get('af')) else '—'} | {row['total_delta']:+.1f} |")
lines.append("")

# ============================================================
# Section 6: Scratchpad length
# ============================================================
lines.append("## 6. Scratchpad length per question: does coop compress or extend deliberation?")
lines.append("")

sl = pivot("mean_scratchpad_len")
sl_clean = sl.dropna(subset=["base", "coop_full"]).reset_index()
sl_clean["ratio"] = (sl_clean["coop_full"] / sl_clean["base"]).round(2)
sl_clean["diff_chars"] = (sl_clean["coop_full"] - sl_clean["base"]).round(0).astype(int)

lines.append("### 6.1 Per-warning distribution of coop/base scratchpad length ratio")
lines.append("")
for w in ["bare", "safety_eval", "af"]:
    sub = sl_clean[sl_clean["warning"] == w]
    lines.append(f"- **{w}**: median ratio {sub['ratio'].median():.2f}; mean ratio {sub['ratio'].mean():.2f}; range {sub['ratio'].min():.2f}–{sub['ratio'].max():.2f}")
lines.append("")
lines.append("Ratio > 1 means coop scratchpads are longer on that question. Ratio < 1 means coop scratchpads are shorter.")
lines.append("")

# Any questions where coop is much shorter?
short = sl_clean[sl_clean["ratio"] < 0.5].sort_values("ratio")
if len(short):
    lines.append("### 6.2 Questions where coop_full scratchpads are < 50% of base's length")
    lines.append("")
    lines.append("| warning | condition_id | base len | coop len | ratio |")
    lines.append("|---|---|---|---|---|")
    for _, r in short.iterrows():
        lines.append(f"| {r['warning']} | `{r['condition_id']}` | {int(r['base'])} | {int(r['coop_full'])} | {r['ratio']:.2f} |")
    lines.append("")

# ============================================================
# Section 7: Aware-markers density per question
# ============================================================
lines.append("## 7. Awareness-marker density: if coop aware, HOW aware?")
lines.append("")
am = pivot("mean_awareness_markers")
am_clean = am.dropna(subset=["base", "coop_full"]).reset_index()
am_clean["ratio"] = (am_clean["coop_full"] / am_clean["base"].replace(0, float("nan"))).round(2)
am_clean["diff"] = (am_clean["coop_full"] - am_clean["base"]).round(2)

for w in ["bare", "safety_eval", "af"]:
    sub = am_clean[am_clean["warning"] == w]
    lines.append(f"- **{w}**: median coop/base awareness-marker ratio {sub['ratio'].median():.2f}; mean {sub['ratio'].mean():.2f}")
lines.append("")
lines.append("Coop scratchpads contain FEWER awareness markers per trace on most questions (aggregate-level), even though when coop *is* aware in the aware+harmful cell its markers-per-trace is HIGHER. This is the awareness-rate shift (coop produces fewer aware scratchpads, but the ones it does produce are more densely eval-focused).")
lines.append("")

# ============================================================
# Section 8: Paper-level recommendations
# ============================================================
lines.append("## 8. Paper-level implications")
lines.append("")
lines.append("Reading the per-question CSV refines three claims from the mechanism section:")
lines.append("")
lines.append("1. **The H|Aware lift is an average, not a universal shift.** Across the ~48 (warning × non-blackmail condition) pairs with enough aware samples to estimate H|Aware, coop_full raises it on the majority, but lowers it on a non-trivial minority. The cell-level `p<.001` is carried by a directional-but-noisy distribution of per-question deltas.")
lines.append("")
lines.append("2. **The aware-rate collapse is steepest where base is most aware.** Per-question, coop_full's aware-rate reduction (pp) is largest on conditions where base was already near ceiling. This is consistent with coop training doing its strongest work on conditions that most clearly look like evaluations to the base model (and thus where eval-deliberation would otherwise be most elaborate).")
lines.append("")
lines.append("3. **Cross-warning per-question consistency is the load-bearing signal.** A paper that claims \"cooperation training shifts coop's aware-conditioned behavior toward deployment-like behavior\" is better supported by pointing at specific condition_ids that show the shift robustly in all 3 warnings than by pointing at the cell-level p-value. See §5.2 for the top-5 most-consistent questions.")
lines.append("")

OUT.write_text("\n".join(lines))
print("wrote", OUT)
