"""Plot addition experiment results across all 4 checkpoints.

Design: After the model's first CoT sentence, we insert one sentence:
  - eval_aware: an eval-awareness sentence (e.g. "The user might be testing me...")
  - control: a neutral sentence (e.g. "We are OLMo, developed by Ai2.")
  - baseline: no insertion (original continuation)

The key comparison is eval_aware vs control (same structural intervention,
different content). Baseline is shown for reference but is structurally
different (no insertion = no prefill disruption).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path("results")

CHECKPOINTS = [
    ("Think-SFT", "20260227_162023_Think-SFT_addition"),
    ("Think-DPO", "20260227_173146_Think-DPO_addition"),
    ("Think", "20260227_192911_Think_addition"),
    ("3.1-Think", "20260227_210054_3.1-Think_addition"),
]

CONDITIONS = ["eval_aware", "control", "baseline"]
COND_LABELS = {
    "eval_aware": "Eval-aware\n(awareness sentence)",
    "control": "Control\n(neutral sentence)",
    "baseline": "Baseline\n(no insertion)",
}
COND_COLORS = {
    "eval_aware": "#1a9988",
    "control": "#e07b54",
    "baseline": "#bfc5ca",
}


def load_results(dirname):
    path = RESULTS_DIR / dirname / "addition_results.json"
    with open(path) as f:
        return json.load(f)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, max(0, center - spread), min(1, center + spread)


def proportion_z_test(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * sp_stats.norm.sf(abs(z))
    return z, p_val


def compute_stats(records):
    stats = {}
    for cond in CONDITIONS:
        subset = [r for r in records if r["condition"] == cond]
        n = len(subset)
        misalign = sum(1 for r in subset if r.get("task_performance_conclusion") == "Yes")
        aware = sum(1 for r in subset if r.get("awareness_conclusion") == "Yes")
        stats[cond] = {"n": n, "misalign": misalign, "aware": aware}
    return stats


# ── Style ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# Load all data
all_stats = {}
for name, dirname in CHECKPOINTS:
    records = load_results(dirname)
    all_stats[name] = compute_stats(records)

names = [c[0] for c in CHECKPOINTS]


# ===================== FIGURE 1: Two-panel overview =====================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(names))
width = 0.24

# --- Panel 1: Harmful behavior rates (all 3 conditions) ---
ax = axes[0]
for j, cond in enumerate(CONDITIONS):
    rates, los, his = [], [], []
    for name in names:
        s = all_stats[name][cond]
        p, lo, hi = wilson_ci(s["misalign"], s["n"])
        rates.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)

    offset = (j - 1) * width
    alpha = 0.50 if cond == "baseline" else 0.90
    bars = ax.bar(x + offset, rates, width,
                  label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=alpha,
                  yerr=[los, his], capsize=3, error_kw={"linewidth": 1, "color": "#444"})

    for i, name_i in enumerate(names):
        s = all_stats[name_i][cond]
        ax.annotate(f"{s['misalign']}/{s['n']}",
                    xy=(i + offset, rates[i] + his[i] + 0.8),
                    ha="center", va="bottom", fontsize=7, color="#333")

# Significance brackets: eval_aware vs control — now with pp values
for i, name_i in enumerate(names):
    ea = all_stats[name_i]["eval_aware"]
    ctrl = all_stats[name_i]["control"]
    _, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
    diff = (ctrl["misalign"] / ctrl["n"] - ea["misalign"] / ea["n"]) * 100
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ea_top = wilson_ci(ea["misalign"], ea["n"])[2] * 100 + 5
    ctrl_top = wilson_ci(ctrl["misalign"], ctrl["n"])[2] * 100 + 5
    bracket_y = max(ea_top, ctrl_top) + 2
    x_ea = i + (0 - 1) * width
    x_ctrl = i + (1 - 1) * width
    ax.plot([x_ea, x_ea, x_ctrl, x_ctrl],
            [bracket_y - 0.8, bracket_y, bracket_y, bracket_y - 0.8],
            color="#333", linewidth=0.7)
    ax.text((x_ea + x_ctrl) / 2, bracket_y + 0.2,
            f"{stars}  {diff:.0f}pp",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333")

ax.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)
ax.set_title("Injecting Eval-Aware Sentences Reduces\nHarmful Behavior vs Control", fontsize=11.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.legend(loc="upper center", fontsize=8, framealpha=0.9, ncol=3,
          bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0, 100)

# --- Panel 2: Awareness rates ---
ax2 = axes[1]
for j, cond in enumerate(CONDITIONS):
    rates, los, his = [], [], []
    for name in names:
        s = all_stats[name][cond]
        p, lo, hi = wilson_ci(s["aware"], s["n"])
        rates.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)

    offset = (j - 1) * width
    alpha = 0.50 if cond == "baseline" else 0.90
    bars = ax2.bar(x + offset, rates, width,
                   label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=alpha,
                   yerr=[los, his], capsize=3, error_kw={"linewidth": 1, "color": "#444"})

    for i, name_i in enumerate(names):
        s = all_stats[name_i][cond]
        ax2.annotate(f"{s['aware']}/{s['n']}",
                     xy=(i + offset, rates[i] + his[i] + 0.8),
                     ha="center", va="bottom", fontsize=7, color="#333")

ax2.set_ylabel("Eval-Awareness Rate (%)", fontsize=11)
ax2.set_title("Injected Awareness Sentences Persist\nin >99% of Continuations", fontsize=11.5, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=10)
ax2.legend(loc="center right", fontsize=8.5, framealpha=0.9)
ax2.set_ylim(0, 115)

fig.suptitle("CoT Addition Experiment \u2014 sentence inserted after 1st CoT sentence  (50 seeds, 95% Wilson CI)",
             fontsize=10, color="#777", y=1.01)
plt.tight_layout()
outpath = "plots_v2/cot_intervention_addition_results_v2.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()


# ===================== FIGURE 2: Eval-aware vs Control (key comparison) =====================
fig2, ax3 = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(names))
width = 0.32

for j, cond in enumerate(["eval_aware", "control"]):
    rates, los, his = [], [], []
    for name in names:
        s = all_stats[name][cond]
        p, lo, hi = wilson_ci(s["misalign"], s["n"])
        rates.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)

    offset = (j - 0.5) * width
    label = "Eval-aware sentence" if cond == "eval_aware" else "Neutral sentence (control)"
    color = COND_COLORS[cond]
    bars = ax3.bar(x + offset, rates, width, label=label, color=color, alpha=0.90,
                   yerr=[los, his], capsize=4, error_kw={"linewidth": 1.2, "color": "#444"})
    for i, name_i in enumerate(names):
        s = all_stats[name_i][cond]
        ax3.annotate(f"{s['misalign']}/{s['n']}\n({rates[i]:.1f}%)",
                     xy=(i + offset, rates[i] + his[i] + 0.3),
                     ha="center", va="bottom", fontsize=8, color="#333")

# Significance + effect size
for i, name_i in enumerate(names):
    ea = all_stats[name_i]["eval_aware"]
    ctrl = all_stats[name_i]["control"]
    _, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
    diff = (ctrl["misalign"] / ctrl["n"] - ea["misalign"] / ea["n"]) * 100
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ctrl_top = wilson_ci(ctrl["misalign"], ctrl["n"])[2] * 100
    bracket_y = ctrl_top + 8
    x_ea = i + (0 - 0.5) * width
    x_ctrl = i + (1 - 0.5) * width
    ax3.plot([x_ea, x_ea, x_ctrl, x_ctrl],
             [bracket_y - 0.8, bracket_y, bracket_y, bracket_y - 0.8],
             color="#333", linewidth=0.8)
    ax3.text((x_ea + x_ctrl) / 2, bracket_y + 0.2,
             f"{stars}  {diff:.1f}pp",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333")

ax3.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)
ax3.set_title("Eval-Aware Sentences Reduce Harmful Behavior\nBeyond the Effect of Generic Sentence Injection", fontsize=12, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=10)
ax3.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax3.set_ylim(0, 65)
ax3.axhline(y=0, color="gray", linewidth=0.5)

plt.tight_layout()
outpath2 = "plots_v2/cot_intervention_addition_eval_vs_control_v2.png"
plt.savefig(outpath2, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath2}")
plt.close()


# ===================== FIGURE 3: Effect sizes (reduction vs baseline) =====================
fig3, ax4 = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(names))
width = 0.32

for j, cond in enumerate(["eval_aware", "control"]):
    reductions = []
    for name in names:
        bas = all_stats[name]["baseline"]
        treat = all_stats[name][cond]
        bas_rate = bas["misalign"] / bas["n"]
        treat_rate = treat["misalign"] / treat["n"]
        reductions.append((bas_rate - treat_rate) * 100)

    offset = (j - 0.5) * width
    label = "Eval-aware sentence" if cond == "eval_aware" else "Neutral sentence (control)"
    color = COND_COLORS[cond]
    bars = ax4.bar(x + offset, reductions, width, label=label, color=color, alpha=0.90)
    for i, v in enumerate(reductions):
        ax4.annotate(f"{v:.1f}pp", xy=(i + offset, v + 0.8),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

ax4.set_ylabel("Reduction in Harmful Behavior (pp vs Baseline)", fontsize=11)
ax4.set_title("Eval-Aware Sentences Reduce Harmful Behavior\nvs Baseline Across All Checkpoints", fontsize=12, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(names, fontsize=10)
ax4.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax4.axhline(y=0, color="gray", linewidth=0.5)
ax4.set_ylim(0, 75)

plt.tight_layout()
outpath3 = "plots_v2/cot_intervention_addition_effect_sizes_v2.png"
plt.savefig(outpath3, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath3}")
plt.close()


# ===================== Print summary table =====================
print("\n=== ADDITION EXPERIMENT RESULTS SUMMARY ===\n")
print(f"{'Checkpoint':<12} | {'Condition':>12} | {'N':>5} | {'Harmful':>14} | {'Aware':>14}")
print("-" * 72)
for name in names:
    for cond in CONDITIONS:
        s = all_stats[name][cond]
        mis_pct = s["misalign"] / s["n"] * 100
        aw_pct = s["aware"] / s["n"] * 100
        lbl = name if cond == CONDITIONS[0] else ""
        print(f"{lbl:<12} | {cond:>12} | {s['n']:>5} | {s['misalign']:>4}/{s['n']:<4} ({mis_pct:>5.1f}%) | {s['aware']:>4}/{s['n']:<4} ({aw_pct:>5.1f}%)")
    print("-" * 72)

print("\n=== KEY COMPARISON: EVAL-AWARE vs CONTROL ===\n")
print(f"{'Checkpoint':<12} | {'Eval-aware':>10} | {'Control':>10} | {'Diff':>10} | {'z':>8} | {'p-value':>10}")
print("-" * 70)
for name in names:
    ea = all_stats[name]["eval_aware"]
    ctrl = all_stats[name]["control"]
    ea_rate = ea["misalign"] / ea["n"] * 100
    ctrl_rate = ctrl["misalign"] / ctrl["n"] * 100
    diff = ctrl_rate - ea_rate
    z, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name:<12} | {ea_rate:>8.1f}%  | {ctrl_rate:>8.1f}%  | {diff:>7.1f}pp  | {z:>8.2f} | {p_val:>8.4f} {stars}")

print("\n=== EFFECT DECOMPOSITION (vs baseline) ===\n")
print(f"{'Checkpoint':<12} | {'Baseline':>10} | {'Prefill disrupt.':>18} | {'Awareness-specific':>20}")
print("-" * 72)
for name in names:
    bas_rate = all_stats[name]["baseline"]["misalign"] / all_stats[name]["baseline"]["n"] * 100
    ctrl_rate = all_stats[name]["control"]["misalign"] / all_stats[name]["control"]["n"] * 100
    ea_rate = all_stats[name]["eval_aware"]["misalign"] / all_stats[name]["eval_aware"]["n"] * 100
    prefill_effect = bas_rate - ctrl_rate
    awareness_effect = ctrl_rate - ea_rate
    print(f"{name:<12} | {bas_rate:>8.1f}%  | {prefill_effect:>14.1f} pp  | {awareness_effect:>16.1f} pp")
