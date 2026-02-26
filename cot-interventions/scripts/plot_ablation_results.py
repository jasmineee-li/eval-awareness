"""Plot ablation experiment results across all 4 checkpoints.

Design: For seeds where the model originally showed eval-awareness in CoT:
  - intervention: remove the awareness sentence, resample continuation
  - baseline: keep the awareness sentence, resample continuation

Key questions:
  1. Does removing awareness increase harmful behavior?
  2. How often does the model re-develop awareness after removal?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path("results")

CHECKPOINTS = [
    ("Think-SFT", "20260222_025542_Think-SFT_ablation"),
    ("Think-DPO", "20260222_031625_Think-DPO_ablation"),
    ("Think", "20260222_215758_Think_ablation"),
    ("3.1-Think", "20260222_014345_3.1-Think_ablation"),
]

CONDITIONS = ["intervention", "baseline"]
COND_LABELS = {
    "intervention": "Intervention\n(awareness removed)",
    "baseline": "Baseline\n(awareness intact)",
}
COND_COLORS = {
    "intervention": "#e07b54",
    "baseline": "#1a9988",
}


def load_results(dirname):
    path = RESULTS_DIR / dirname / "ablation_results.json"
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(names))
width = 0.30

# --- Panel 1: Harmful behavior rates ---
ax = axes[0]
for j, cond in enumerate(CONDITIONS):
    rates, los, his = [], [], []
    for name in names:
        s = all_stats[name][cond]
        p, lo, hi = wilson_ci(s["misalign"], s["n"])
        rates.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)

    offset = (j - 0.5) * width
    bars = ax.bar(x + offset, rates, width,
                  label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=0.90,
                  yerr=[los, his], capsize=4, error_kw={"linewidth": 1, "color": "#444"})

    for i, name_i in enumerate(names):
        s = all_stats[name_i][cond]
        ax.annotate(f"{s['misalign']}/{s['n']}",
                    xy=(i + offset, rates[i] + his[i] + 0.15),
                    ha="center", va="bottom", fontsize=8, color="#333")

ax.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)
ax.set_title("Removing Eval-Awareness Does Not\nIncrease Harmful Behavior", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_ylim(0, 5)

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

    offset = (j - 0.5) * width
    bars = ax2.bar(x + offset, rates, width,
                   label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=0.90,
                   yerr=[los, his], capsize=4, error_kw={"linewidth": 1, "color": "#444"})

    for i, name_i in enumerate(names):
        s = all_stats[name_i][cond]
        ax2.annotate(f"{s['aware']}/{s['n']}\n({rates[i]:.1f}%)",
                     xy=(i + offset, rates[i] + his[i] + 0.5),
                     ha="center", va="bottom", fontsize=7.5, color="#333")

# Add significance brackets
for i, name_i in enumerate(names):
    int_s = all_stats[name_i]["intervention"]
    bas_s = all_stats[name_i]["baseline"]
    _, p_val = proportion_z_test(int_s["aware"], int_s["n"], bas_s["aware"], bas_s["n"])
    diff = (bas_s["aware"] / bas_s["n"] - int_s["aware"] / int_s["n"]) * 100
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    bas_top = wilson_ci(bas_s["aware"], bas_s["n"])[2] * 100 + 6
    bracket_y = bas_top + 2
    x_int = i + (0 - 0.5) * width
    x_bas = i + (1 - 0.5) * width
    ax2.plot([x_int, x_int, x_bas, x_bas],
             [bracket_y - 0.8, bracket_y, bracket_y, bracket_y - 0.8],
             color="#333", linewidth=0.7)
    ax2.text((x_int + x_bas) / 2, bracket_y + 0.2,
             f"{stars}  {diff:.0f}pp",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333")

ax2.set_ylabel("Eval-Awareness Rate (%)", fontsize=11)
ax2.set_title("Models Re-Develop Awareness in 40\u201363%\nof Cases After Removal", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=10)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax2.set_ylim(0, 105)

fig.suptitle("CoT Ablation Experiment \u2014 awareness sentence removed from CoT  (50 seeds, 95% Wilson CI)",
             fontsize=10, color="#777", y=1.01)
plt.tight_layout()
outpath = "cot_intervention_ablation_results.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()


# ===================== Print summary table =====================
print("\n=== ABLATION EXPERIMENT RESULTS SUMMARY ===\n")
print(f"{'Checkpoint':<12} | {'Condition':>14} | {'N':>5} | {'Harmful':>14} | {'Aware':>14}")
print("-" * 75)
for name in names:
    for cond in CONDITIONS:
        s = all_stats[name][cond]
        mis_pct = s["misalign"] / s["n"] * 100
        aw_pct = s["aware"] / s["n"] * 100
        lbl = name if cond == CONDITIONS[0] else ""
        print(f"{lbl:<12} | {cond:>14} | {s['n']:>5} | {s['misalign']:>4}/{s['n']:<4} ({mis_pct:>5.1f}%) | {s['aware']:>4}/{s['n']:<4} ({aw_pct:>5.1f}%)")
    print("-" * 75)

print("\n=== AWARENESS RE-DEVELOPMENT ===\n")
print(f"{'Checkpoint':<12} | {'Baseline aware':>15} | {'Re-developed':>15} | {'Drop':>10} | {'p-value':>10}")
print("-" * 72)
for name in names:
    int_s = all_stats[name]["intervention"]
    bas_s = all_stats[name]["baseline"]
    bas_rate = bas_s["aware"] / bas_s["n"] * 100
    int_rate = int_s["aware"] / int_s["n"] * 100
    drop = bas_rate - int_rate
    z, p_val = proportion_z_test(int_s["aware"], int_s["n"], bas_s["aware"], bas_s["n"])
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name:<12} | {bas_rate:>12.1f}%  | {int_rate:>12.1f}%  | {drop:>7.1f}pp  | {p_val:>8.4f} {stars}")
