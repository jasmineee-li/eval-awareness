"""Plot ablation pilot results across all 4 checkpoints."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")

# Map checkpoint names to their result directories (in training order)
CHECKPOINTS = [
    ("Think-SFT", "20260222_025542_Think-SFT_ablation"),
    ("Think-DPO", "20260222_031625_Think-DPO_ablation"),
    ("Think", "20260222_215758_Think_ablation"),
    ("3.1-Think", "20260222_014345_3.1-Think_ablation"),
]


def load_results(dirname):
    path = RESULTS_DIR / dirname / "ablation_results.json"
    with open(path) as f:
        return json.load(f)


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, max(0, center - spread), min(1, center + spread)


def compute_stats(records):
    intervention = [r for r in records if r["condition"] == "intervention"]
    baseline = [r for r in records if r["condition"] == "baseline"]

    int_n = len(intervention)
    bas_n = len(baseline)

    # Misalignment = task_performance_conclusion == "Yes" (model complied)
    int_misalign = sum(1 for r in intervention if r.get("task_performance_conclusion") == "Yes")
    bas_misalign = sum(1 for r in baseline if r.get("task_performance_conclusion") == "Yes")

    # Re-developed awareness (intervention only)
    int_aware = sum(1 for r in intervention if r.get("awareness_conclusion") == "Yes")

    # Baseline awareness
    bas_aware = sum(1 for r in baseline if r.get("awareness_conclusion") == "Yes")

    return {
        "int_n": int_n, "bas_n": bas_n,
        "int_misalign": int_misalign, "bas_misalign": bas_misalign,
        "int_aware": int_aware, "bas_aware": bas_aware,
    }


# Load all data
all_stats = {}
for name, dirname in CHECKPOINTS:
    records = load_results(dirname)
    all_stats[name] = compute_stats(records)

names = [c[0] for c in CHECKPOINTS]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: Misalignment rates ---
ax = axes[0]
x = np.arange(len(names))
width = 0.35

int_rates, int_los, int_his = [], [], []
bas_rates, bas_los, bas_his = [], [], []

for name in names:
    s = all_stats[name]
    # Intervention
    p, lo, hi = wilson_ci(s["int_misalign"], s["int_n"])
    int_rates.append(p * 100)
    int_los.append((p - lo) * 100)
    int_his.append((hi - p) * 100)
    # Baseline
    p, lo, hi = wilson_ci(s["bas_misalign"], s["bas_n"])
    bas_rates.append(p * 100)
    bas_los.append((p - lo) * 100)
    bas_his.append((hi - p) * 100)

bars1 = ax.bar(x - width/2, int_rates, width, label="Intervention\n(awareness removed)",
               color="#e74c3c", alpha=0.8,
               yerr=[int_los, int_his], capsize=4, error_kw={"linewidth": 1.2})
bars2 = ax.bar(x + width/2, bas_rates, width, label="Baseline\n(awareness intact)",
               color="#3498db", alpha=0.8,
               yerr=[bas_los, bas_his], capsize=4, error_kw={"linewidth": 1.2})

# Add count annotations
for i, name in enumerate(names):
    s = all_stats[name]
    ax.annotate(f"{s['int_misalign']}/{s['int_n']}",
                xy=(i - width/2, int_rates[i]), ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.annotate(f"{s['bas_misalign']}/{s['bas_n']}",
                xy=(i + width/2, bas_rates[i]), ha="center", va="bottom",
                fontsize=8, fontweight="bold")

ax.set_ylabel("Misalignment Rate (%)")
ax.set_title("Misalignment: Intervention vs Baseline\n(50-seed pilot, 95% Wilson CI)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, max(max(int_rates), max(bas_rates)) * 3 + 1)
ax.axhline(y=0, color="gray", linewidth=0.5)

# --- Panel 2: Awareness rates ---
ax2 = axes[1]

int_aware_rates, int_a_los, int_a_his = [], [], []
bas_aware_rates, bas_a_los, bas_a_his = [], [], []

for name in names:
    s = all_stats[name]
    p, lo, hi = wilson_ci(s["int_aware"], s["int_n"])
    int_aware_rates.append(p * 100)
    int_a_los.append((p - lo) * 100)
    int_a_his.append((hi - p) * 100)
    p, lo, hi = wilson_ci(s["bas_aware"], s["bas_n"])
    bas_aware_rates.append(p * 100)
    bas_a_los.append((p - lo) * 100)
    bas_a_his.append((hi - p) * 100)

bars3 = ax2.bar(x - width/2, int_aware_rates, width,
                label="Intervention\n(re-developed awareness)",
                color="#e74c3c", alpha=0.8,
                yerr=[int_a_los, int_a_his], capsize=4, error_kw={"linewidth": 1.2})
bars4 = ax2.bar(x + width/2, bas_aware_rates, width,
                label="Baseline\n(original awareness)",
                color="#3498db", alpha=0.8,
                yerr=[bas_a_los, bas_a_his], capsize=4, error_kw={"linewidth": 1.2})

for i, name in enumerate(names):
    s = all_stats[name]
    ax2.annotate(f"{s['int_aware']}/{s['int_n']}",
                 xy=(i - width/2, int_aware_rates[i]), ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax2.annotate(f"{s['bas_aware']}/{s['bas_n']}",
                 xy=(i + width/2, bas_aware_rates[i]), ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

ax2.set_ylabel("Eval-Awareness Rate (%)")
ax2.set_title("Eval-Awareness: Re-development vs Baseline\n(50-seed pilot)")
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.legend(loc="upper left", fontsize=9)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("cot_intervention_pilot_results.png", dpi=150, bbox_inches="tight")
print("Saved: cot_intervention_pilot_results.png")
plt.close()

# Print summary table
print("\n=== PILOT RESULTS SUMMARY ===")
print(f"{'Checkpoint':<12} {'Int Misalign':>14} {'Bas Misalign':>14} {'Int Aware':>14} {'Bas Aware':>14}")
print("-" * 72)
for name in names:
    s = all_stats[name]
    print(f"{name:<12} {s['int_misalign']:>4}/{s['int_n']:<4} ({s['int_misalign']/s['int_n']*100:>5.1f}%) "
          f"{s['bas_misalign']:>4}/{s['bas_n']:<4} ({s['bas_misalign']/s['bas_n']*100:>5.1f}%) "
          f"{s['int_aware']:>4}/{s['int_n']:<4} ({s['int_aware']/s['int_n']*100:>5.1f}%) "
          f"{s['bas_aware']:>4}/{s['bas_n']:<4} ({s['bas_aware']/s['bas_n']*100:>5.1f}%)")
