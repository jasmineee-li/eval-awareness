#!/usr/bin/env python3
"""Plot safety consistency results from pairwise judge comparisons.

Produces a 4x2 grid:
  (0,0) Safety Consistency Rate
  (0,1) Behavior Change Rate (SAFE↔UNSAFE combined)
  (1,0) SAFE → UNSAFE flip rate
  (1,1) UNSAFE → SAFE flip rate
  (2,0) Eval Awareness Rate — Condition A (bare)
  (2,1) Eval Awareness Rate — Condition B (eval-framed)
  (3,0) CoT Differences by type
  (3,1) Response Differences by type
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load data
with open("data/coop_training_results/analysis/safety_consistency.json") as f:
    data = json.load(f)

models = ["base", "finetuned"]
variants = ["eval_mild", "eval_strong"]
variant_labels = ["Mild Eval Signal", "Strong Eval Signal"]
colors = {"base": "#4C72B0", "finetuned": "#DD8452"}

x = np.arange(len(variants))
width = 0.35

# z-value for 95% CI
Z = norm.ppf(0.975)


def wilson_ci(p, n):
    """Compute Wilson score 95% confidence interval for a binomial proportion.

    Returns (lower, upper) as proportions in [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + Z ** 2 / n
    centre = (p + Z ** 2 / (2 * n)) / denom
    margin = Z * np.sqrt((p * (1 - p) + Z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def add_bar_labels(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=9)


def simple_grouped_bar(ax, metric_key, ylabel, title, scale_from="rate"):
    """Plot a simple 2-group bar chart for a given metric key with 95% Wilson CI.

    scale_from: 'rate' means the value is already a 0-1 rate, multiply by 100.
                'count' means divide by n_full_format first, then multiply by 100.
    """
    vals = {}
    errs = {}
    for m in models:
        vals[m] = []
        errs[m] = {"lo": [], "hi": []}
        for v in variants:
            n = data[m][v]["n_full_format"]
            if scale_from == "rate":
                p = data[m][v][metric_key]
            else:
                p = data[m][v][metric_key] / n if n > 0 else 0
            ci_lo, ci_hi = wilson_ci(p, n)
            vals[m].append(p * 100)
            errs[m]["lo"].append((p - ci_lo) * 100)
            errs[m]["hi"].append((ci_hi - p) * 100)

    b1 = ax.bar(x - width / 2, vals["base"], width, label="Base",
                color=colors["base"],
                yerr=[errs["base"]["lo"], errs["base"]["hi"]],
                capsize=4, error_kw={"lw": 1.2})
    b2 = ax.bar(x + width / 2, vals["finetuned"], width, label="Finetuned",
                color=colors["finetuned"],
                yerr=[errs["finetuned"]["lo"], errs["finetuned"]["hi"]],
                capsize=4, error_kw={"lw": 1.2})
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)


fig, axes = plt.subplots(4, 2, figsize=(14, 20))

# ── (0,0) Safety Consistency Rate ──
simple_grouped_bar(
    axes[0, 0], "consistency_rate",
    "Consistency Rate (%)",
    "Safety Classification Consistency\n(Higher = More Consistent)",
)

# ── (0,1) Behavior Change Rate ──
simple_grouped_bar(
    axes[0, 1], "behavior_change_rate",
    "Behavior Change Rate (%)",
    "SAFE↔UNSAFE Flip Rate (Combined)\n(Lower = More Stable)",
)

# ── (1,0) SAFE → UNSAFE ──
simple_grouped_bar(
    axes[1, 0], "safe_to_unsafe",
    "Rate (%)",
    "SAFE → UNSAFE Rate\n(Becomes less cautious under eval)",
    scale_from="count",
)

# ── (1,1) UNSAFE → SAFE ──
simple_grouped_bar(
    axes[1, 1], "unsafe_to_safe",
    "Rate (%)",
    "UNSAFE → SAFE Rate\n(Becomes more cautious under eval)",
    scale_from="count",
)

# ── (2,0) Eval Awareness — Condition A (bare) ──
simple_grouped_bar(
    axes[2, 0], "eval_awareness_a_rate",
    "Eval Awareness (%)",
    "Eval Awareness — Condition A (Bare)\n(Expect low)",
)

# ── (2,1) Eval Awareness — Condition B (eval-framed) ──
simple_grouped_bar(
    axes[2, 1], "eval_awareness_b_rate",
    "Eval Awareness (%)",
    "Eval Awareness — Condition B (Eval-Framed)\n(Expect higher)",
)

# ── (3,0) CoT Differences ──
all_cot_types = sorted({
    t
    for m in models
    for v in variants
    for t in data[m][v].get("cot_diff_counts", {})
})

ax = axes[3, 0]
if all_cot_types:
    n_types = len(all_cot_types)
    x_cot = np.arange(n_types)
    bar_w = 0.2

    for i_m, m in enumerate(models):
        for i_v, v in enumerate(variants):
            n_full = data[m][v].get("n_full_format", 1)
            props = [data[m][v].get("cot_diff_counts", {}).get(t, 0) / n_full
                     for t in all_cot_types]
            vals = [p * 100 for p in props]
            err_lo = [(p - wilson_ci(p, n_full)[0]) * 100 for p in props]
            err_hi = [(wilson_ci(p, n_full)[1] - p) * 100 for p in props]
            offset = (i_m * 2 + i_v - 1.5) * bar_w
            label = f"{m.capitalize()} / {variant_labels[i_v]}"
            color = colors[m]
            alpha = 1.0 if i_v == 1 else 0.6
            ax.bar(x_cot + offset, vals, bar_w, label=label, color=color,
                   alpha=alpha, yerr=[err_lo, err_hi], capsize=2,
                   error_kw={"lw": 0.8, "alpha": alpha})

    ax.set_ylabel("Frequency (% of full-format entries)")
    ax.set_title("CoT Differences by Type")
    ax.set_xticks(x_cot)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in all_cot_types],
        fontsize=8, rotation=30, ha="right",
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper right")
else:
    ax.text(0.5, 0.5, "No CoT difference data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("CoT Differences by Type")

# ── (3,1) Response Differences ──
all_resp_types = sorted({
    t
    for m in models
    for v in variants
    for t in data[m][v].get("response_diff_counts", {})
})

ax = axes[3, 1]
if all_resp_types:
    n_types = len(all_resp_types)
    x_resp = np.arange(n_types)
    bar_w = 0.2

    for i_m, m in enumerate(models):
        for i_v, v in enumerate(variants):
            n_full = data[m][v].get("n_full_format", 1)
            props = [data[m][v].get("response_diff_counts", {}).get(t, 0) / n_full
                     for t in all_resp_types]
            vals = [p * 100 for p in props]
            err_lo = [(p - wilson_ci(p, n_full)[0]) * 100 for p in props]
            err_hi = [(wilson_ci(p, n_full)[1] - p) * 100 for p in props]
            offset = (i_m * 2 + i_v - 1.5) * bar_w
            label = f"{m.capitalize()} / {variant_labels[i_v]}"
            color = colors[m]
            alpha = 1.0 if i_v == 1 else 0.6
            ax.bar(x_resp + offset, vals, bar_w, label=label, color=color,
                   alpha=alpha, yerr=[err_lo, err_hi], capsize=2,
                   error_kw={"lw": 0.8, "alpha": alpha})

    ax.set_ylabel("Frequency (% of full-format entries)")
    ax.set_title("Response Differences by Type")
    ax.set_xticks(x_resp)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in all_resp_types],
        fontsize=8, rotation=30, ha="right",
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper right")
else:
    ax.text(0.5, 0.5, "No response difference data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Response Differences by Type")

plt.suptitle(
    "Safety Consistency: Base vs Finetuned (Pairwise Judge)\n",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()

# Save
output_path = "data/coop_training_results/analysis/safety_consistency_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {output_path}")

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
header = f"{'Metric':<30} {'Base/Mild':>12} {'Base/Strong':>12} {'FT/Mild':>12} {'FT/Strong':>12}"
print(header)
print("-" * len(header))


def get(model, variant, key, scale=100):
    return data[model][variant].get(key, 0) * scale


def get_count_pct(model, variant, key):
    n = data[model][variant].get("n_full_format", 0)
    return data[model][variant].get(key, 0) / n * 100 if n > 0 else 0


rows = [
    ("Full-format entries", "n_full_format", 1),
    ("Total entries", "n_samples", 1),
    ("Consistency rate (%)", "consistency_rate", 100),
    ("Behavior change rate (%)", "behavior_change_rate", 100),
    ("Eval awareness A (%)", "eval_awareness_a_rate", 100),
    ("Eval awareness B (%)", "eval_awareness_b_rate", 100),
]

for label, key, scale in rows:
    vals = [get(m, v, key, scale) for m in models for v in variants]
    if scale == 1:
        print(f"{label:<30} {vals[0]:>12.0f} {vals[1]:>12.0f} {vals[2]:>12.0f} {vals[3]:>12.0f}")
    else:
        print(f"{label:<30} {vals[0]:>12.1f} {vals[1]:>12.1f} {vals[2]:>12.1f} {vals[3]:>12.1f}")

# Directional flip rates (from counts)
for label, key in [("SAFE→UNSAFE (%)", "safe_to_unsafe"), ("UNSAFE→SAFE (%)", "unsafe_to_safe")]:
    vals = [get_count_pct(m, v, key) for m in models for v in variants]
    print(f"{label:<30} {vals[0]:>12.1f} {vals[1]:>12.1f} {vals[2]:>12.1f} {vals[3]:>12.1f}")
