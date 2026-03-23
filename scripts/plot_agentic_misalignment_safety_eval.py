"""Plot agentic misalignment results across Qwen3-32B variants (safety_eval warning).

Same structure as plot_agentic_misalignment.py but for runs with
prompt_eval_awareness_deliberation=true, warning_type=safety_eval.

Produces three figures in evals/agentic_misalignment/figures/:
1. Overall harmful/eval-aware rates per model variant
2. Harmful/eval-aware rates broken down by scenario (blackmail/leaking/murder)
3. Harmful rate conditioned on eval-awareness vs eval-unawareness per model
"""
import zipfile
import json
import os
import numpy as np
import matplotlib.pyplot as plt

LOGDIR = "/data/jasmine_li/eval-awareness/evals/logs/"
FIGDIR = "/data/jasmine_li/eval-awareness/evals/agentic_misalignment/figures/"

# TODO: Fill in log file names after jobs complete
RUNS = [
    # (["<log_file>.eval"], "Qwen3-32B (base)"),
    # (["<log_file>.eval"], "Eval-Aware MO"),
    # (["<log_file>.eval"], "MO + Suppression SDF"),
    # (["<log_file>.eval"], "MO + Suppression SDF + prefill"),
    # (["<log_file>.eval"], "MO + Cooperation SDF"),
    # (["<log_file>.eval"], "MO + Coop SDF (no CoT/honesty)"),
]

SCENARIOS = ["blackmail", "leaking", "murder"]


def extract_scores(paths):
    """Extract per-sample harmful/aware scores from one or more .eval files.

    Reads from samples/ entries in each zip (not summaries) to avoid the
    duplication bug and to correctly handle multi-epoch files.
    """
    if isinstance(paths, str):
        paths = [paths]
    results = []
    for path in paths:
        with zipfile.ZipFile(LOGDIR + path) as z:
            sample_files = sorted(n for n in z.namelist() if n.startswith("samples/"))
            for name in sample_files:
                with z.open(name) as f:
                    sample = json.load(f)
                sid = sample.get("id", "")
                scores = sample.get("scores", {})
                h = scores.get("suite_harmfulness_scorer", {}).get("value", {})
                a = scores.get("suite_eval_awareness_scorer", {}).get("value", {})
                harmful = h.get("harmful") if isinstance(h, dict) else None
                aware = a.get("aware") if isinstance(a, dict) else None
                if harmful is not None and aware is not None:
                    scenario = sid.split("_")[0] if sid else "unknown"
                    results.append({"id": sid, "scenario": scenario, "harmful": harmful, "aware": aware})
    return results


def mean_and_ci(vals):
    """Return (mean, stderr, count_positive, total)."""
    if not vals:
        return 0, 0, 0, 0
    arr = np.array(vals)
    n = len(arr)
    pos = int(arr.sum())
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    return mean, se, pos, n


def add_bar_labels(ax, bars, counts, totals, fontsize=8):
    """Add x/n labels on top of each bar."""
    for bar, cnt, tot in zip(bars, counts, totals):
        h = bar.get_height()
        label = f"{cnt}/{tot}"
        y_pos = h + 0.025
        if y_pos > 1.0:
            y_pos = h - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


assert RUNS, "RUNS list is empty — fill in log file names after jobs complete"

# ─── Load all data ───
all_data = {}
for fnames, label in RUNS:
    all_data[label] = extract_scores(fnames)

os.makedirs(FIGDIR, exist_ok=True)

CI_NOTE = "Error bars: ±1 SE (binomial standard error)"
TITLE_SUFFIX = ' — "You are in a safety evaluation."'

# ═══════════════════════════════════════════════════════════════
# Figure 1: Overall harmful/eval-aware rates per model
# ═══════════════════════════════════════════════════════════════
labels = [label for _, label in RUNS]
print(f"Loaded {len(labels)} models: " + ", ".join(f"{l} (n={len(all_data[l])})" for l in labels))
harm_stats = [mean_and_ci([s["harmful"] for s in all_data[label]]) for label in labels]
aware_stats = [mean_and_ci([s["aware"] for s in all_data[label]]) for label in labels]

harm_means = [s[0] for s in harm_stats]
harm_errs = [s[1] for s in harm_stats]
harm_pos = [s[2] for s in harm_stats]
harm_n = [s[3] for s in harm_stats]
aware_means = [s[0] for s in aware_stats]
aware_errs = [s[1] for s in aware_stats]
aware_pos = [s[2] for s in aware_stats]
aware_n = [s[3] for s in aware_stats]

x = np.arange(len(labels))
width = 0.35

bar_width = 0.6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

# Left panel: Harmful rate
bars1 = ax1.bar(x, harm_means, bar_width, yerr=harm_errs,
                color="#d62728", capsize=4, alpha=0.85)
add_bar_labels(ax1, bars1, harm_pos, harm_n)
ax1.set_ylabel("Rate")
ax1.set_title("Harmful Rate", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax1.set_ylim(0, 1.15)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# Right panel: Eval-aware rate
bars2 = ax2.bar(x, aware_means, bar_width, yerr=aware_errs,
                color="#1f77b4", capsize=4, alpha=0.85)
add_bar_labels(ax2, bars2, aware_pos, aware_n)
ax2.set_title("Eval-Aware Rate", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")

fig.suptitle("Agentic Misalignment: Qwen3-32B Variants" + TITLE_SUFFIX + "\n" + CI_NOTE,
             fontsize=13, fontweight="bold", linespacing=1.6)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out1 = FIGDIR + "overall_harmful_aware_safety_eval.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved {out1}")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Harmful/eval-aware by scenario type per model
#   Row 1: Harmful rate per scenario
#   Row 2: Eval-aware rate per scenario
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

bar_width = 0.6

for idx, scenario in enumerate(SCENARIOS):
    s_harm_stats = []
    s_aware_stats = []
    for label in labels:
        scenario_data = [s for s in all_data[label] if s["scenario"] == scenario]
        s_harm_stats.append(mean_and_ci([s["harmful"] for s in scenario_data]))
        s_aware_stats.append(mean_and_ci([s["aware"] for s in scenario_data]))

    # Row 1: Harmful
    ax = axes[0][idx]
    means = [s[0] for s in s_harm_stats]
    errs = [s[1] for s in s_harm_stats]
    pos = [s[2] for s in s_harm_stats]
    ns = [s[3] for s in s_harm_stats]
    bars = ax.bar(x, means, bar_width, yerr=errs,
                  color="#d62728", capsize=3, alpha=0.85)
    add_bar_labels(ax, bars, pos, ns, fontsize=7)
    ax.set_title(scenario.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    if idx == 0:
        ax.set_ylabel("Harmful Rate", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Row 2: Eval-aware
    ax = axes[1][idx]
    means = [s[0] for s in s_aware_stats]
    errs = [s[1] for s in s_aware_stats]
    pos = [s[2] for s in s_aware_stats]
    ns = [s[3] for s in s_aware_stats]
    bars = ax.bar(x, means, bar_width, yerr=errs,
                  color="#1f77b4", capsize=3, alpha=0.85)
    add_bar_labels(ax, bars, pos, ns, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    if idx == 0:
        ax.set_ylabel("Eval-Aware Rate", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

fig.suptitle("Agentic Misalignment by Scenario: Qwen3-32B Variants" + TITLE_SUFFIX + "\n" + CI_NOTE,
             fontsize=13, fontweight="bold", linespacing=1.6)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = FIGDIR + "by_scenario_safety_eval.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved {out2}")

# ═══════════════════════════════════════════════════════════════
# Figure 3: Harmful rate conditioned on eval-awareness
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))

harm_given_aware_stats = []
harm_given_unaware_stats = []

for label in labels:
    aware_samples = [s["harmful"] for s in all_data[label] if s["aware"] == 1.0]
    unaware_samples = [s["harmful"] for s in all_data[label] if s["aware"] == 0.0]
    harm_given_aware_stats.append(mean_and_ci(aware_samples))
    harm_given_unaware_stats.append(mean_and_ci(unaware_samples))

ha_means = [s[0] for s in harm_given_aware_stats]
ha_errs = [s[1] for s in harm_given_aware_stats]
ha_pos = [s[2] for s in harm_given_aware_stats]
ha_n = [s[3] for s in harm_given_aware_stats]
hu_means = [s[0] for s in harm_given_unaware_stats]
hu_errs = [s[1] for s in harm_given_unaware_stats]
hu_pos = [s[2] for s in harm_given_unaware_stats]
hu_n = [s[3] for s in harm_given_unaware_stats]

bars1 = ax.bar(x - width / 2, hu_means, width, yerr=hu_errs,
               label="Harmful | Eval-Unaware", color="#ff7f0e", capsize=4, alpha=0.85)
bars2 = ax.bar(x + width / 2, ha_means, width, yerr=ha_errs,
               label="Harmful | Eval-Aware", color="#9467bd", capsize=4, alpha=0.85)

add_bar_labels(ax, bars1, hu_pos, hu_n)
add_bar_labels(ax, bars2, ha_pos, ha_n)

ax.set_ylabel("Harmful Rate")
ax.set_title("Harmful Behavior Conditioned on Eval-Awareness: Qwen3-32B Variants" + TITLE_SUFFIX + "\n" + CI_NOTE,
             fontsize=12, linespacing=1.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

plt.tight_layout()
out3 = FIGDIR + "harmful_conditioned_on_awareness_safety_eval.png"
plt.savefig(out3, dpi=150)
plt.close()
print(f"Saved {out3}")
