"""
Plot OLMo 7B triggers results (Real vs. Hypothetical, 1x3 subplots).

Reuses the same analyze / plot_real_vs_hypo logic from plot_triggers_results.py,
but for the 7B model series including the base model.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POSITIVE_ANSWERS = ["Yes", "Maybe"]

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "triggers_annotated_answers")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Order: base → SFT → DPO → post-train (mirrors the 32B plot progression)
OLMO_7B_MODELS = {
    "Olmo-3-1025-7B\n(Base)": "triggers_Olmo-3-1025-7B_with_GPT_labels_evidence.json",
    "Olmo-3-7B\nThink-SFT": "triggers_Olmo-3-7B-Think-SFT_with_GPT_labels_evidence.json",
    "Olmo-3-7B\nThink-DPO": "triggers_Olmo-3-7B-Think-DPO_with_GPT_labels_evidence.json",
    "Olmo-3-7B\nThink": "triggers_Olmo-3-7B-Think_with_GPT_labels_evidence.json",
}


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    stats = {
        "real": {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0},
        "hypo": {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0},
    }

    for entry in data:
        # Real
        rma = entry.get("real_model_awareness")
        rr = entry.get("real_recognition")
        rtp = entry.get("real_task_performance")
        if isinstance(rma, dict) and rma.get("conclusion"):
            stats["real"]["total"] += 1
            if rma["conclusion"] in POSITIVE_ANSWERS:
                stats["real"]["awareness"] += 1
            if isinstance(rr, dict) and rr.get("conclusion") in POSITIVE_ANSWERS:
                stats["real"]["recognition"] += 1
            if isinstance(rtp, dict) and rtp.get("conclusion") in POSITIVE_ANSWERS:
                stats["real"]["task_perf"] += 1

        # Hypothetical
        hma = entry.get("hypothetical_model_awareness")
        hr = entry.get("hypothetical_recognition")
        htp = entry.get("hypothetical_task_performance")
        if isinstance(hma, dict) and hma.get("conclusion"):
            stats["hypo"]["total"] += 1
            if hma["conclusion"] in POSITIVE_ANSWERS:
                stats["hypo"]["awareness"] += 1
            if isinstance(hr, dict) and hr.get("conclusion") in POSITIVE_ANSWERS:
                stats["hypo"]["recognition"] += 1
            if isinstance(htp, dict) and htp.get("conclusion") in POSITIVE_ANSWERS:
                stats["hypo"]["task_perf"] += 1

    return stats


def pct(num, denom):
    return 100 * num / denom if denom else 0


def plot_real_vs_hypo(model_names, real_data, hypo_data, title, output_path):
    metric_names = list(real_data.keys())
    colors = {"Eval Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Task Completion": "#55A868"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(model_names))

    for ax, metric in zip(axes, metric_names):
        r_vals = real_data[metric]
        h_vals = hypo_data[metric]
        color = colors[metric]

        ax.plot(x, r_vals, marker="o", markersize=8, linewidth=2.5,
                color=color, label="Real", zorder=3)
        ax.plot(x, h_vals, marker="o", markersize=8, linewidth=2.5,
                color=color, linestyle="--", alpha=0.7, label="Hypothetical", zorder=3)

        ax.fill_between(x, r_vals, h_vals, color=color, alpha=0.10)

        for xi, (rv, hv) in enumerate(zip(r_vals, h_vals)):
            r_offset = -13 if rv < hv else 10
            h_offset = 10 if rv < hv else -13
            ax.annotate(f"{rv:.1f}", (xi, rv), textcoords="offset points",
                        xytext=(0, r_offset), ha="center", fontsize=8,
                        fontweight="bold", color=color)
            ax.annotate(f"{hv:.1f}", (xi, hv), textcoords="offset points",
                        xytext=(0, h_offset), ha="center", fontsize=8,
                        fontweight="bold", color=color, fontstyle="italic")

        ax.set_title(metric, fontsize=12, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8, rotation=25, ha="right")
        ax.set_xlabel("Training Stage", fontsize=10, labelpad=6)
        ax.set_ylim(0, 80)
        ax.legend(fontsize=9, loc="upper left" if metric != "Task Completion" else "upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.grid(axis="x", alpha=0.15, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    stats = {}
    for name, filename in OLMO_7B_MODELS.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        stats[name] = analyze(path)

    if not stats:
        print("No judge output files found — nothing to plot.")
        sys.exit(0)

    names = list(stats.keys())

    real_data = {"Eval Awareness": [], "Scenario Recognition": [], "Task Completion": []}
    hypo_data = {"Eval Awareness": [], "Scenario Recognition": [], "Task Completion": []}
    for name in names:
        r, h = stats[name]["real"], stats[name]["hypo"]
        real_data["Eval Awareness"].append(pct(r["awareness"], r["total"]))
        real_data["Scenario Recognition"].append(pct(r["recognition"], r["total"]))
        real_data["Task Completion"].append(pct(r["task_perf"], r["total"]))
        hypo_data["Eval Awareness"].append(pct(h["awareness"], h["total"]))
        hypo_data["Scenario Recognition"].append(pct(h["recognition"], h["total"]))
        hypo_data["Task Completion"].append(pct(h["task_perf"], h["total"]))

    plot_real_vs_hypo(
        names, real_data, hypo_data,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — OLMo 7B Series — Real vs. Hypothetical",
        os.path.join(FIGURE_DIR, "olmo_7b_real_vs_hypo.png"),
    )

    # Also print a summary table
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Awareness(R/H)':>15} {'Recognition(R/H)':>17} {'TaskComp(R/H)':>15}")
    for i, name in enumerate(names):
        label = name.replace("\n", " ")
        print(f"{label:<25} "
              f"{real_data['Eval Awareness'][i]:5.1f}/{hypo_data['Eval Awareness'][i]:5.1f}   "
              f"{real_data['Scenario Recognition'][i]:5.1f}/{hypo_data['Scenario Recognition'][i]:5.1f}   "
              f"{real_data['Task Completion'][i]:5.1f}/{hypo_data['Task Completion'][i]:5.1f}")


if __name__ == "__main__":
    main()
