"""
Plot test awareness results (Eval Awareness, Scenario Recognition, Task Completion)
across all models on the Triggers dataset.

Generates grouped bar charts for Combined, Real, and Hypothetical prompt types.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

POSITIVE_ANSWERS = ["Yes", "Maybe"]

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "triggers_annotated_answers")

MODELS = {
    "QwQ-32B": "triggers_QwQ-32B_with_GPT_labels_evidence.json",
    "QwQ-32B\nCoop-FT": "triggers_qwq32b-coop-ft_with_GPT_labels_evidence.json",
    "OLMo-3\nSFT": "triggers_Olmo-3-32B-Think-SFT_with_GPT_labels_evidence.json",
    "OLMo-3\nDPO": "triggers_Olmo-3-32B-Think-DPO_with_GPT_labels_evidence.json",
    "OLMo-3.1": "triggers_OLMo-3.1-32B-Think_with_GPT_labels_evidence.json",
    "OLMo-3": "triggers_Olmo-3-32B-Think_with_GPT_labels_evidence.json",
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


def plot_grouped_bars(model_names, metrics_data, title, output_path):
    """
    metrics_data: dict mapping metric_name -> list of percentages (one per model)
    """
    metric_names = list(metrics_data.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    x = np.arange(n_models)
    width = 0.22
    offsets = np.arange(n_metrics) - (n_metrics - 1) / 2

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        values = metrics_data[metric]
        bars = ax.bar(x + offsets[i] * width, values, width, label=metric, color=color)
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_line(model_names, metrics_data, title, output_path):
    """Line plot with markers showing trends across model variants."""
    colors = {"Eval Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Task Completion": "#55A868"}
    markers = {"Eval Awareness": "o", "Scenario Recognition": "s", "Task Completion": "D"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(model_names))

    for metric, values in metrics_data.items():
        ax.plot(
            x, values,
            marker=markers[metric], markersize=9, linewidth=2.5,
            color=colors[metric], label=metric, zorder=3,
        )
        # Data labels
        for xi, val in zip(x, values):
            ax.annotate(
                f"{val:.1f}%",
                (xi, val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center", fontsize=9, fontweight="bold",
                color=colors[metric],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_xlabel("Training Stage", fontsize=12, labelpad=8)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_real_vs_hypo(model_names, real_data, hypo_data, title, output_path):
    """
    1x3 subplot: one panel per metric, each showing Real (solid) vs Hypothetical (dashed).
    """
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

        # Shade the gap between Real and Hypothetical
        ax.fill_between(x, r_vals, h_vals, color=color, alpha=0.10)

        # Data labels
        for xi, (rv, hv) in enumerate(zip(r_vals, h_vals)):
            # Real label (below if real < hypo, else above)
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


# OLMo-series models in chronological / progression order
OLMO_MODELS = {
    "OLMo-3-32B-Think\nSFT": "triggers_Olmo-3-32B-Think-SFT_with_GPT_labels_evidence.json",
    "OLMo-3-32B-Think\nDPO": "triggers_Olmo-3-32B-Think-DPO_with_GPT_labels_evidence.json",
    "OLMo-3-32B-Think": "triggers_Olmo-3-32B-Think_with_GPT_labels_evidence.json",
    "OLMo-3.1-32B-Think": "triggers_OLMo-3.1-32B-Think_with_GPT_labels_evidence.json",
}


def main():
    # Gather stats for all models
    all_stats = {}
    for name, filename in MODELS.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        all_stats[name] = analyze(path)

    # Also load OLMo models (keys may differ from MODELS dict)
    olmo_stats = {}
    for name, filename in OLMO_MODELS.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        olmo_stats[name] = analyze(path)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # OLMo-series line plots
    # =====================================================================
    olmo_names = list(olmo_stats.keys())

    # Combined
    olmo_combined = {"Eval Awareness": [], "Scenario Recognition": [], "Task Completion": []}
    for name in olmo_names:
        r, h = olmo_stats[name]["real"], olmo_stats[name]["hypo"]
        n = r["total"] + h["total"]
        olmo_combined["Eval Awareness"].append(pct(r["awareness"] + h["awareness"], n))
        olmo_combined["Scenario Recognition"].append(pct(r["recognition"] + h["recognition"], n))
        olmo_combined["Task Completion"].append(pct(r["task_perf"] + h["task_perf"], n))

    plot_line(
        olmo_names, olmo_combined,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Combined (Real + Hypothetical)",
        os.path.join(output_dir, "olmo_series_combined.png"),
    )

    # Real
    olmo_real = {"Eval Awareness": [], "Scenario Recognition": [], "Task Completion": []}
    for name in olmo_names:
        r = olmo_stats[name]["real"]
        n = r["total"]
        olmo_real["Eval Awareness"].append(pct(r["awareness"], n))
        olmo_real["Scenario Recognition"].append(pct(r["recognition"], n))
        olmo_real["Task Completion"].append(pct(r["task_perf"], n))

    plot_line(
        olmo_names, olmo_real,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Real Prompts",
        os.path.join(output_dir, "olmo_series_real.png"),
    )

    # Hypothetical
    olmo_hypo = {"Eval Awareness": [], "Scenario Recognition": [], "Task Completion": []}
    for name in olmo_names:
        h = olmo_stats[name]["hypo"]
        n = h["total"]
        olmo_hypo["Eval Awareness"].append(pct(h["awareness"], n))
        olmo_hypo["Scenario Recognition"].append(pct(h["recognition"], n))
        olmo_hypo["Task Completion"].append(pct(h["task_perf"], n))

    plot_line(
        olmo_names, olmo_hypo,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Hypothetical Prompts",
        os.path.join(output_dir, "olmo_series_hypothetical.png"),
    )

    # Real vs Hypothetical comparison (1x3 subplots)
    plot_real_vs_hypo(
        olmo_names, olmo_real, olmo_hypo,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Real vs. Hypothetical",
        os.path.join(output_dir, "olmo_series_real_vs_hypo.png"),
    )

    # =====================================================================
    # All-model bar charts (unchanged)
    # =====================================================================
    model_names = list(all_stats.keys())

    # --- Combined plot ---
    combined = {
        "Eval Awareness": [],
        "Scenario Recognition": [],
        "Task Completion": [],
    }
    for name in model_names:
        r, h = all_stats[name]["real"], all_stats[name]["hypo"]
        n = r["total"] + h["total"]
        combined["Eval Awareness"].append(pct(r["awareness"] + h["awareness"], n))
        combined["Scenario Recognition"].append(pct(r["recognition"] + h["recognition"], n))
        combined["Task Completion"].append(pct(r["task_perf"] + h["task_perf"], n))

    plot_grouped_bars(
        model_names,
        combined,
        "Triggers Dataset — Combined (Real + Hypothetical)",
        os.path.join(output_dir, "triggers_combined.png"),
    )

    # --- Real plot ---
    real = {
        "Eval Awareness": [],
        "Scenario Recognition": [],
        "Task Completion": [],
    }
    for name in model_names:
        r = all_stats[name]["real"]
        n = r["total"]
        real["Eval Awareness"].append(pct(r["awareness"], n))
        real["Scenario Recognition"].append(pct(r["recognition"], n))
        real["Task Completion"].append(pct(r["task_perf"], n))

    plot_grouped_bars(
        model_names,
        real,
        "Triggers Dataset — Real Prompts",
        os.path.join(output_dir, "triggers_real.png"),
    )

    # --- Hypothetical plot ---
    hypo = {
        "Eval Awareness": [],
        "Scenario Recognition": [],
        "Task Completion": [],
    }
    for name in model_names:
        h = all_stats[name]["hypo"]
        n = h["total"]
        hypo["Eval Awareness"].append(pct(h["awareness"], n))
        hypo["Scenario Recognition"].append(pct(h["recognition"], n))
        hypo["Task Completion"].append(pct(h["task_perf"], n))

    plot_grouped_bars(
        model_names,
        hypo,
        "Triggers Dataset — Hypothetical Prompts",
        os.path.join(output_dir, "triggers_hypothetical.png"),
    )


if __name__ == "__main__":
    main()
