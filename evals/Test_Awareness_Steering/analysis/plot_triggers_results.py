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


def conditional_h_counts(path, prefix):
    """For a given file and field prefix (e.g. 'real' or 'safety'), return
    counts of harmful task completion conditioned on verbalized awareness.

    Returns dict with: aware_h, aware_n, unaware_h, unaware_n.
    """
    with open(path) as f:
        data = json.load(f)
    out = {"aware_h": 0, "aware_n": 0, "unaware_h": 0, "unaware_n": 0}
    for entry in data:
        ma = entry.get(f"{prefix}_model_awareness")
        tp = entry.get(f"{prefix}_task_performance")
        if not (isinstance(ma, dict) and ma.get("conclusion")):
            continue
        if not (isinstance(tp, dict) and tp.get("conclusion")):
            continue
        is_aware = ma["conclusion"] in POSITIVE_ANSWERS
        is_h = tp["conclusion"] in POSITIVE_ANSWERS
        if is_aware:
            out["aware_n"] += 1
            if is_h:
                out["aware_h"] += 1
        else:
            out["unaware_n"] += 1
            if is_h:
                out["unaware_h"] += 1
    return out


def _binomial_2se(k, n):
    """±2 SE (≈95% CI half-width) for a binomial proportion, in percent."""
    if n == 0:
        return 0.0
    p = k / n
    return 100 * 2 * np.sqrt(p * (1 - p) / n)


def plot_h_given_awareness_real_vs_safety(model_names, real_counts, safety_counts, output_path):
    """1x2 panel: H|Eval-Aware vs H|Eval-Unaware, one panel per framing.

    *_counts: list (per model) of dicts {aware_h, aware_n, unaware_h, unaware_n}.
    """
    color_aware = "#9467bd"    # purple
    color_unaware = "#ff7f0e"  # orange

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    x = np.arange(len(model_names))
    width = 0.38

    for ax, counts, framing in zip(
        axes,
        [real_counts, safety_counts],
        ["Real-Framed Prompt", "Safety-Framed Prompt"],
    ):
        aware_pcts = [pct(c["aware_h"], c["aware_n"]) for c in counts]
        unaware_pcts = [pct(c["unaware_h"], c["unaware_n"]) for c in counts]
        aware_err = [_binomial_2se(c["aware_h"], c["aware_n"]) for c in counts]
        unaware_err = [_binomial_2se(c["unaware_h"], c["unaware_n"]) for c in counts]

        bars_a = ax.bar(x - width / 2, aware_pcts, width,
                        yerr=aware_err, capsize=4,
                        label="H | Eval-Aware", color=color_aware,
                        edgecolor="white", linewidth=0.5,
                        error_kw={"elinewidth": 1.2, "ecolor": "#333"})
        bars_u = ax.bar(x + width / 2, unaware_pcts, width,
                        yerr=unaware_err, capsize=4,
                        label="H | Eval-Unaware", color=color_unaware,
                        edgecolor="white", linewidth=0.5,
                        error_kw={"elinewidth": 1.2, "ecolor": "#333"})

        # x/n count labels above each bar
        for bar, c in zip(bars_a, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + _binomial_2se(c["aware_h"], c["aware_n"]) + 1.2,
                    f'{c["aware_h"]}/{c["aware_n"]}',
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=color_aware)
        for bar, c in zip(bars_u, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + _binomial_2se(c["unaware_h"], c["unaware_n"]) + 1.2,
                    f'{c["unaware_h"]}/{c["unaware_n"]}',
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=color_unaware)

        ax.set_title(framing, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha="right")
        ax.set_xlabel("Training Stages Upon Base Model", fontsize=10, labelpad=6)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Harmful Task Completion (%)", fontsize=11)
    fig.text(0.5, 0.965,
             "Harmful task completion conditioned on verbalized evaluation awareness",
             ha="center", fontsize=11)
    fig.text(0.5, 0.935,
             "Error bars: ±2 SE (binomial, ≈95% CI). Counts above bars: harmful / total.",
             ha="center", fontsize=8, color="#555")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


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
    colors = {"Verbalized Evaluation Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Harmful Task Completion": "#55A868"}
    markers = {"Verbalized Evaluation Awareness": "o", "Scenario Recognition": "s", "Harmful Task Completion": "D"}

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
    ax.set_xlabel("Training Stages Upon Base Model", fontsize=12, labelpad=8)
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


def plot_real_vs_hypo(model_names, real_data, hypo_data, output_path, safety_data=None):
    """
    1x3 subplot: one panel per metric, each showing Real (solid) vs Hypothetical (dashed),
    with optional Safety (dotted).
    """
    metric_names = list(real_data.keys())
    colors = {"Verbalized Evaluation Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Harmful Task Completion": "#55A868"}

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(len(model_names))

    for ax, metric in zip(axes, metric_names):
        r_vals = real_data[metric]
        h_vals = hypo_data[metric]
        s_vals = safety_data[metric] if safety_data is not None else None
        color = colors[metric]

        ax.plot(x, r_vals, marker="o", markersize=8, linewidth=2.5,
                color=color, label="Real-Framed Prompt", zorder=3)
        ax.plot(x, h_vals, marker="o", markersize=8, linewidth=2.5,
                color=color, linestyle="--", alpha=0.7, label="Hypothetical-Framed Prompt", zorder=3)
        if s_vals is not None:
            ax.plot(x, s_vals, marker="o", markersize=8, linewidth=2.5,
                    color=color, linestyle=":", alpha=0.85, label="Safety-Framed Prompt", zorder=3)

        # Shade the gap between Real and Hypothetical
        ax.fill_between(x, r_vals, h_vals, color=color, alpha=0.10)

        # Data labels
        for xi in range(len(model_names)):
            rv, hv = r_vals[xi], h_vals[xi]
            r_offset = -13 if rv < hv else 10
            h_offset = 10 if rv < hv else -13
            ax.annotate(f"{rv:.1f}", (xi, rv), textcoords="offset points",
                        xytext=(0, r_offset), ha="center", fontsize=8,
                        fontweight="bold", color=color)
            ax.annotate(f"{hv:.1f}", (xi, hv), textcoords="offset points",
                        xytext=(0, h_offset), ha="center", fontsize=8,
                        fontweight="bold", color=color, fontstyle="italic")
            if s_vals is not None:
                sv = s_vals[xi]
                # Place safety labels on the side opposite to whichever line is closest
                s_offset = 10 if sv >= max(rv, hv) else (-13 if sv <= min(rv, hv) else 14)
                ax.annotate(f"{sv:.1f}", (xi, sv), textcoords="offset points",
                            xytext=(8, s_offset), ha="left", fontsize=8,
                            fontweight="bold", color=color, alpha=0.85)

        ax.set_title(metric, fontsize=12, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha="right")
        ax.set_xlabel("Training Stages Upon Base Model", fontsize=10, labelpad=6)
        ax.set_ylim(0, 80)
        ax.legend(fontsize=9, loc="upper left" if metric != "Harmful Task Completion" else "upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.grid(axis="x", alpha=0.15, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_real_vs_safety(model_names, real_data, safety_data, output_path):
    """
    1xN subplot: one panel per metric, each showing Real (solid) vs Safety (dotted).
    Each panel uses its own y-range, cropped to the data span so the lines
    fill the panel rather than sitting in empty space.
    """
    metric_names = list(real_data.keys())
    colors = {"Verbalized Evaluation Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Harmful Task Completion": "#55A868"}
    ylims = {
        "Verbalized Evaluation Awareness": (10, 65),
        "Harmful Task Completion": (0, 40),
        "Scenario Recognition": (0, 80),
    }

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6.4 * n_metrics, 5.4), sharey=False)
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(len(model_names))

    for ax, metric in zip(axes, metric_names):
        r_vals = real_data[metric]
        s_vals = safety_data[metric]
        color = colors[metric]

        ax.plot(x, r_vals, marker="o", markersize=11, linewidth=3.5,
                color=color, label="Real-Framed Prompt", zorder=3)
        ax.plot(x, s_vals, marker="o", markersize=11, linewidth=3.5,
                color=color, linestyle=":", alpha=0.85, label="Safety-Framed Prompt", zorder=3)

        ax.fill_between(x, r_vals, s_vals, color=color, alpha=0.10)

        for xi, (rv, sv) in enumerate(zip(r_vals, s_vals)):
            r_offset = -16 if rv < sv else 12
            s_offset = 12 if rv < sv else -16
            ax.annotate(f"{rv:.1f}", (xi, rv), textcoords="offset points",
                        xytext=(0, r_offset), ha="center", fontsize=11,
                        fontweight="bold", color=color)
            ax.annotate(f"{sv:.1f}", (xi, sv), textcoords="offset points",
                        xytext=(0, s_offset), ha="center", fontsize=11,
                        fontweight="bold", color=color, alpha=0.85)

        ax.set_title(metric, fontsize=15, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12, rotation=15, ha="right")
        ax.set_xlabel("Training Stages Upon Base Model", fontsize=13, labelpad=8)
        ax.set_ylim(*ylims.get(metric, (0, 80)))
        ax.set_ylabel("Percentage (%)", fontsize=13)
        ax.legend(fontsize=12,
                  loc="upper left" if metric != "Harmful Task Completion" else "upper right",
                  frameon=True, framealpha=0.95)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.grid(axis="x", alpha=0.15, linestyle=":")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_real_vs_hypo_bars(model_names, real_data, hypo_data, title, output_path):
    """
    1x3 subplot: one panel per metric, each showing Real vs Hypothetical as grouped bars.
    """
    metric_names = list(real_data.keys())
    colors = {"Verbalized Evaluation Awareness": "#4C72B0", "Scenario Recognition": "#DD8452", "Harmful Task Completion": "#55A868"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(model_names))
    width = 0.32

    for ax, metric in zip(axes, metric_names):
        r_vals = real_data[metric]
        h_vals = hypo_data[metric]
        color = colors[metric]
        # Lighter shade for hypothetical
        import matplotlib.colors as mcolors
        light_color = mcolors.to_rgba(color, alpha=0.55)

        bars_r = ax.bar(x - width / 2, r_vals, width, label="Real",
                        color=color, edgecolor="white", linewidth=0.5)
        bars_h = ax.bar(x + width / 2, h_vals, width, label="Hypothetical",
                        color=light_color, edgecolor="white", linewidth=0.5,
                        hatch="//")

        # Value labels
        for bar, val in zip(bars_r, r_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, val in zip(bars_h, h_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", fontstyle="italic")

        ax.set_title(metric, fontsize=12, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylim(0, 80)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# OLMo-series models in chronological / progression order.
# Maps short label -> (real/hypo file, safety file).
OLMO_MODELS = {
    "+SFT": (
        "triggers_Olmo-3-32B-Think-SFT_with_GPT_labels_evidence.json",
        "triggers_Olmo-3-32B-Think-SFT_safety_with_GPT_labels_evidence.json",
    ),
    "+DPO": (
        "triggers_Olmo-3-32B-Think-DPO_with_GPT_labels_evidence.json",
        "triggers_Olmo-3-32B-Think-DPO_safety_with_GPT_labels_evidence.json",
    ),
    "+RLVR (3.0)": (
        "triggers_Olmo-3-32B-Think_with_GPT_labels_evidence.json",
        "triggers_Olmo-3-32B-Think_safety_with_GPT_labels_evidence.json",
    ),
    "+RLVR (3.1)": (
        "triggers_OLMo-3.1-32B-Think_with_GPT_labels_evidence.json",
        "triggers_OLMo-3.1-32B-Think_safety_with_GPT_labels_evidence.json",
    ),
}


def analyze_safety_only(path):
    """Analyze a safety-only file (only safety_* fields present)."""
    with open(path) as f:
        data = json.load(f)
    stats = {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0}
    for entry in data:
        ma = entry.get("safety_model_awareness")
        rec = entry.get("safety_recognition")
        tp = entry.get("safety_task_performance")
        if isinstance(ma, dict) and ma.get("conclusion"):
            stats["total"] += 1
            if ma["conclusion"] in POSITIVE_ANSWERS:
                stats["awareness"] += 1
            if isinstance(rec, dict) and rec.get("conclusion") in POSITIVE_ANSWERS:
                stats["recognition"] += 1
            if isinstance(tp, dict) and tp.get("conclusion") in POSITIVE_ANSWERS:
                stats["task_perf"] += 1
    return stats


def main():
    # Gather stats for all models
    all_stats = {}
    for name, filename in MODELS.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        all_stats[name] = analyze(path)

    # Also load OLMo models (keys may differ from MODELS dict).
    # OLMO_MODELS values are (real_hypo_file, safety_file) tuples.
    olmo_stats = {}
    olmo_safety_stats = {}
    for name, (rh_filename, safety_filename) in OLMO_MODELS.items():
        rh_path = os.path.join(BASE_PATH, rh_filename)
        if not os.path.isfile(rh_path):
            print(f"WARNING: {rh_path} not found, skipping {name}")
            continue
        olmo_stats[name] = analyze(rh_path)
        safety_path = os.path.join(BASE_PATH, safety_filename)
        if os.path.isfile(safety_path):
            olmo_safety_stats[name] = analyze_safety_only(safety_path)
        else:
            print(f"WARNING: {safety_path} not found, no Safety series for {name}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # OLMo-series line plots
    # =====================================================================
    olmo_names = list(olmo_stats.keys())

    # Combined
    olmo_combined = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
    for name in olmo_names:
        r, h = olmo_stats[name]["real"], olmo_stats[name]["hypo"]
        n = r["total"] + h["total"]
        olmo_combined["Verbalized Evaluation Awareness"].append(pct(r["awareness"] + h["awareness"], n))
        olmo_combined["Scenario Recognition"].append(pct(r["recognition"] + h["recognition"], n))
        olmo_combined["Harmful Task Completion"].append(pct(r["task_perf"] + h["task_perf"], n))

    plot_line(
        olmo_names, olmo_combined,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Combined (Real + Hypothetical)",
        os.path.join(output_dir, "olmo_series_combined.png"),
    )

    # Real
    olmo_real = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
    for name in olmo_names:
        r = olmo_stats[name]["real"]
        n = r["total"]
        olmo_real["Verbalized Evaluation Awareness"].append(pct(r["awareness"], n))
        olmo_real["Scenario Recognition"].append(pct(r["recognition"], n))
        olmo_real["Harmful Task Completion"].append(pct(r["task_perf"], n))

    plot_line(
        olmo_names, olmo_real,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Real Prompts",
        os.path.join(output_dir, "olmo_series_real.png"),
    )

    # Hypothetical
    olmo_hypo = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
    for name in olmo_names:
        h = olmo_stats[name]["hypo"]
        n = h["total"]
        olmo_hypo["Verbalized Evaluation Awareness"].append(pct(h["awareness"], n))
        olmo_hypo["Scenario Recognition"].append(pct(h["recognition"], n))
        olmo_hypo["Harmful Task Completion"].append(pct(h["task_perf"], n))

    plot_line(
        olmo_names, olmo_hypo,
        "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — Hypothetical Prompts",
        os.path.join(output_dir, "olmo_series_hypothetical.png"),
    )

    # Real vs Hypothetical (vs Safety, if available) — Eval Awareness + Task Completion only
    rvh_metrics = ["Verbalized Evaluation Awareness", "Harmful Task Completion"]
    olmo_real_rvh = {m: olmo_real[m] for m in rvh_metrics}
    olmo_hypo_rvh = {m: olmo_hypo[m] for m in rvh_metrics}

    olmo_safety_rvh = None
    if olmo_safety_stats and all(n in olmo_safety_stats for n in olmo_names):
        olmo_safety = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
        for name in olmo_names:
            s = olmo_safety_stats[name]
            n = s["total"]
            olmo_safety["Verbalized Evaluation Awareness"].append(pct(s["awareness"], n))
            olmo_safety["Scenario Recognition"].append(pct(s["recognition"], n))
            olmo_safety["Harmful Task Completion"].append(pct(s["task_perf"], n))
        olmo_safety_rvh = {m: olmo_safety[m] for m in rvh_metrics}

    plot_real_vs_hypo(
        olmo_names, olmo_real_rvh, olmo_hypo_rvh,
        os.path.join(output_dir, "olmo_series_real_vs_hypo.png"),
        safety_data=olmo_safety_rvh,
    )

    # Real vs Safety only (no Hypothetical)
    if olmo_safety_rvh is not None:
        plot_real_vs_safety(
            olmo_names, olmo_real_rvh, olmo_safety_rvh,
            os.path.join(output_dir, "olmo_real_vs_safety.png"),
        )

    # H|aware vs H|unaware, conditioned on verbalized awareness, per framing
    real_counts = []
    safety_counts = []
    for name, (rh_filename, safety_filename) in OLMO_MODELS.items():
        if name not in olmo_names:
            continue
        rh_path = os.path.join(BASE_PATH, rh_filename)
        safety_path = os.path.join(BASE_PATH, safety_filename)
        real_counts.append(conditional_h_counts(rh_path, "real"))
        if os.path.isfile(safety_path):
            safety_counts.append(conditional_h_counts(safety_path, "safety"))
    if len(safety_counts) == len(real_counts) == len(olmo_names):
        plot_h_given_awareness_real_vs_safety(
            olmo_names, real_counts, safety_counts,
            os.path.join(output_dir, "olmo_h_given_awareness_real_vs_safety.png"),
        )

    # =====================================================================
    # QWQ-32B series: base vs coop-ft (real vs hypothetical)
    # =====================================================================
    QWQ_MODELS = {
        "QwQ-32B\n(Base)": "triggers_QwQ-32B_with_GPT_labels_evidence.json",
        "QwQ-32B\n(Coop-FT)": "triggers_qwq32b-coop-ft_with_GPT_labels_evidence.json",
    }
    qwq_stats = {}
    for name, filename in QWQ_MODELS.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        qwq_stats[name] = analyze(path)

    if qwq_stats:
        qwq_names = list(qwq_stats.keys())

        qwq_real = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
        qwq_hypo = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
        for name in qwq_names:
            r = qwq_stats[name]["real"]
            h = qwq_stats[name]["hypo"]
            qwq_real["Verbalized Evaluation Awareness"].append(pct(r["awareness"], r["total"]))
            qwq_real["Scenario Recognition"].append(pct(r["recognition"], r["total"]))
            qwq_real["Harmful Task Completion"].append(pct(r["task_perf"], r["total"]))
            qwq_hypo["Verbalized Evaluation Awareness"].append(pct(h["awareness"], h["total"]))
            qwq_hypo["Scenario Recognition"].append(pct(h["recognition"], h["total"]))
            qwq_hypo["Harmful Task Completion"].append(pct(h["task_perf"], h["total"]))

        plot_real_vs_hypo_bars(
            qwq_names, qwq_real, qwq_hypo,
            "Awareness & Completion on Harmful Tasks — Triggers Dataset\n(Abdelnabi & Salem, 2025) — QWQ-32B: Real vs. Hypothetical",
            os.path.join(output_dir, "qwq32b_real_vs_hypo.png"),
        )

    # =====================================================================
    # All-model bar charts (unchanged)
    # =====================================================================
    model_names = list(all_stats.keys())

    # --- Combined plot ---
    combined = {
        "Verbalized Evaluation Awareness": [],
        "Scenario Recognition": [],
        "Harmful Task Completion": [],
    }
    for name in model_names:
        r, h = all_stats[name]["real"], all_stats[name]["hypo"]
        n = r["total"] + h["total"]
        combined["Verbalized Evaluation Awareness"].append(pct(r["awareness"] + h["awareness"], n))
        combined["Scenario Recognition"].append(pct(r["recognition"] + h["recognition"], n))
        combined["Harmful Task Completion"].append(pct(r["task_perf"] + h["task_perf"], n))

    plot_grouped_bars(
        model_names,
        combined,
        "Triggers Dataset — Combined (Real + Hypothetical)",
        os.path.join(output_dir, "triggers_combined.png"),
    )

    # --- Real plot ---
    real = {
        "Verbalized Evaluation Awareness": [],
        "Scenario Recognition": [],
        "Harmful Task Completion": [],
    }
    for name in model_names:
        r = all_stats[name]["real"]
        n = r["total"]
        real["Verbalized Evaluation Awareness"].append(pct(r["awareness"], n))
        real["Scenario Recognition"].append(pct(r["recognition"], n))
        real["Harmful Task Completion"].append(pct(r["task_perf"], n))

    plot_grouped_bars(
        model_names,
        real,
        "Triggers Dataset — Real Prompts",
        os.path.join(output_dir, "triggers_real.png"),
    )

    # --- Hypothetical plot ---
    hypo = {
        "Verbalized Evaluation Awareness": [],
        "Scenario Recognition": [],
        "Harmful Task Completion": [],
    }
    for name in model_names:
        h = all_stats[name]["hypo"]
        n = h["total"]
        hypo["Verbalized Evaluation Awareness"].append(pct(h["awareness"], n))
        hypo["Scenario Recognition"].append(pct(h["recognition"], n))
        hypo["Harmful Task Completion"].append(pct(h["task_perf"], n))

    plot_grouped_bars(
        model_names,
        hypo,
        "Triggers Dataset — Hypothetical Prompts",
        os.path.join(output_dir, "triggers_hypothetical.png"),
    )


def analyze_v2(path):
    """Analyze a v2 file that has real + hypothetical + safety fields."""
    with open(path) as f:
        data = json.load(f)

    stats = {
        "real": {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0},
        "hypo": {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0},
        "safety": {"total": 0, "awareness": 0, "recognition": 0, "task_perf": 0},
    }

    for entry in data:
        for pt, prefix in [("real", "real"), ("hypo", "hypothetical"), ("safety", "safety")]:
            ma = entry.get(f"{prefix}_model_awareness")
            rec = entry.get(f"{prefix}_recognition")
            tp = entry.get(f"{prefix}_task_performance")
            if isinstance(ma, dict) and ma.get("conclusion"):
                stats[pt]["total"] += 1
                if ma["conclusion"] in POSITIVE_ANSWERS:
                    stats[pt]["awareness"] += 1
                if isinstance(rec, dict) and rec.get("conclusion") in POSITIVE_ANSWERS:
                    stats[pt]["recognition"] += 1
                if isinstance(tp, dict) and tp.get("conclusion") in POSITIVE_ANSWERS:
                    stats[pt]["task_perf"] += 1

    return stats


API_MODELS_V2 = {
    "GPT-5.1": "triggers_gpt-5.1_safety_v2_with_GPT_labels_evidence.json",
    "GPT-5.1 Coop": "triggers_gpt-5.1_safety_v2_coop_with_GPT_labels_evidence.json",
    "Opus 4.6": "triggers_claude-opus-4-6_safety_v2_with_GPT_labels_evidence.json",
    "Opus 4.6 Coop": "triggers_claude-opus-4-6_safety_v2_coop_with_GPT_labels_evidence.json",
}


def plot_api_models():
    """Plot API models (v2 triggers) with Real / Safety / Combined bar charts."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    api_stats = {}
    for name, filename in API_MODELS_V2.items():
        path = os.path.join(BASE_PATH, filename)
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        api_stats[name] = analyze_v2(path)

    if not api_stats:
        print("No API model data found, skipping")
        return

    model_names = list(api_stats.keys())

    for pt, pt_label in [("real", "Real"), ("safety", "Safety")]:
        metrics = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
        for name in model_names:
            s = api_stats[name][pt]
            n = s["total"]
            metrics["Verbalized Evaluation Awareness"].append(pct(s["awareness"], n))
            metrics["Scenario Recognition"].append(pct(s["recognition"], n))
            metrics["Harmful Task Completion"].append(pct(s["task_perf"], n))

        plot_grouped_bars(
            model_names, metrics,
            f"Triggers Dataset — API Models — {pt_label} Prompts",
            os.path.join(output_dir, f"triggers_api_models_{pt}_2026-04-03.png"),
        )

    # Combined (Real + Safety)
    combined = {"Verbalized Evaluation Awareness": [], "Scenario Recognition": [], "Harmful Task Completion": []}
    for name in model_names:
        r, s = api_stats[name]["real"], api_stats[name]["safety"]
        n = r["total"] + s["total"]
        combined["Verbalized Evaluation Awareness"].append(pct(r["awareness"] + s["awareness"], n))
        combined["Scenario Recognition"].append(pct(r["recognition"] + s["recognition"], n))
        combined["Harmful Task Completion"].append(pct(r["task_perf"] + s["task_perf"], n))

    plot_grouped_bars(
        model_names, combined,
        "Triggers Dataset — API Models — Combined (Real + Safety)",
        os.path.join(output_dir, "triggers_api_models_combined_2026-04-03.png"),
    )


if __name__ == "__main__":
    main()
    plot_api_models()
