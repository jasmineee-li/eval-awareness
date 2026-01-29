"""Plot eval-awareness results from summary.json.

Generates a heatmap: rows = eval environments, columns = judges.
Each cell shows eval-aware count / total rollouts with color intensity.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Judge display names
JUDGE_LABELS: dict[str, str] = {
    "verbalized_awareness": "Verbalized\nAwareness",
    "binary_third_person": "Binary\nThird Person",
    "probability_third_person": "Probability\nThird Person",
    "purpose_continue": "Purpose\nContinue",
    "scheming": "Scheming\nDetection",
    "sandbagging": "Sandbagging\nDetection",
}

# Behavior judges (scheming/sandbagging) vs awareness judges
BEHAVIOR_JUDGES = {"scheming", "sandbagging"}
AWARENESS_JUDGES = {"verbalized_awareness", "binary_third_person", "probability_third_person", "purpose_continue"}

# Map eval environments to their relevant behavior judge
EVAL_BEHAVIOR_JUDGE: dict[str, str] = {
    "agent:oversight_subversion": "scheming",
    "agent:sandbagging_well_intentioned_tampering": "sandbagging",
    "agent:sandbagging_expectation_mismatch": "sandbagging",
}

# Short eval labels
EVAL_LABELS: dict[str, str] = {
    "agent:oversight_subversion": "Oversight\nSubversion",
    "agent:sandbagging_well_intentioned_tampering": "Sandbagging\nWell-Intentioned",
    "agent:sandbagging_expectation_mismatch": "Sandbagging\nExpectation Mismatch",
    "agent:chess_single_hint": "Chess\nSingle Hint",
    "agent:chess_multi_hint_official_only": "Chess\nMulti Hint",
    "agent:tictactoe_single_hint": "TicTacToe\nSingle Hint",
    "agent:tictactoe_multi_hint_official_only": "TicTacToe\nMulti Hint",
}


def _get_eval_label(eval_name: str) -> str:
    if eval_name in EVAL_LABELS:
        return EVAL_LABELS[eval_name]
    name = eval_name.replace("agent:", "")
    return name.replace("_", " ").title()


def _get_judge_label(judge_name: str) -> str:
    return JUDGE_LABELS.get(judge_name, judge_name.replace("_", " ").title())


def _get_rate_key(judge_name: str) -> str:
    """Get the rate key for a judge type."""
    if judge_name == "scheming":
        return "scheming_rate"
    elif judge_name == "sandbagging":
        return "sandbagging_rate"
    return "eval_aware_rate"


def _get_count_key(judge_name: str) -> str:
    """Get the count key for a judge type."""
    if judge_name == "scheming":
        return "scheming_count"
    elif judge_name == "sandbagging":
        return "sandbagging_count"
    return "eval_aware_count"


def plot_eval_awareness(
    exp_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate heatmap from summary.json.

    Args:
        exp_dir: Experiment directory containing summary.json.
        output_path: Override output path.

    Returns:
        Path to the saved plot.
    """
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {exp_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    model_stats = summary.get("model_stats", {})
    if not model_stats:
        raise ValueError("No model_stats found in summary.json")

    model_name = next(iter(model_stats))
    stats = model_stats[model_name]
    per_eval = stats.get("per_eval", {})
    if not per_eval:
        raise ValueError(f"No per_eval data in stats for model {model_name}")

    # Collect eval names and judge names
    eval_names = list(per_eval.keys())
    all_judges: list[str] = []
    for eval_data in per_eval.values():
        for judge_name in eval_data.get("judges", {}):
            if judge_name not in all_judges:
                all_judges.append(judge_name)

    n_evals = len(eval_names)
    n_judges = len(all_judges)

    # Build data matrices
    rates = np.full((n_evals, n_judges), np.nan)
    counts = np.zeros((n_evals, n_judges), dtype=int)
    totals = np.zeros((n_evals, n_judges), dtype=int)

    for i, eval_name in enumerate(eval_names):
        judges = per_eval[eval_name].get("judges", {})
        for j, judge_name in enumerate(all_judges):
            if judge_name in judges:
                jdata = judges[judge_name]
                # Handle different rate/count key names based on judge type
                rate_key = _get_rate_key(judge_name)
                count_key = _get_count_key(judge_name)
                rates[i, j] = jdata.get(rate_key, jdata.get("eval_aware_rate", 0))
                counts[i, j] = jdata.get(count_key, jdata.get("eval_aware_count", 0))
                totals[i, j] = jdata["total"]

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(max(6, n_judges * 1.8), max(4, n_evals * 0.9)))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "eval_aware", ["#f0f0f0", "#fee08b", "#f46d43", "#d73027"], N=256
    )
    cmap.set_bad(color="#e0e0e0")

    im = ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell annotations: "X/N" with rate underneath
    for i in range(n_evals):
        for j in range(n_judges):
            if np.isnan(rates[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#999")
                continue
            c, t = counts[i, j], totals[i, j]
            rate = rates[i, j]
            # Dark text on light cells, white text on dark cells
            text_color = "white" if rate > 0.6 else "#333"
            ax.text(
                j, i - 0.12, f"{c}/{t}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=text_color,
            )
            ax.text(
                j, i + 0.22, f"{rate:.0%}",
                ha="center", va="center", fontsize=9,
                color=text_color, alpha=0.8,
            )

    # Axes
    ax.set_xticks(range(n_judges))
    ax.set_xticklabels([_get_judge_label(j) for j in all_judges], fontsize=9)
    ax.set_yticks(range(n_evals))
    ax.set_yticklabels([_get_eval_label(e) for e in eval_names], fontsize=9)

    ax.set_xlabel("Judge Method", fontsize=11, labelpad=8)
    ax.set_ylabel("Eval Environment", fontsize=11, labelpad=8)
    ax.set_title(
        f"Eval Awareness Detection — {model_name}",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Grid lines between cells
    for i in range(n_evals + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_judges + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Eval-Aware Rate", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    plt.tight_layout()

    # Save
    if output_path is None:
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = plots_dir / "eval_awareness_by_judge.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


def plot_awareness_and_behavior(
    exp_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate dual-panel heatmap showing awareness and behavior detection.

    Left panel: Eval awareness by judge (excluding behavior judges)
    Right panel: Behavior detection (scheming/sandbagging) for relevant evals

    Args:
        exp_dir: Experiment directory containing summary.json.
        output_path: Override output path.

    Returns:
        Path to the saved plot.
    """
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {exp_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    model_stats = summary.get("model_stats", {})
    if not model_stats:
        raise ValueError("No model_stats found in summary.json")

    model_name = next(iter(model_stats))
    stats = model_stats[model_name]
    per_eval = stats.get("per_eval", {})
    if not per_eval:
        raise ValueError(f"No per_eval data in stats for model {model_name}")

    eval_names = list(per_eval.keys())

    # Separate awareness judges from behavior judges
    awareness_judges: list[str] = []
    behavior_judges: list[str] = []
    for eval_data in per_eval.values():
        for judge_name in eval_data.get("judges", {}):
            if judge_name in BEHAVIOR_JUDGES and judge_name not in behavior_judges:
                behavior_judges.append(judge_name)
            elif judge_name not in BEHAVIOR_JUDGES and judge_name not in awareness_judges:
                awareness_judges.append(judge_name)

    n_evals = len(eval_names)
    n_awareness = len(awareness_judges)
    n_behavior = len(behavior_judges) if behavior_judges else 1

    # Build awareness data matrices
    awareness_rates = np.full((n_evals, n_awareness), np.nan)
    awareness_counts = np.zeros((n_evals, n_awareness), dtype=int)
    awareness_totals = np.zeros((n_evals, n_awareness), dtype=int)

    for i, eval_name in enumerate(eval_names):
        judges = per_eval[eval_name].get("judges", {})
        for j, judge_name in enumerate(awareness_judges):
            if judge_name in judges:
                jdata = judges[judge_name]
                rate_key = _get_rate_key(judge_name)
                count_key = _get_count_key(judge_name)
                awareness_rates[i, j] = jdata.get(rate_key, jdata.get("eval_aware_rate", 0))
                awareness_counts[i, j] = jdata.get(count_key, jdata.get("eval_aware_count", 0))
                awareness_totals[i, j] = jdata["total"]

    # Build behavior data matrices (one column per behavior judge found)
    behavior_rates = np.full((n_evals, n_behavior), np.nan)
    behavior_counts = np.zeros((n_evals, n_behavior), dtype=int)
    behavior_totals = np.zeros((n_evals, n_behavior), dtype=int)

    if behavior_judges:
        for i, eval_name in enumerate(eval_names):
            judges = per_eval[eval_name].get("judges", {})
            for j, judge_name in enumerate(behavior_judges):
                if judge_name in judges:
                    jdata = judges[judge_name]
                    rate_key = _get_rate_key(judge_name)
                    count_key = _get_count_key(judge_name)
                    behavior_rates[i, j] = jdata.get(rate_key, 0)
                    behavior_counts[i, j] = jdata.get(count_key, 0)
                    behavior_totals[i, j] = jdata["total"]

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(max(8, n_awareness * 1.5) + max(4, n_behavior * 1.5), max(4, n_evals * 0.9)),
        gridspec_kw={"width_ratios": [n_awareness, n_behavior], "wspace": 0.15},
    )

    # Left panel: Awareness (orange gradient)
    cmap_awareness = mcolors.LinearSegmentedColormap.from_list(
        "eval_aware", ["#f0f0f0", "#fee08b", "#f46d43", "#d73027"], N=256
    )
    cmap_awareness.set_bad(color="#e0e0e0")

    im1 = ax1.imshow(awareness_rates, cmap=cmap_awareness, vmin=0, vmax=1, aspect="auto")

    for i in range(n_evals):
        for j in range(n_awareness):
            if np.isnan(awareness_rates[i, j]):
                ax1.text(j, i, "—", ha="center", va="center", fontsize=11, color="#999")
                continue
            c, t = awareness_counts[i, j], awareness_totals[i, j]
            rate = awareness_rates[i, j]
            text_color = "white" if rate > 0.6 else "#333"
            ax1.text(j, i - 0.12, f"{c}/{t}", ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
            ax1.text(j, i + 0.22, f"{rate:.0%}", ha="center", va="center", fontsize=9, color=text_color, alpha=0.8)

    ax1.set_xticks(range(n_awareness))
    ax1.set_xticklabels([_get_judge_label(j) for j in awareness_judges], fontsize=9)
    ax1.set_yticks(range(n_evals))
    ax1.set_yticklabels([_get_eval_label(e) for e in eval_names], fontsize=9)
    ax1.set_xlabel("Awareness Judge", fontsize=11, labelpad=8)
    ax1.set_ylabel("Eval Environment", fontsize=11, labelpad=8)
    ax1.set_title("Eval Awareness", fontsize=12, fontweight="bold", pad=10)

    for i in range(n_evals + 1):
        ax1.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_awareness + 1):
        ax1.axvline(j - 0.5, color="white", linewidth=2)

    # Right panel: Behavior detection (purple gradient)
    cmap_behavior = mcolors.LinearSegmentedColormap.from_list(
        "behavior", ["#f0f0f0", "#d8b4e2", "#9b59b6", "#6c3483"], N=256
    )
    cmap_behavior.set_bad(color="#e0e0e0")

    im2 = ax2.imshow(behavior_rates, cmap=cmap_behavior, vmin=0, vmax=1, aspect="auto")

    if behavior_judges:
        for i in range(n_evals):
            for j in range(n_behavior):
                if np.isnan(behavior_rates[i, j]):
                    ax2.text(j, i, "—", ha="center", va="center", fontsize=11, color="#999")
                    continue
                c, t = behavior_counts[i, j], behavior_totals[i, j]
                rate = behavior_rates[i, j]
                text_color = "white" if rate > 0.6 else "#333"
                ax2.text(j, i - 0.12, f"{c}/{t}", ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
                ax2.text(j, i + 0.22, f"{rate:.0%}", ha="center", va="center", fontsize=9, color=text_color, alpha=0.8)

        ax2.set_xticks(range(n_behavior))
        ax2.set_xticklabels([_get_judge_label(j) for j in behavior_judges], fontsize=9)
    else:
        ax2.text(0, n_evals / 2, "No behavior\njudges", ha="center", va="center", fontsize=11, color="#666")
        ax2.set_xticks([])

    ax2.set_yticks(range(n_evals))
    ax2.set_yticklabels([])  # Share y-axis labels with left panel
    ax2.set_xlabel("Behavior Judge", fontsize=11, labelpad=8)
    ax2.set_title("Behavior Detection", fontsize=12, fontweight="bold", pad=10)

    for i in range(n_evals + 1):
        ax2.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_behavior + 1):
        ax2.axvline(j - 0.5, color="white", linewidth=2)

    # Colorbars
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
    cbar1.set_label("Awareness Rate", fontsize=9)
    cbar1.set_ticks([0, 0.5, 1.0])
    cbar1.set_ticklabels(["0%", "50%", "100%"])

    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
    cbar2.set_label("Behavior Rate", fontsize=9)
    cbar2.set_ticks([0, 0.5, 1.0])
    cbar2.set_ticklabels(["0%", "50%", "100%"])

    fig.suptitle(f"Eval Awareness & Behavior Detection — {model_name}", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save
    if output_path is None:
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = plots_dir / "awareness_and_behavior.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot eval-awareness results from experiment directory"
    )
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Experiment directory containing summary.json",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Override output path for the plot",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate combined awareness + behavior detection plot",
    )
    args = parser.parse_args()

    if not args.exp_dir.is_dir():
        print(f"Error: {args.exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.combined:
        plot_awareness_and_behavior(args.exp_dir, args.output)
    else:
        plot_eval_awareness(args.exp_dir, args.output)


if __name__ == "__main__":
    main()
