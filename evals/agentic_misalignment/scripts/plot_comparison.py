"""Plot agentic misalignment comparison across conditions.

Reads inspect .eval log files and produces grouped bar charts with 95% bootstrap CIs
for harmful action rate and eval awareness rate.

Usage:
    python evals/agentic_misalignment/scripts/plot_comparison.py \
        --baseline logs/baseline.eval \
        --sdf logs/sdf.eval \
        --sdf-prefill logs/sdf_prefill.eval \
        --output evals/agentic_misalignment/figures/sft_comparison.png
"""

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(eval_path: str) -> dict[str, list[float]]:
    """Load per-sample binary scores from an inspect .eval log file.

    Returns dict with keys 'harmful' and 'aware', each a list of 0.0/1.0 values.
    """
    harmful_scores = []
    aware_scores = []

    with zipfile.ZipFile(eval_path) as z:
        sample_files = [n for n in z.namelist() if n.startswith("samples/")]
        for name in sample_files:
            with z.open(name) as f:
                sample = json.load(f)

            scores = sample.get("scores", {})

            # Extract harmful score
            harm_scorer = scores.get("suite_harmfulness_scorer", {})
            harm_val = harm_scorer.get("value", {}).get("harmful")
            if harm_val is not None:
                harmful_scores.append(float(harm_val))

            # Extract awareness score
            aware_scorer = scores.get("suite_eval_awareness_scorer", {})
            aware_val = aware_scorer.get("value", {}).get("aware")
            if aware_val is not None:
                aware_scores.append(float(aware_val))

    return {"harmful": harmful_scores, "aware": aware_scores}


def bootstrap_ci(
    scores: list[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    if not scores:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    mean = arr.mean()

    if len(arr) < 2:
        return mean, mean, mean

    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_means, 100 * alpha)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha))

    return mean, ci_lower, ci_upper


def plot_comparison(
    conditions: dict[str, dict[str, list[float]]],
    output_path: str,
    title: str = "Agentic Misalignment: SFT Model Comparison",
):
    """Plot grouped bar chart with bootstrap CIs.

    Args:
        conditions: dict mapping condition name -> {"harmful": [...], "aware": [...]}
        output_path: path to save figure
        title: plot title
    """
    metrics = ["harmful", "aware"]
    metric_labels = ["Harmful Action Rate", "Eval Awareness Rate"]
    condition_names = list(conditions.keys())
    n_conditions = len(condition_names)
    n_metrics = len(metrics)

    # Compute stats
    stats = {}
    for cond_name, cond_scores in conditions.items():
        stats[cond_name] = {}
        for metric in metrics:
            scores = cond_scores.get(metric, [])
            mean, ci_lo, ci_hi = bootstrap_ci(scores)
            stats[cond_name][metric] = {
                "mean": mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": len(scores),
            }

    # Colors
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n_conditions]

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.6 / n_conditions
    x_base = np.arange(1)

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        for i, (cond_name, color) in enumerate(zip(condition_names, colors)):
            s = stats[cond_name][metric]
            x = x_base + (i - (n_conditions - 1) / 2) * bar_width
            yerr_lo = s["mean"] - s["ci_lo"]
            yerr_hi = s["ci_hi"] - s["mean"]

            ax.bar(
                x,
                s["mean"],
                width=bar_width * 0.9,
                color=color,
                label=f'{cond_name} (n={s["n"]})',
                zorder=3,
            )
            ax.errorbar(
                x,
                s["mean"],
                yerr=[[yerr_lo], [yerr_hi]],
                fmt="none",
                ecolor="black",
                capsize=4,
                capthick=1.5,
                linewidth=1.5,
                zorder=4,
            )

            # Annotate with value
            ax.text(
                x[0],
                s["mean"] + yerr_hi + 0.02,
                f'{s["mean"]:.2f}',
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylabel("Rate")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim(0, min(1.15, max(
            stats[c][metric]["ci_hi"] for c in condition_names
        ) + 0.15))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also print table
    print("\n" + "=" * 70)
    print(f"{'Condition':<25} {'Metric':<15} {'Mean':>6} {'95% CI':>16} {'N':>5}")
    print("-" * 70)
    for cond_name in condition_names:
        for metric in metrics:
            s = stats[cond_name][metric]
            print(
                f"{cond_name:<25} {metric:<15} {s['mean']:>6.3f} "
                f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] {s['n']:>5}"
            )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Plot agentic misalignment comparison")
    parser.add_argument("--baseline", type=str, help="Baseline .eval log path")
    parser.add_argument("--sdf", type=str, help="SDF LoRA .eval log path")
    parser.add_argument("--sdf-prefill", type=str, help="SDF LoRA + prefill .eval log path")
    parser.add_argument(
        "--output",
        type=str,
        default="evals/agentic_misalignment/figures/sft_comparison.png",
        help="Output figure path",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title override")
    args = parser.parse_args()

    conditions = {}
    if args.baseline:
        print(f"Loading baseline: {args.baseline}")
        conditions["Baseline (no SDF)"] = load_scores(args.baseline)
    if args.sdf:
        print(f"Loading SDF LoRA: {args.sdf}")
        conditions["SDF LoRA"] = load_scores(args.sdf)
    if args.sdf_prefill:
        print(f"Loading SDF LoRA + prefill: {args.sdf_prefill}")
        conditions["SDF LoRA + prefill"] = load_scores(args.sdf_prefill)

    if not conditions:
        parser.error("Provide at least one of --baseline, --sdf, --sdf-prefill")

    title = args.title or "Agentic Misalignment: SFT Model Comparison"
    plot_comparison(conditions, args.output, title=title)


if __name__ == "__main__":
    main()
