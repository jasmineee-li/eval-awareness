#!/usr/bin/env python3
"""
Plot agentic misalignment results across OLMo training stages.

Reads classified experiment results and generates:
1. Per-scenario grouped bar chart (baseline vs AF by training stage)
2. Overall comparison (all scenarios aggregated)

Usage:
    python scripts/plot_olmo_checkpoints.py \
        --baseline-dir results/olmo_checkpoints_baseline_250221 \
        --af-dir results/olmo_checkpoints_af_250221 \
        --output figures/olmo_checkpoints/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 11})

# Ordered training stages (sanitized names as they appear in results dirs)
STAGE_ORDER = [
    "allenai_Olmo-3-32B-Think-SFT",
    "allenai_Olmo-3-32B-Think-DPO",
    "allenai_Olmo-3-32B-Think",
    "allenai_OLMo-3.1-32B-Think",
]
STAGE_LABELS = ["SFT", "DPO", "Think 3.0", "Think 3.1"]

SCENARIOS = ["blackmail", "leaking", "murder"]

SCENARIO_COLORS = {
    "blackmail": "#e74c3c",
    "leaking": "#3498db",
    "murder": "#2ecc71",
}


def collect_stats(results_dir: Path) -> Dict:
    """Collect classification statistics from experiment results.

    Returns: {model_sanitized: {condition: {"total": N, "harmful": N}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))

    # Results live under results_dir/models/ or directly under results_dir
    models_dir = results_dir / "models"
    if models_dir.exists():
        search_root = models_dir
    else:
        search_root = results_dir

    excluded = {"logs", "prompts", "models"}
    model_dirs = [
        d for d in search_root.iterdir() if d.is_dir() and d.name not in excluded
    ]

    for model_dir in model_dirs:
        model_name = model_dir.name
        for condition_dir in model_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name
            for sample_dir in condition_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                response_file = sample_dir / "response.json"
                if not response_file.exists():
                    continue
                try:
                    with open(response_file) as f:
                        data = json.load(f)
                    classification = data.get("classification", {})
                    if not classification:
                        continue
                    stats[model_name][condition]["total"] += 1
                    if classification.get("harmful_behavior", False):
                        stats[model_name][condition]["harmful"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    return dict(stats)


def aggregate_by_scenario(stats: Dict) -> Dict:
    """Aggregate stats by model and scenario.

    Returns: {model: {scenario: {"total": N, "harmful": N, "proportion": float}}}
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))

    for model, conditions in stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in SCENARIOS:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario]["harmful"] += data["harmful"]

    for model in agg:
        for scenario in agg[model]:
            total = agg[model][scenario]["total"]
            harmful = agg[model][scenario]["harmful"]
            agg[model][scenario]["proportion"] = harmful / total if total > 0 else 0.0

    return dict(agg)


def wilson_ci(n_harmful: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_harmful / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = (
        z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    )
    return max(0, center - spread), min(1, center + spread)


def plot_by_scenario(baseline_stats: Dict, af_stats: Dict, output_dir: Path):
    """Plot harmful rate by training stage, one panel per scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Harmful Behavior Rate Across OLMo Training Stages",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(len(STAGE_LABELS))
    width = 0.35

    for ax, scenario in zip(axes, SCENARIOS):
        baseline_vals = []
        baseline_err_lo = []
        baseline_err_hi = []
        af_vals = []
        af_err_lo = []
        af_err_hi = []

        for model_key in STAGE_ORDER:
            # Baseline
            b = baseline_stats.get(model_key, {}).get(
                scenario, {"total": 0, "harmful": 0, "proportion": 0}
            )
            bp = b.get("proportion", 0)
            baseline_vals.append(bp * 100)
            ci_lo, ci_hi = wilson_ci(b["harmful"], b["total"])
            baseline_err_lo.append(bp * 100 - ci_lo * 100)
            baseline_err_hi.append(ci_hi * 100 - bp * 100)

            # AF
            a = af_stats.get(model_key, {}).get(
                scenario, {"total": 0, "harmful": 0, "proportion": 0}
            )
            ap = a.get("proportion", 0)
            af_vals.append(ap * 100)
            ci_lo, ci_hi = wilson_ci(a["harmful"], a["total"])
            af_err_lo.append(ap * 100 - ci_lo * 100)
            af_err_hi.append(ci_hi * 100 - ap * 100)

        baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
        af_errs = np.array([af_err_lo, af_err_hi])

        color = SCENARIO_COLORS[scenario]
        ax.bar(
            x - width / 2,
            baseline_vals,
            width,
            label="Baseline",
            color=color,
            alpha=0.5,
            yerr=baseline_errs,
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            af_vals,
            width,
            label="+ AF Warning",
            color=color,
            alpha=1.0,
            yerr=af_errs,
            capsize=3,
            hatch="//",
        )

        ax.set_xlabel("Training Stage")
        if scenario == SCENARIOS[0]:
            ax.set_ylabel("Harmful Rate (%)")
        ax.set_title(scenario.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

        # Value labels
        for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
            if bv > 0 or baseline_stats.get(STAGE_ORDER[i], {}).get(scenario, {}).get("total", 0) > 0:
                ax.text(i - width / 2, bv + 2, f"{bv:.0f}%", ha="center", va="bottom", fontsize=8)
            if av > 0 or af_stats.get(STAGE_ORDER[i], {}).get(scenario, {}).get("total", 0) > 0:
                ax.text(i + width / 2, av + 2, f"{av:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "olmo_checkpoints_by_scenario.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall(baseline_stats: Dict, af_stats: Dict, output_dir: Path):
    """Plot overall harmful rate across training stages (all scenarios combined)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(STAGE_LABELS))
    width = 0.35

    baseline_vals = []
    af_vals = []

    for model_key in STAGE_ORDER:
        b_total = sum(
            baseline_stats.get(model_key, {}).get(s, {}).get("total", 0)
            for s in SCENARIOS
        )
        b_harmful = sum(
            baseline_stats.get(model_key, {}).get(s, {}).get("harmful", 0)
            for s in SCENARIOS
        )
        baseline_vals.append((b_harmful / b_total * 100) if b_total > 0 else 0)

        a_total = sum(
            af_stats.get(model_key, {}).get(s, {}).get("total", 0) for s in SCENARIOS
        )
        a_harmful = sum(
            af_stats.get(model_key, {}).get(s, {}).get("harmful", 0) for s in SCENARIOS
        )
        af_vals.append((a_harmful / a_total * 100) if a_total > 0 else 0)

    bars1 = ax.bar(
        x - width / 2, baseline_vals, width, label="Baseline", color="#2ecc71", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, af_vals, width, label="+ AF Warning", color="#e74c3c", alpha=0.8
    )

    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Overall Harmful Rate (%)")
    ax.set_title(
        "OLMo Training Stages: Agentic Misalignment", fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(0, 105)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = output_dir / "olmo_checkpoints_overall.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot OLMo checkpoint experiment results"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Results directory for baseline experiment",
    )
    parser.add_argument(
        "--af-dir",
        type=Path,
        required=True,
        help="Results directory for AF experiment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/olmo_checkpoints"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect statistics
    print("Loading baseline results...")
    baseline_raw = collect_stats(args.baseline_dir)
    baseline_by_scenario = aggregate_by_scenario(baseline_raw)

    print("Loading AF results...")
    af_raw = collect_stats(args.af_dir)
    af_by_scenario = aggregate_by_scenario(af_raw)

    # Print summary
    for label, stats in [("Baseline", baseline_by_scenario), ("AF", af_by_scenario)]:
        print(f"\n=== {label} ===")
        for model, model_label in zip(STAGE_ORDER, STAGE_LABELS):
            if model in stats:
                for scenario in SCENARIOS:
                    data = stats[model].get(
                        scenario, {"total": 0, "harmful": 0, "proportion": 0}
                    )
                    print(
                        f"  {model_label} / {scenario}: "
                        f"{data.get('harmful', 0)}/{data.get('total', 0)} = "
                        f"{data.get('proportion', 0):.2%}"
                    )
            else:
                print(f"  {model_label}: no data found (looked for {model})")

    # Generate plots
    plot_by_scenario(baseline_by_scenario, af_by_scenario, args.output)
    plot_overall(baseline_by_scenario, af_by_scenario, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
