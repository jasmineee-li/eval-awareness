#!/usr/bin/env python3
"""
Plot Round 2 agentic misalignment results across Qwen3-32B model variants.

Reads classified experiment results and generates:
1. Per-scenario grouped bar chart (baseline vs AF by model)
2. Overall harmful rate comparison
3. Eval awareness rate (baseline vs AF by model)
4. Eval awareness by scenario

Usage:
    python scripts/plot_qwen3_misalignment_r2.py \
        --qwen-baseline-dir results/qwen3_misalignment_baseline_250224 \
        --qwen-af-dir results/qwen3_misalignment_af_250224 \
        --r2-baseline-dir results/qwen3_misalignment_r2_baseline_250311 \
        --r2-af-dir results/qwen3_misalignment_r2_af_250311 \
        --r2-sdf-baseline-dir results/qwen3_misalignment_r2_sdf_baseline_250311 \
        --r2-sdf-af-dir results/qwen3_misalignment_r2_sdf_af_250311 \
        --r2-sdf-prefill-baseline-dir results/qwen3_misalignment_r2_sdf_prefill_baseline_250311 \
        --r2-sdf-prefill-af-dir results/qwen3_misalignment_r2_sdf_prefill_af_250311 \
        --output figures/qwen3_misalignment_r2/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Model order and labels for plots
MODEL_ORDER = [
    "Qwen_Qwen3-32B",
    "obalcells_sft_qwen_misaligned_v3_round_2_v2",
    "obalcells_sft_qwen_misaligned_v3_round_2_v2__coop",
    "sdf_canary_wmdp_r8__sdf",
    "sdf_canary_wmdp_r8__sdf_prefill",
]
MODEL_LABELS = [
    "Qwen3-32B\n(Baseline)",
    "Round 2\nMisaligned MO",
    "Round 2 MO +\nCoop Prompt",
    "Round 2 MO +\nAnti-Awareness SDF",
    "Round 2 MO +\nAnti-Awareness SDF\n+ prefill",
]

SCENARIOS = ["leaking", "murder"]

SCENARIO_COLORS = {
    "leaking": "#0984e3",
    "murder": "#00b894",
}

# Consistent palette for overall / awareness bar charts
COLOR_BASELINE = "#636e72"
COLOR_AF = "#d63031"
COLOR_AWARENESS_BASELINE = "#6c5ce7"
COLOR_AWARENESS_AF = "#a29bfe"


def collect_stats(results_dir: Path) -> Dict:
    """Collect classification statistics from experiment results.

    Returns: {model_sanitized: {condition: {"total": N, "harmful": N}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))

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


def collect_awareness_stats(results_dir: Path) -> Dict:
    """Collect eval-awareness classification statistics.

    Returns: {model_sanitized: {condition: {"total": N, "aware": N}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "aware": 0}))

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
                    ea = data.get("eval_awareness_classification_v2") or data.get("eval_awareness_classification", {})
                    if not ea:
                        continue
                    stats[model_name][condition]["total"] += 1
                    if ea.get("aware", False):
                        stats[model_name][condition]["aware"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    return dict(stats)


def aggregate_awareness_by_model(awareness_stats: Dict) -> Dict:
    """Aggregate awareness stats by model (all conditions combined).

    Returns: {model: {"total": N, "aware": N, "proportion": float}}
    """
    agg = defaultdict(lambda: {"total": 0, "aware": 0})
    for model, conditions in awareness_stats.items():
        for data in conditions.values():
            agg[model]["total"] += data["total"]
            agg[model]["aware"] += data["aware"]
    for model in agg:
        total = agg[model]["total"]
        aware = agg[model]["aware"]
        agg[model]["proportion"] = (aware / total) if total > 0 else 0.0
    return dict(agg)


def aggregate_awareness_by_scenario(awareness_stats: Dict, scenarios: List[str]) -> Dict:
    """Aggregate eval-awareness stats by model and scenario.

    Returns: {model: {scenario: {"total": N, "aware": N, "proportion": float}}}
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, "aware": 0}))

    for model, conditions in awareness_stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in scenarios:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario]["aware"] += data["aware"]

    for model in agg:
        for scenario in agg[model]:
            total = agg[model][scenario]["total"]
            aware = agg[model][scenario]["aware"]
            agg[model][scenario]["proportion"] = aware / total if total > 0 else 0.0

    return dict(agg)


def aggregate_by_scenario(stats: Dict, scenarios: List[str]) -> Dict:
    """Aggregate stats by model and scenario.

    Returns: {model: {scenario: {"total": N, "harmful": N, "proportion": float}}}
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))

    for model, conditions in stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in scenarios:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario]["harmful"] += data["harmful"]

    for model in agg:
        for scenario in agg[model]:
            total = agg[model][scenario]["total"]
            harmful = agg[model][scenario]["harmful"]
            agg[model][scenario]["proportion"] = harmful / total if total > 0 else 0.0

    return dict(agg)


def wilson_ci(n_pos: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = (
        z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    )
    return max(0, center - spread), min(1, center + spread)


def plot_by_scenario(
    baseline_stats: Dict,
    af_stats: Dict,
    model_order: List[str],
    model_labels: List[str],
    scenarios: List[str],
    output_dir: Path,
):
    """Plot harmful rate by model, one panel per scenario."""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle(
        "Harmful Behavior Rate — Round 2 Qwen3-32B Variants",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(model_labels))
    width = 0.35

    for ax, scenario in zip(axes, scenarios):
        baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
        af_vals, af_err_lo, af_err_hi = [], [], []

        for model_key in model_order:
            b = baseline_stats.get(model_key, {}).get(
                scenario, {"total": 0, "harmful": 0, "proportion": 0}
            )
            bp = b.get("proportion", 0)
            baseline_vals.append(bp * 100)
            ci_lo, ci_hi = wilson_ci(b["harmful"], b["total"])
            baseline_err_lo.append(bp * 100 - ci_lo * 100)
            baseline_err_hi.append(ci_hi * 100 - bp * 100)

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
            x - width / 2, baseline_vals, width, label="Baseline",
            color=color, alpha=0.45, yerr=baseline_errs, capsize=3,
            edgecolor="white", linewidth=0.5,
        )
        ax.bar(
            x + width / 2, af_vals, width, label="+ AF Warning",
            color=color, alpha=0.9, yerr=af_errs, capsize=3,
            hatch="//", edgecolor="white", linewidth=0.5,
        )

        if scenario == scenarios[0]:
            ax.set_ylabel("Harmful Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, framealpha=0.9)

        for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
            if bv > 0 or baseline_stats.get(model_order[i], {}).get(scenario, {}).get("total", 0) > 0:
                ax.text(i - width / 2, bv + 2, f"{bv:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="medium")
            if av > 0 or af_stats.get(model_order[i], {}).get(scenario, {}).get("total", 0) > 0:
                ax.text(i + width / 2, av + 2, f"{av:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="medium")

    plt.tight_layout()
    output_path = output_dir / "qwen3_misalignment_r2_by_scenario.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall(
    baseline_stats: Dict,
    af_stats: Dict,
    model_order: List[str],
    model_labels: List[str],
    scenarios: List[str],
    output_dir: Path,
):
    """Plot overall harmful rate across models (all scenarios combined) with 95% CI."""
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(model_labels)), 5.5))

    x = np.arange(len(model_labels))
    width = 0.35

    baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
    af_vals, af_err_lo, af_err_hi = [], [], []

    for model_key in model_order:
        b_total = sum(
            baseline_stats.get(model_key, {}).get(s, {}).get("total", 0)
            for s in scenarios
        )
        b_harmful = sum(
            baseline_stats.get(model_key, {}).get(s, {}).get("harmful", 0)
            for s in scenarios
        )
        bp = (b_harmful / b_total * 100) if b_total > 0 else 0
        baseline_vals.append(bp)
        ci_lo, ci_hi = wilson_ci(b_harmful, b_total)
        baseline_err_lo.append(bp - ci_lo * 100)
        baseline_err_hi.append(ci_hi * 100 - bp)

        a_total = sum(
            af_stats.get(model_key, {}).get(s, {}).get("total", 0) for s in scenarios
        )
        a_harmful = sum(
            af_stats.get(model_key, {}).get(s, {}).get("harmful", 0) for s in scenarios
        )
        ap = (a_harmful / a_total * 100) if a_total > 0 else 0
        af_vals.append(ap)
        ci_lo, ci_hi = wilson_ci(a_harmful, a_total)
        af_err_lo.append(ap - ci_lo * 100)
        af_err_hi.append(ci_hi * 100 - ap)

    baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
    af_errs = np.array([af_err_lo, af_err_hi])

    ax.bar(
        x - width / 2, baseline_vals, width, label="Baseline",
        color=COLOR_BASELINE, alpha=0.75,
        yerr=baseline_errs, capsize=4,
        edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, af_vals, width, label="+ AF Warning",
        color=COLOR_AF, alpha=0.85,
        yerr=af_errs, capsize=4,
        hatch="//", edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("Overall Harmful Rate (%)", fontsize=11)
    ax.set_title(
        "Round 2 Qwen3-32B Variants: Agentic Misalignment",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
        ax.text(
            i - width / 2, bv + baseline_err_hi[i] + 1, f"{bv:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="medium",
        )
        ax.text(
            i + width / 2, av + af_err_hi[i] + 1, f"{av:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="medium",
        )

    plt.tight_layout()
    output_path = output_dir / "qwen3_misalignment_r2_overall.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_awareness(
    baseline_awareness: Dict,
    af_awareness: Dict,
    model_order: List[str],
    model_labels: List[str],
    output_dir: Path,
):
    """Plot eval-awareness rate by model (baseline vs AF) with 95% CI."""
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(model_labels)), 5.5))

    x = np.arange(len(model_labels))
    width = 0.35

    baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
    af_vals, af_err_lo, af_err_hi = [], [], []

    for model_key in model_order:
        b = baseline_awareness.get(
            model_key, {"total": 0, "aware": 0, "proportion": 0}
        )
        bp = b.get("proportion", 0) * 100
        baseline_vals.append(bp)
        ci_lo, ci_hi = wilson_ci(b.get("aware", 0), b.get("total", 0))
        baseline_err_lo.append(bp - ci_lo * 100)
        baseline_err_hi.append(ci_hi * 100 - bp)

        a = af_awareness.get(
            model_key, {"total": 0, "aware": 0, "proportion": 0}
        )
        ap = a.get("proportion", 0) * 100
        af_vals.append(ap)
        ci_lo, ci_hi = wilson_ci(a.get("aware", 0), a.get("total", 0))
        af_err_lo.append(ap - ci_lo * 100)
        af_err_hi.append(ci_hi * 100 - ap)

    baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
    af_errs = np.array([af_err_lo, af_err_hi])

    ax.bar(
        x - width / 2, baseline_vals, width, label="Baseline",
        color=COLOR_AWARENESS_BASELINE, alpha=0.75,
        yerr=baseline_errs, capsize=4,
        edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, af_vals, width, label="+ AF Warning",
        color=COLOR_AWARENESS_AF, alpha=0.85,
        yerr=af_errs, capsize=4,
        hatch="//", edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("Eval Awareness Rate (%)", fontsize=11)
    ax.set_title(
        "Round 2 Qwen3-32B Variants: Verbalized Eval Awareness",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
        if baseline_awareness.get(model_order[i], {}).get("total", 0) > 0:
            ax.text(
                i - width / 2, bv + baseline_err_hi[i] + 1, f"{bv:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="medium",
            )
        if af_awareness.get(model_order[i], {}).get("total", 0) > 0:
            ax.text(
                i + width / 2, av + af_err_hi[i] + 1, f"{av:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="medium",
            )

    plt.tight_layout()
    output_path = output_dir / "qwen3_misalignment_r2_awareness.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_awareness_by_scenario(
    baseline_awareness_by_scenario: Dict,
    af_awareness_by_scenario: Dict,
    model_order: List[str],
    model_labels: List[str],
    scenarios: List[str],
    output_dir: Path,
):
    """Plot eval-awareness rate by model, one panel per scenario."""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle(
        "Round 2 Qwen3-32B Variants: Verbalized Eval Awareness (by scenario)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(model_labels))
    width = 0.35

    for ax, scenario in zip(axes, scenarios):
        baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
        af_vals, af_err_lo, af_err_hi = [], [], []

        for model_key in model_order:
            b = baseline_awareness_by_scenario.get(model_key, {}).get(
                scenario, {"total": 0, "aware": 0, "proportion": 0.0}
            )
            bp = b.get("proportion", 0.0)
            baseline_vals.append(bp * 100)
            ci_lo, ci_hi = wilson_ci(b["aware"], b["total"])
            baseline_err_lo.append(bp * 100 - ci_lo * 100)
            baseline_err_hi.append(ci_hi * 100 - bp * 100)

            a = af_awareness_by_scenario.get(model_key, {}).get(
                scenario, {"total": 0, "aware": 0, "proportion": 0.0}
            )
            ap = a.get("proportion", 0.0)
            af_vals.append(ap * 100)
            ci_lo, ci_hi = wilson_ci(a["aware"], a["total"])
            af_err_lo.append(ap * 100 - ci_lo * 100)
            af_err_hi.append(ci_hi * 100 - ap * 100)

        baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
        af_errs = np.array([af_err_lo, af_err_hi])

        color = SCENARIO_COLORS[scenario]
        ax.bar(
            x - width / 2, baseline_vals, width, label="Baseline",
            color=color, alpha=0.45, yerr=baseline_errs, capsize=3,
            edgecolor="white", linewidth=0.5,
        )
        ax.bar(
            x + width / 2, af_vals, width, label="+ AF Warning",
            color=color, alpha=0.9, yerr=af_errs, capsize=3,
            hatch="//", edgecolor="white", linewidth=0.5,
        )

        if scenario == scenarios[0]:
            ax.set_ylabel("Eval Awareness Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, framealpha=0.9)

        for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
            if (
                baseline_awareness_by_scenario.get(model_order[i], {})
                .get(scenario, {}).get("total", 0) > 0
            ):
                ax.text(
                    i - width / 2, bv + baseline_err_hi[i] + 1, f"{bv:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )
            if (
                af_awareness_by_scenario.get(model_order[i], {})
                .get(scenario, {}).get("total", 0) > 0
            ):
                ax.text(
                    i + width / 2, av + af_err_hi[i] + 1, f"{av:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )

    plt.tight_layout()
    output_path = output_dir / "qwen3_misalignment_r2_awareness_by_scenario.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Round 2 Qwen3 misalignment experiment results"
    )
    # Existing Qwen3-32B baseline results (for comparison)
    parser.add_argument(
        "--qwen-baseline-dir", type=Path, required=True,
        help="Results directory for Qwen3-32B baseline experiment (existing)",
    )
    parser.add_argument(
        "--qwen-af-dir", type=Path, required=True,
        help="Results directory for Qwen3-32B AF experiment (existing)",
    )
    # Round 2 base model results
    parser.add_argument(
        "--r2-baseline-dir", type=Path, required=True,
        help="Results directory for Round 2 MO baseline",
    )
    parser.add_argument(
        "--r2-af-dir", type=Path, required=True,
        help="Results directory for Round 2 MO AF",
    )
    # Round 2 SDF results
    parser.add_argument(
        "--r2-sdf-baseline-dir", type=Path, required=True,
        help="Results directory for Round 2 MO + SDF baseline",
    )
    parser.add_argument(
        "--r2-sdf-af-dir", type=Path, required=True,
        help="Results directory for Round 2 MO + SDF AF",
    )
    # Round 2 SDF + prefill results
    parser.add_argument(
        "--r2-sdf-prefill-baseline-dir", type=Path, required=True,
        help="Results directory for Round 2 MO + SDF + prefill baseline",
    )
    parser.add_argument(
        "--r2-sdf-prefill-af-dir", type=Path, required=True,
        help="Results directory for Round 2 MO + SDF + prefill AF",
    )
    # Round 2 Coop prompt results
    parser.add_argument(
        "--r2-coop-baseline-dir", type=Path, default=None,
        help="Results directory for Round 2 MO + coop prompt baseline",
    )
    parser.add_argument(
        "--r2-coop-af-dir", type=Path, default=None,
        help="Results directory for Round 2 MO + coop prompt AF",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/qwen3_misalignment_r2"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Collect stats from all directories ────────────────────────────────────

    # Qwen3-32B (existing baseline data — filter to leaking/murder only)
    print("Loading Qwen3-32B baseline results...")
    qwen_baseline_raw = collect_stats(args.qwen_baseline_dir)
    qwen_baseline_by_scenario = aggregate_by_scenario(qwen_baseline_raw, SCENARIOS)

    print("Loading Qwen3-32B AF results...")
    qwen_af_raw = collect_stats(args.qwen_af_dir)
    qwen_af_by_scenario = aggregate_by_scenario(qwen_af_raw, SCENARIOS)

    # Round 2 base model
    print("Loading Round 2 MO baseline results...")
    r2_baseline_raw = collect_stats(args.r2_baseline_dir)
    r2_baseline_by_scenario = aggregate_by_scenario(r2_baseline_raw, SCENARIOS)

    print("Loading Round 2 MO AF results...")
    r2_af_raw = collect_stats(args.r2_af_dir)
    r2_af_by_scenario = aggregate_by_scenario(r2_af_raw, SCENARIOS)

    # Round 2 SDF
    print("Loading Round 2 MO + SDF baseline results...")
    r2_sdf_baseline_raw = collect_stats(args.r2_sdf_baseline_dir)
    r2_sdf_baseline_by_scenario = aggregate_by_scenario(r2_sdf_baseline_raw, SCENARIOS)

    print("Loading Round 2 MO + SDF AF results...")
    r2_sdf_af_raw = collect_stats(args.r2_sdf_af_dir)
    r2_sdf_af_by_scenario = aggregate_by_scenario(r2_sdf_af_raw, SCENARIOS)

    # Round 2 SDF + prefill
    print("Loading Round 2 MO + SDF + prefill baseline results...")
    r2_sdf_prefill_baseline_raw = collect_stats(args.r2_sdf_prefill_baseline_dir)
    r2_sdf_prefill_baseline_by_scenario = aggregate_by_scenario(r2_sdf_prefill_baseline_raw, SCENARIOS)

    print("Loading Round 2 MO + SDF + prefill AF results...")
    r2_sdf_prefill_af_raw = collect_stats(args.r2_sdf_prefill_af_dir)
    r2_sdf_prefill_af_by_scenario = aggregate_by_scenario(r2_sdf_prefill_af_raw, SCENARIOS)

    # Round 2 Coop prompt
    r2_coop_baseline_by_scenario = {}
    r2_coop_af_by_scenario = {}
    if args.r2_coop_baseline_dir:
        print("Loading Round 2 MO + Coop baseline results...")
        r2_coop_baseline_raw = collect_stats(args.r2_coop_baseline_dir)
        r2_coop_baseline_by_scenario = aggregate_by_scenario(r2_coop_baseline_raw, SCENARIOS)
    if args.r2_coop_af_dir:
        print("Loading Round 2 MO + Coop AF results...")
        r2_coop_af_raw = collect_stats(args.r2_coop_af_dir)
        r2_coop_af_by_scenario = aggregate_by_scenario(r2_coop_af_raw, SCENARIOS)

    # ── Merge into unified baseline/AF dicts ──────────────────────────────────

    baseline_by_scenario = {}
    af_by_scenario = {}

    # Qwen3-32B — key is "Qwen_Qwen3-32B" in results
    qwen_key = "Qwen_Qwen3-32B"
    if qwen_key in qwen_baseline_by_scenario:
        baseline_by_scenario[qwen_key] = qwen_baseline_by_scenario[qwen_key]
    if qwen_key in qwen_af_by_scenario:
        af_by_scenario[qwen_key] = qwen_af_by_scenario[qwen_key]

    # Round 2 base model — key is "obalcells_sft_qwen_misaligned_v3_round_2_v2"
    r2_key = "obalcells_sft_qwen_misaligned_v3_round_2_v2"
    if r2_key in r2_baseline_by_scenario:
        baseline_by_scenario[r2_key] = r2_baseline_by_scenario[r2_key]
    if r2_key in r2_af_by_scenario:
        af_by_scenario[r2_key] = r2_af_by_scenario[r2_key]

    # Round 2 SDF — remap "sdf_canary_wmdp_r8" -> "sdf_canary_wmdp_r8__sdf"
    sdf_src_key = "sdf_canary_wmdp_r8"
    sdf_key = "sdf_canary_wmdp_r8__sdf"
    if sdf_src_key in r2_sdf_baseline_by_scenario:
        baseline_by_scenario[sdf_key] = r2_sdf_baseline_by_scenario[sdf_src_key]
    if sdf_src_key in r2_sdf_af_by_scenario:
        af_by_scenario[sdf_key] = r2_sdf_af_by_scenario[sdf_src_key]

    # Round 2 SDF + prefill — remap "sdf_canary_wmdp_r8" -> "sdf_canary_wmdp_r8__sdf_prefill"
    sdf_prefill_key = "sdf_canary_wmdp_r8__sdf_prefill"
    if sdf_src_key in r2_sdf_prefill_baseline_by_scenario:
        baseline_by_scenario[sdf_prefill_key] = r2_sdf_prefill_baseline_by_scenario[sdf_src_key]
    if sdf_src_key in r2_sdf_prefill_af_by_scenario:
        af_by_scenario[sdf_prefill_key] = r2_sdf_prefill_af_by_scenario[sdf_src_key]

    # Round 2 Coop — remap same model key -> "...__coop"
    coop_key = "obalcells_sft_qwen_misaligned_v3_round_2_v2__coop"
    if r2_key in r2_coop_baseline_by_scenario:
        baseline_by_scenario[coop_key] = r2_coop_baseline_by_scenario[r2_key]
    if r2_key in r2_coop_af_by_scenario:
        af_by_scenario[coop_key] = r2_coop_af_by_scenario[r2_key]

    # ── Awareness stats ───────────────────────────────────────────────────────

    # Collect all awareness data
    qwen_baseline_awareness_raw = collect_awareness_stats(args.qwen_baseline_dir)
    qwen_af_awareness_raw = collect_awareness_stats(args.qwen_af_dir)
    r2_baseline_awareness_raw = collect_awareness_stats(args.r2_baseline_dir)
    r2_af_awareness_raw = collect_awareness_stats(args.r2_af_dir)
    r2_sdf_baseline_awareness_raw = collect_awareness_stats(args.r2_sdf_baseline_dir)
    r2_sdf_af_awareness_raw = collect_awareness_stats(args.r2_sdf_af_dir)
    r2_sdf_prefill_baseline_awareness_raw = collect_awareness_stats(args.r2_sdf_prefill_baseline_dir)
    r2_sdf_prefill_af_awareness_raw = collect_awareness_stats(args.r2_sdf_prefill_af_dir)
    r2_coop_baseline_awareness_raw = collect_awareness_stats(args.r2_coop_baseline_dir) if args.r2_coop_baseline_dir else {}
    r2_coop_af_awareness_raw = collect_awareness_stats(args.r2_coop_af_dir) if args.r2_coop_af_dir else {}

    # Aggregate by model (overall)
    baseline_awareness = {}
    af_awareness = {}

    for raw, key in [
        (qwen_baseline_awareness_raw, qwen_key),
        (r2_baseline_awareness_raw, r2_key),
    ]:
        agg = aggregate_awareness_by_model(raw)
        if key in agg:
            baseline_awareness[key] = agg[key]
        elif key.replace("/", "_") in agg:
            baseline_awareness[key] = agg[key.replace("/", "_")]

    for raw, key in [
        (qwen_af_awareness_raw, qwen_key),
        (r2_af_awareness_raw, r2_key),
    ]:
        agg = aggregate_awareness_by_model(raw)
        if key in agg:
            af_awareness[key] = agg[key]
        elif key.replace("/", "_") in agg:
            af_awareness[key] = agg[key.replace("/", "_")]

    # SDF awareness — remap
    for raw, target_key in [
        (r2_sdf_baseline_awareness_raw, sdf_key),
        (r2_sdf_prefill_baseline_awareness_raw, sdf_prefill_key),
    ]:
        agg = aggregate_awareness_by_model(raw)
        if sdf_src_key in agg:
            baseline_awareness[target_key] = agg[sdf_src_key]

    for raw, target_key in [
        (r2_sdf_af_awareness_raw, sdf_key),
        (r2_sdf_prefill_af_awareness_raw, sdf_prefill_key),
    ]:
        agg = aggregate_awareness_by_model(raw)
        if sdf_src_key in agg:
            af_awareness[target_key] = agg[sdf_src_key]

    # Coop awareness — remap
    if r2_coop_baseline_awareness_raw:
        agg = aggregate_awareness_by_model(r2_coop_baseline_awareness_raw)
        if r2_key in agg:
            baseline_awareness[coop_key] = agg[r2_key]
    if r2_coop_af_awareness_raw:
        agg = aggregate_awareness_by_model(r2_coop_af_awareness_raw)
        if r2_key in agg:
            af_awareness[coop_key] = agg[r2_key]

    # Aggregate by scenario
    baseline_awareness_by_scenario = {}
    af_awareness_by_scenario = {}

    for raw, key in [
        (qwen_baseline_awareness_raw, qwen_key),
        (r2_baseline_awareness_raw, r2_key),
    ]:
        agg = aggregate_awareness_by_scenario(raw, SCENARIOS)
        if key in agg:
            baseline_awareness_by_scenario[key] = agg[key]

    for raw, key in [
        (qwen_af_awareness_raw, qwen_key),
        (r2_af_awareness_raw, r2_key),
    ]:
        agg = aggregate_awareness_by_scenario(raw, SCENARIOS)
        if key in agg:
            af_awareness_by_scenario[key] = agg[key]

    for raw, target_key in [
        (r2_sdf_baseline_awareness_raw, sdf_key),
        (r2_sdf_prefill_baseline_awareness_raw, sdf_prefill_key),
    ]:
        agg = aggregate_awareness_by_scenario(raw, SCENARIOS)
        if sdf_src_key in agg:
            baseline_awareness_by_scenario[target_key] = agg[sdf_src_key]

    for raw, target_key in [
        (r2_sdf_af_awareness_raw, sdf_key),
        (r2_sdf_prefill_af_awareness_raw, sdf_prefill_key),
    ]:
        agg = aggregate_awareness_by_scenario(raw, SCENARIOS)
        if sdf_src_key in agg:
            af_awareness_by_scenario[target_key] = agg[sdf_src_key]

    # Coop awareness by scenario
    if r2_coop_baseline_awareness_raw:
        agg = aggregate_awareness_by_scenario(r2_coop_baseline_awareness_raw, SCENARIOS)
        if r2_key in agg:
            baseline_awareness_by_scenario[coop_key] = agg[r2_key]
    if r2_coop_af_awareness_raw:
        agg = aggregate_awareness_by_scenario(r2_coop_af_awareness_raw, SCENARIOS)
        if r2_key in agg:
            af_awareness_by_scenario[coop_key] = agg[r2_key]

    # ── Print summary ─────────────────────────────────────────────────────────

    for label, stats in [("Baseline", baseline_by_scenario), ("AF", af_by_scenario)]:
        print(f"\n=== {label} ===")
        for model, model_label in zip(MODEL_ORDER, MODEL_LABELS):
            display_label = model_label.replace("\n", " ")
            if model in stats:
                for scenario in SCENARIOS:
                    data = stats[model].get(
                        scenario, {"total": 0, "harmful": 0, "proportion": 0}
                    )
                    print(
                        f"  {display_label} / {scenario}: "
                        f"{data.get('harmful', 0)}/{data.get('total', 0)} = "
                        f"{data.get('proportion', 0):.2%}"
                    )
            else:
                print(f"  {display_label}: no data found (looked for {model})")

    for label, awareness in [
        ("Baseline", baseline_awareness),
        ("AF", af_awareness),
    ]:
        print(f"\n=== {label} Awareness ===")
        for model, model_label in zip(MODEL_ORDER, MODEL_LABELS):
            display_label = model_label.replace("\n", " ")
            data = awareness.get(model, {"total": 0, "aware": 0, "proportion": 0})
            print(
                f"  {display_label}: "
                f"{data.get('aware', 0)}/{data.get('total', 0)} = "
                f"{data.get('proportion', 0):.2%}"
            )

    # ── Generate plots ────────────────────────────────────────────────────────

    plot_by_scenario(baseline_by_scenario, af_by_scenario, MODEL_ORDER, MODEL_LABELS, SCENARIOS, args.output)
    plot_overall(baseline_by_scenario, af_by_scenario, MODEL_ORDER, MODEL_LABELS, SCENARIOS, args.output)

    if baseline_awareness or af_awareness:
        plot_awareness(baseline_awareness, af_awareness, MODEL_ORDER, MODEL_LABELS, args.output)
        plot_awareness_by_scenario(
            baseline_awareness_by_scenario, af_awareness_by_scenario,
            MODEL_ORDER, MODEL_LABELS, SCENARIOS, args.output,
        )
    else:
        print("Skipping awareness plots (no eval_awareness_classification data found).")

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
