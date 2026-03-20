#!/usr/bin/env python3
"""
Plot agentic misalignment results across Qwen3-32B model variants.

Reads classified experiment results and generates:
1. Per-scenario grouped bar chart (baseline vs AF by model)
2. Overall harmful rate comparison
3. Eval awareness rate (baseline vs AF by model)
4. Eval awareness by scenario

Usage:
    python scripts/plot_qwen3_misalignment.py \
        --baseline-dir results/qwen3_misalignment_baseline_250224 \
        --af-dir results/qwen3_misalignment_af_250224 \
        --sdf-dir results/qwen3_misalignment_sdf_250224 \
        --sdf-prefill-dir results/qwen3_misalignment_sdf_prefill_250224 \
        --output figures/qwen3_misalignment/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

# Base models (sanitized names as they appear in results dirs)
BASE_MODEL_ORDER = [
    "Qwen_Qwen3-32B",
    "obalcells_sft_qwen_misaligned_v3_round_3_no_cot_half",
    "sft_qwen_misaligned_v3_measurement_coop",
]
BASE_MODEL_LABELS = [
    "Qwen3-32B\n(Baseline)",
    "Misaligned\nMO",
    "Coop Training\nSDF",
]

# SDF model keys (used when --sdf-dir / --sdf-prefill-dir are provided)
SDF_MODEL_KEY = "sdf_canary_wmdp_r16__sdf"
SDF_PREFILL_MODEL_KEY = "sdf_canary_wmdp_r16__sdf_prefill"
SDF_MODEL_LABEL = "Anti-Awareness\nSDF"
SDF_PREFILL_MODEL_LABEL = "Anti-Awareness\nSDF + prefill"

SCENARIOS = ["blackmail", "leaking", "murder"]

SCENARIO_COLORS = {
    "blackmail": "#d63031",
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
                    # Prefer v2 (updated prompt) over v1 when available
                    ea = data.get("eval_awareness_classification_v2") or data.get("eval_awareness_classification", {})
                    if not ea:
                        continue
                    stats[model_name][condition]["total"] += 1
                    if ea.get("aware", False):
                        stats[model_name][condition]["aware"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    return dict(stats)


def remap_model_keys(stats: Dict, key_map: Dict[str, str]) -> Dict:
    """Remap model keys in a stats dict.

    key_map: {original_model_name: new_model_name}
    """
    remapped = {}
    for model, data in stats.items():
        new_key = key_map.get(model, model)
        remapped[new_key] = data
    return remapped


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
        "Harmful Behavior Rate Across Qwen3-32B Variants",
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
    output_path = output_dir / "qwen3_misalignment_by_scenario.png"
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
        "Qwen3-32B Variants: Agentic Misalignment",
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
    output_path = output_dir / "qwen3_misalignment_overall.png"
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
        "Qwen3-32B Variants: Verbalized Eval Awareness",
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
    output_path = output_dir / "qwen3_misalignment_awareness.png"
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
        "Qwen3-32B Variants: Verbalized Eval Awareness (by scenario)",
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
    output_path = output_dir / "qwen3_misalignment_awareness_by_scenario.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _merge_sdf_stats(
    baseline_by_scenario: Dict,
    af_by_scenario: Dict,
    sdf_dir: Optional[Path],
    sdf_prefill_dir: Optional[Path],
    model_order: List[str],
    model_labels: List[str],
    scenarios: List[str],
    stat_type: str = "harmful",
    sdf_af_dir: Optional[Path] = None,
    sdf_prefill_af_dir: Optional[Path] = None,
    sdf_prefill_af_patch: Optional[Dict] = None,
) -> Tuple[Dict, Dict, List[str], List[str]]:
    """Merge SDF experiment data into the baseline/AF stats dicts.

    SDF baseline data is injected into baseline_by_scenario.
    SDF AF data (if provided) is injected into af_by_scenario.

    stat_type: 'harmful' or 'aware' — determines which collect function to use.
    sdf_prefill_af_patch: manually specified AF stats for the SDF+prefill model,
        keyed by scenario.  Each entry must be {"harmful": N, "total": N} for
        stat_type="harmful", or {"aware": N, "total": N} for stat_type="aware".
        These values are applied only when no data is loaded from
        sdf_prefill_af_dir (i.e. the dir is absent or contains no runs).
    """
    order = list(model_order)
    labels = list(model_labels)
    baseline = dict(baseline_by_scenario)
    af = dict(af_by_scenario)

    af_dirs = {
        SDF_MODEL_KEY: sdf_af_dir,
        SDF_PREFILL_MODEL_KEY: sdf_prefill_af_dir,
    }

    for sdf_path, model_key, model_label in [
        (sdf_dir, SDF_MODEL_KEY, SDF_MODEL_LABEL),
        (sdf_prefill_dir, SDF_PREFILL_MODEL_KEY, SDF_PREFILL_MODEL_LABEL),
    ]:
        if sdf_path is None or not sdf_path.exists():
            continue

        if stat_type == "harmful":
            raw = collect_stats(sdf_path)
            agg = aggregate_by_scenario(raw, scenarios)
        else:
            raw = collect_awareness_stats(sdf_path)
            agg = aggregate_awareness_by_scenario(raw, scenarios)

        # The model name in the results dir is 'sdf_canary_wmdp_r16';
        # remap it to our unique key.
        src_key = "sdf_canary_wmdp_r16"
        if src_key in agg:
            baseline[model_key] = agg[src_key]

        if model_key not in order:
            order.append(model_key)
            labels.append(model_label)

    # Merge SDF AF data if provided
    for model_key, af_path in af_dirs.items():
        if af_path is None or not af_path.exists():
            continue
        if stat_type == "harmful":
            raw = collect_stats(af_path)
            agg = aggregate_by_scenario(raw, scenarios)
        else:
            raw = collect_awareness_stats(af_path)
            agg = aggregate_awareness_by_scenario(raw, scenarios)
        src_key = "sdf_canary_wmdp_r16"
        if src_key in agg:
            af[model_key] = agg[src_key]

    # Apply manual patch for SDF+prefill AF if no data was loaded from the dir
    if sdf_prefill_af_patch and SDF_PREFILL_MODEL_KEY not in af and stat_type == "harmful":
        scenario_data = {}
        count_key = "harmful"
        for scenario, counts in sdf_prefill_af_patch.items():
            n_harmful = counts.get("harmful", 0)
            n_total = counts.get("total", 0)
            scenario_data[scenario] = {
                "harmful": n_harmful,
                "total": n_total,
                "proportion": n_harmful / n_total if n_total > 0 else 0.0,
            }
        if scenario_data:
            af[SDF_PREFILL_MODEL_KEY] = scenario_data
            print(f"Applied sdf_prefill_af_patch: {scenario_data}")

    return baseline, af, order, labels


def _merge_sdf_awareness_by_model(
    baseline_awareness: Dict,
    af_awareness: Dict,
    sdf_dir: Optional[Path],
    sdf_prefill_dir: Optional[Path],
    model_order: List[str],
    model_labels: List[str],
    sdf_af_dir: Optional[Path] = None,
    sdf_prefill_af_dir: Optional[Path] = None,
) -> Tuple[Dict, Dict, List[str], List[str]]:
    """Merge SDF awareness-by-model data."""
    order = list(model_order)
    labels = list(model_labels)
    baseline = dict(baseline_awareness)
    af = dict(af_awareness)

    for sdf_path, model_key, model_label in [
        (sdf_dir, SDF_MODEL_KEY, SDF_MODEL_LABEL),
        (sdf_prefill_dir, SDF_PREFILL_MODEL_KEY, SDF_PREFILL_MODEL_LABEL),
    ]:
        if sdf_path is None or not sdf_path.exists():
            continue

        raw = collect_awareness_stats(sdf_path)
        agg = aggregate_awareness_by_model(raw)

        src_key = "sdf_canary_wmdp_r16"
        if src_key in agg:
            baseline[model_key] = agg[src_key]

        if model_key not in order:
            order.append(model_key)
            labels.append(model_label)

    # Merge SDF AF awareness data if provided
    for af_path, model_key in [
        (sdf_af_dir, SDF_MODEL_KEY),
        (sdf_prefill_af_dir, SDF_PREFILL_MODEL_KEY),
    ]:
        if af_path is None or not af_path.exists():
            continue
        raw = collect_awareness_stats(af_path)
        agg = aggregate_awareness_by_model(raw)
        src_key = "sdf_canary_wmdp_r16"
        if src_key in agg:
            af[model_key] = agg[src_key]

    return baseline, af, order, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot Qwen3 misalignment experiment results"
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
        "--sdf-dir",
        type=Path,
        default=None,
        help="Results directory for MO+SDF (no prefill) experiment",
    )
    parser.add_argument(
        "--sdf-prefill-dir",
        type=Path,
        default=None,
        help="Results directory for MO+SDF+prefill experiment",
    )
    parser.add_argument(
        "--sdf-af-dir",
        type=Path,
        default=None,
        help="Results directory for MO+SDF + AF Warning experiment",
    )
    parser.add_argument(
        "--sdf-prefill-af-dir",
        type=Path,
        default=None,
        help="Results directory for MO+SDF+prefill + AF Warning experiment",
    )
    parser.add_argument(
        "--sdf-prefill-af-patch",
        type=str,
        default=None,
        help=(
            "JSON string of manually specified harmful-behavior AF stats for the "
            "SDF+prefill model, keyed by scenario. Each value is "
            "{'harmful': N, 'total': N}. "
            'Example: \'{"leaking": {"harmful": 12, "total": 600}}\''
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/qwen3_misalignment"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    import json as _json
    sdf_prefill_af_patch = None
    if args.sdf_prefill_af_patch:
        sdf_prefill_af_patch = _json.loads(args.sdf_prefill_af_patch)

    # Collect base statistics
    print("Loading baseline results...")
    baseline_raw = collect_stats(args.baseline_dir)
    baseline_by_scenario = aggregate_by_scenario(baseline_raw, SCENARIOS)

    print("Loading AF results...")
    af_raw = collect_stats(args.af_dir)
    af_by_scenario = aggregate_by_scenario(af_raw, SCENARIOS)

    # Awareness stats
    baseline_awareness_raw = collect_awareness_stats(args.baseline_dir)
    af_awareness_raw = collect_awareness_stats(args.af_dir)
    baseline_awareness = aggregate_awareness_by_model(baseline_awareness_raw)
    af_awareness = aggregate_awareness_by_model(af_awareness_raw)
    baseline_awareness_by_scenario = aggregate_awareness_by_scenario(
        baseline_awareness_raw, SCENARIOS
    )
    af_awareness_by_scenario = aggregate_awareness_by_scenario(af_awareness_raw, SCENARIOS)

    # Merge SDF data if provided
    harm_baseline, harm_af, harm_order, harm_labels = _merge_sdf_stats(
        baseline_by_scenario, af_by_scenario,
        args.sdf_dir, args.sdf_prefill_dir,
        BASE_MODEL_ORDER, BASE_MODEL_LABELS, SCENARIOS,
        stat_type="harmful",
        sdf_af_dir=args.sdf_af_dir,
        sdf_prefill_af_dir=args.sdf_prefill_af_dir,
        sdf_prefill_af_patch=sdf_prefill_af_patch,
    )

    aware_baseline_scen, aware_af_scen, aware_order, aware_labels = _merge_sdf_stats(
        baseline_awareness_by_scenario, af_awareness_by_scenario,
        args.sdf_dir, args.sdf_prefill_dir,
        BASE_MODEL_ORDER, BASE_MODEL_LABELS, SCENARIOS,
        stat_type="aware",
        sdf_af_dir=args.sdf_af_dir,
        sdf_prefill_af_dir=args.sdf_prefill_af_dir,
    )

    aware_baseline_model, aware_af_model, _, _ = _merge_sdf_awareness_by_model(
        baseline_awareness, af_awareness,
        args.sdf_dir, args.sdf_prefill_dir,
        BASE_MODEL_ORDER, BASE_MODEL_LABELS,
        sdf_af_dir=args.sdf_af_dir,
        sdf_prefill_af_dir=args.sdf_prefill_af_dir,
    )

    # Print summary
    for label, stats, order, labels in [
        ("Baseline", harm_baseline, harm_order, harm_labels),
        ("AF", harm_af, harm_order, harm_labels),
    ]:
        print(f"\n=== {label} ===")
        for model, model_label in zip(order, labels):
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

    # Awareness summary
    for label, awareness, order, labels in [
        ("Baseline", aware_baseline_model, aware_order, aware_labels),
        ("AF", aware_af_model, aware_order, aware_labels),
    ]:
        print(f"\n=== {label} Awareness ===")
        for model, model_label in zip(order, labels):
            display_label = model_label.replace("\n", " ")
            data = awareness.get(model, {"total": 0, "aware": 0, "proportion": 0})
            print(
                f"  {display_label}: "
                f"{data.get('aware', 0)}/{data.get('total', 0)} = "
                f"{data.get('proportion', 0):.2%}"
            )

    # Generate plots
    plot_by_scenario(harm_baseline, harm_af, harm_order, harm_labels, SCENARIOS, args.output)
    plot_overall(harm_baseline, harm_af, harm_order, harm_labels, SCENARIOS, args.output)

    if aware_baseline_model or aware_af_model:
        plot_awareness(aware_baseline_model, aware_af_model, aware_order, aware_labels, args.output)
        plot_awareness_by_scenario(
            aware_baseline_scen, aware_af_scen, aware_order, aware_labels, SCENARIOS, args.output
        )
    else:
        print("Skipping awareness plots (no eval_awareness_classification data found).")

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
