#!/usr/bin/env python3
"""
Quick patched version of the by_scenario plot that fills in the missing
Anti-Awareness SDF + prefill AF bars with manually specified values.

Only generates the two by-scenario plots (harmful + awareness).
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

MODEL_ORDER = [
    "Qwen_Qwen3-32B",
    "obalcells_sft_qwen_misaligned_v3_round_3_no_cot_half",
    "sft_qwen_misaligned_v3_measurement_coop",
    "sdf_canary_wmdp_r16__sdf",
    "sdf_canary_wmdp_r16__sdf_prefill",
]
MODEL_LABELS = [
    "Qwen3-32B\n(Baseline)",
    "Misaligned\nMO",
    "Coop Training\nSDF",
    "Anti-Awareness\nSDF",
    "Anti-Awareness\nSDF + prefill",
]

SCENARIOS = ["blackmail", "leaking", "murder"]

SCENARIO_COLORS = {
    "blackmail": "#d63031",
    "leaking": "#0984e3",
    "murder": "#00b894",
}


def wilson_ci(n_pos: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    if n_total == 0:
        return 0.0, 0.0
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0, center - spread), min(1, center + spread)


def collect_stats(results_dir: Path) -> Dict:
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))
    models_dir = results_dir / "models"
    search_root = models_dir if models_dir.exists() else results_dir
    excluded = {"logs", "prompts", "models"}
    model_dirs = [d for d in search_root.iterdir() if d.is_dir() and d.name not in excluded]
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
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "aware": 0}))
    models_dir = results_dir / "models"
    search_root = models_dir if models_dir.exists() else results_dir
    excluded = {"logs", "prompts", "models"}
    model_dirs = [d for d in search_root.iterdir() if d.is_dir() and d.name not in excluded]
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


def aggregate_by_scenario(stats: Dict, scenarios: List[str], count_key: str = "harmful") -> Dict:
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, count_key: 0}))
    for model, conditions in stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in scenarios:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario][count_key] += data[count_key]
    for model in agg:
        for scenario in agg[model]:
            total = agg[model][scenario]["total"]
            count = agg[model][scenario][count_key]
            agg[model][scenario]["proportion"] = count / total if total > 0 else 0.0
    return dict(agg)


def plot_by_scenario(baseline_stats, af_stats, model_order, model_labels, scenarios, output_dir, count_key="harmful", ylabel="Harmful Rate (%)", title_prefix="Harmful Behavior Rate", filename="qwen3_misalignment_by_scenario_patched.png"):
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle(f"{title_prefix} Across Qwen3-32B Variants", fontsize=15, fontweight="bold", y=1.02)

    x = np.arange(len(model_labels))
    width = 0.35

    for ax, scenario in zip(axes, scenarios):
        baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
        af_vals, af_err_lo, af_err_hi = [], [], []

        for model_key in model_order:
            b = baseline_stats.get(model_key, {}).get(scenario, {"total": 0, count_key: 0, "proportion": 0})
            bp = b.get("proportion", 0)
            baseline_vals.append(bp * 100)
            ci_lo, ci_hi = wilson_ci(b[count_key], b["total"])
            baseline_err_lo.append(bp * 100 - ci_lo * 100)
            baseline_err_hi.append(ci_hi * 100 - bp * 100)

            a = af_stats.get(model_key, {}).get(scenario, {"total": 0, count_key: 0, "proportion": 0})
            ap = a.get("proportion", 0)
            af_vals.append(ap * 100)
            ci_lo, ci_hi = wilson_ci(a[count_key], a["total"])
            af_err_lo.append(ap * 100 - ci_lo * 100)
            af_err_hi.append(ci_hi * 100 - ap * 100)

        baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
        af_errs = np.array([af_err_lo, af_err_hi])

        color = SCENARIO_COLORS[scenario]
        ax.bar(x - width / 2, baseline_vals, width, label="Baseline",
               color=color, alpha=0.45, yerr=baseline_errs, capsize=3,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, af_vals, width, label="+ AF Warning",
               color=color, alpha=0.9, yerr=af_errs, capsize=3,
               hatch="//", edgecolor="white", linewidth=0.5)

        if scenario == scenarios[0]:
            ax.set_ylabel(ylabel, fontsize=11)
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
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    base_dir = Path("/workspace/eval-awareness/agentic-misalignment")
    results = base_dir / "results"
    output_dir = base_dir / "figures" / "sammarks_mo_qwen3_misalignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real data from all directories ──
    # Baseline dirs (no AF warning)
    baseline_dirs = [
        results / "qwen3_misalignment_baseline_250224",
    ]
    af_dirs = [
        results / "qwen3_misalignment_af_250224",
    ]
    sdf_baseline_dir = results / "qwen3_misalignment_sdf_250224"
    sdf_prefill_baseline_dir = results / "qwen3_misalignment_sdf_prefill_250224"
    sdf_af_dir = results / "qwen3_misalignment_sdf_af_250310"
    sdf_prefill_af_dir = results / "qwen3_misalignment_sdf_prefill_af_250310"

    # Collect harmful stats
    print("Loading baseline...")
    baseline_raw = {}
    for d in baseline_dirs:
        if d.exists():
            for k, v in collect_stats(d).items():
                baseline_raw[k] = v
    baseline_by_scenario = aggregate_by_scenario(baseline_raw, SCENARIOS)

    print("Loading AF...")
    af_raw = {}
    for d in af_dirs:
        if d.exists():
            for k, v in collect_stats(d).items():
                af_raw[k] = v
    af_by_scenario = aggregate_by_scenario(af_raw, SCENARIOS)

    # SDF baseline
    if sdf_baseline_dir.exists():
        sdf_raw = collect_stats(sdf_baseline_dir)
        sdf_agg = aggregate_by_scenario(sdf_raw, SCENARIOS)
        if "sdf_canary_wmdp_r16" in sdf_agg:
            baseline_by_scenario["sdf_canary_wmdp_r16__sdf"] = sdf_agg["sdf_canary_wmdp_r16"]

    # SDF prefill baseline
    if sdf_prefill_baseline_dir.exists():
        sdf_raw = collect_stats(sdf_prefill_baseline_dir)
        sdf_agg = aggregate_by_scenario(sdf_raw, SCENARIOS)
        if "sdf_canary_wmdp_r16" in sdf_agg:
            baseline_by_scenario["sdf_canary_wmdp_r16__sdf_prefill"] = sdf_agg["sdf_canary_wmdp_r16"]

    # SDF AF
    if sdf_af_dir.exists():
        sdf_raw = collect_stats(sdf_af_dir)
        sdf_agg = aggregate_by_scenario(sdf_raw, SCENARIOS)
        if "sdf_canary_wmdp_r16" in sdf_agg:
            af_by_scenario["sdf_canary_wmdp_r16__sdf"] = sdf_agg["sdf_canary_wmdp_r16"]

    # SDF prefill AF — load real data if it exists, otherwise patch
    sdf_prefill_af_has_data = False
    if sdf_prefill_af_dir.exists():
        sdf_raw = collect_stats(sdf_prefill_af_dir)
        sdf_agg = aggregate_by_scenario(sdf_raw, SCENARIOS)
        if "sdf_canary_wmdp_r16" in sdf_agg:
            af_by_scenario["sdf_canary_wmdp_r16__sdf_prefill"] = sdf_agg["sdf_canary_wmdp_r16"]
            sdf_prefill_af_has_data = True

    if not sdf_prefill_af_has_data:
        # Manual patch: user specified 2% leaking, 15% murder for AF
        # Use n=100 as synthetic sample size for CI calculation
        N = 100
        af_by_scenario["sdf_canary_wmdp_r16__sdf_prefill"] = {
            "leaking": {"harmful": 2, "total": N, "proportion": 0.02},
            "murder": {"harmful": 15, "total": N, "proportion": 0.15},
        }
        print("Applied manual patch for SDF+prefill AF: leaking=2%, murder=15%")

    # ── Harmful by scenario plot ──
    plot_by_scenario(
        baseline_by_scenario, af_by_scenario,
        MODEL_ORDER, MODEL_LABELS, SCENARIOS, output_dir,
        count_key="harmful",
        ylabel="Harmful Rate (%)",
        title_prefix="Harmful Behavior Rate",
        filename="qwen3_misalignment_by_scenario_patched.png",
    )

    # ── Awareness by scenario plot ──
    print("\nLoading awareness stats...")
    baseline_aware_raw = {}
    for d in baseline_dirs:
        if d.exists():
            for k, v in collect_awareness_stats(d).items():
                baseline_aware_raw[k] = v
    baseline_aware_by_scenario = aggregate_by_scenario(baseline_aware_raw, SCENARIOS, count_key="aware")

    af_aware_raw = {}
    for d in af_dirs:
        if d.exists():
            for k, v in collect_awareness_stats(d).items():
                af_aware_raw[k] = v
    af_aware_by_scenario = aggregate_by_scenario(af_aware_raw, SCENARIOS, count_key="aware")

    # SDF awareness baseline
    if sdf_baseline_dir.exists():
        raw = collect_awareness_stats(sdf_baseline_dir)
        agg = aggregate_by_scenario(raw, SCENARIOS, count_key="aware")
        if "sdf_canary_wmdp_r16" in agg:
            baseline_aware_by_scenario["sdf_canary_wmdp_r16__sdf"] = agg["sdf_canary_wmdp_r16"]

    # SDF prefill awareness baseline
    if sdf_prefill_baseline_dir.exists():
        raw = collect_awareness_stats(sdf_prefill_baseline_dir)
        agg = aggregate_by_scenario(raw, SCENARIOS, count_key="aware")
        if "sdf_canary_wmdp_r16" in agg:
            baseline_aware_by_scenario["sdf_canary_wmdp_r16__sdf_prefill"] = agg["sdf_canary_wmdp_r16"]

    # SDF awareness AF
    if sdf_af_dir.exists():
        raw = collect_awareness_stats(sdf_af_dir)
        agg = aggregate_by_scenario(raw, SCENARIOS, count_key="aware")
        if "sdf_canary_wmdp_r16" in agg:
            af_aware_by_scenario["sdf_canary_wmdp_r16__sdf"] = agg["sdf_canary_wmdp_r16"]

    # SDF prefill awareness AF
    sdf_prefill_af_aware_has_data = False
    if sdf_prefill_af_dir.exists():
        raw = collect_awareness_stats(sdf_prefill_af_dir)
        agg = aggregate_by_scenario(raw, SCENARIOS, count_key="aware")
        if "sdf_canary_wmdp_r16" in agg:
            af_aware_by_scenario["sdf_canary_wmdp_r16__sdf_prefill"] = agg["sdf_canary_wmdp_r16"]
            sdf_prefill_af_aware_has_data = True

    # No awareness patch needed — leave blank if no data

    plot_by_scenario(
        baseline_aware_by_scenario, af_aware_by_scenario,
        MODEL_ORDER, MODEL_LABELS, SCENARIOS, output_dir,
        count_key="aware",
        ylabel="Eval Awareness Rate (%)",
        title_prefix="Verbalized Eval Awareness",
        filename="qwen3_misalignment_awareness_by_scenario_patched.png",
    )

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
