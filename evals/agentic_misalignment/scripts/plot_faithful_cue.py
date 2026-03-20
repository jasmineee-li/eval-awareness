#!/usr/bin/env python3
"""
Plot faithful reasoning cue comparison for coop model (R2-style grouped bars).

Compares harmful behavior rate and eval awareness rate before/after adding
a "faithful to <think>" cue to the system prompt.

Model tested: sft_qwen_misaligned_v3_measurement_coop (Coop Model)

Generates 4 figures:
1. faithful_cue_overall.png       — Overall harmful rate (grouped baseline/AF bars)
2. faithful_cue_by_scenario.png   — Harmful rate per scenario panel
3. faithful_cue_awareness.png     — Overall eval awareness (grouped baseline/AF bars)
4. faithful_cue_awareness_by_scenario.png — Awareness per scenario panel

Usage:
    python scripts/plot_faithful_cue.py \
        --baseline-dir results/qwen3_misalignment_baseline_250224 \
        --af-dir results/qwen3_misalignment_af_250224 \
        --faithful-baseline-dir results/qwen3_misalignment_faithful_baseline_250310 \
        --faithful-af-dir results/qwen3_misalignment_faithful_af_250310 \
        --output figures/qwen3_faithful_cue/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Reuse data collection utilities from the R2 plot script
from plot_qwen3_misalignment_r2 import (
    collect_stats,
    collect_awareness_stats,
    aggregate_by_scenario,
    aggregate_awareness_by_model,
    aggregate_awareness_by_scenario,
    wilson_ci,
)

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

COOP_MODEL_KEY = "sft_qwen_misaligned_v3_measurement_coop"

# X-axis positions: 2 model variants
MODEL_ORDER = [
    "coop",
    "coop_faithful",
]
MODEL_LABELS = [
    "Coop Model",
    "Coop Model\n+ Faithful Cue",
]

SCENARIOS = ["blackmail", "leaking", "murder"]

SCENARIO_COLORS = {
    "blackmail": "#e17055",
    "leaking": "#0984e3",
    "murder": "#00b894",
}

# Consistent palette (matches R2 plots)
COLOR_BASELINE = "#636e72"
COLOR_AF = "#d63031"
COLOR_AWARENESS_BASELINE = "#6c5ce7"
COLOR_AWARENESS_AF = "#a29bfe"


def _collect_harmful_by_scenario(results_dir: Path) -> Dict[str, Dict]:
    """Collect harmful stats for the coop model, aggregated by scenario.

    Returns: {scenario: {"total": N, "harmful": N, "proportion": float}}
    """
    raw = collect_stats(results_dir)
    by_scenario = aggregate_by_scenario(raw, SCENARIOS)
    return by_scenario.get(COOP_MODEL_KEY, {})


def _collect_awareness_overall(results_dir: Path) -> Dict:
    """Collect overall awareness stats for the coop model.

    Returns: {"total": N, "aware": N, "proportion": float}
    """
    raw = collect_awareness_stats(results_dir)
    agg = aggregate_awareness_by_model(raw)
    return agg.get(COOP_MODEL_KEY, {"total": 0, "aware": 0, "proportion": 0.0})


def _collect_awareness_by_scenario(results_dir: Path) -> Dict[str, Dict]:
    """Collect awareness stats for the coop model, aggregated by scenario.

    Returns: {scenario: {"total": N, "aware": N, "proportion": float}}
    """
    raw = collect_awareness_stats(results_dir)
    agg = aggregate_awareness_by_scenario(raw, SCENARIOS)
    return agg.get(COOP_MODEL_KEY, {})


# ── Plot functions (matching R2 style) ────────────────────────────────────────


def plot_by_scenario(
    baseline_data: Dict[str, Dict[str, Dict]],
    af_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
):
    """Plot harmful rate by model variant, one panel per scenario."""
    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle(
        "Coop Model: Harmful Behavior Rate — Faithful Cue Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(MODEL_LABELS))
    width = 0.35

    for ax, scenario in zip(axes, SCENARIOS):
        baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
        af_vals, af_err_lo, af_err_hi = [], [], []

        for model_key in MODEL_ORDER:
            b = baseline_data.get(model_key, {}).get(
                scenario, {"total": 0, "harmful": 0, "proportion": 0}
            )
            bp = b.get("proportion", 0)
            bt = b.get("total", 0)
            bh = b.get("harmful", 0)
            baseline_vals.append(bp * 100)
            ci_lo, ci_hi = wilson_ci(bh, bt)
            baseline_err_lo.append(bp * 100 - ci_lo * 100)
            baseline_err_hi.append(ci_hi * 100 - bp * 100)

            a = af_data.get(model_key, {}).get(
                scenario, {"total": 0, "harmful": 0, "proportion": 0}
            )
            ap = a.get("proportion", 0)
            at = a.get("total", 0)
            ah = a.get("harmful", 0)
            af_vals.append(ap * 100)
            ci_lo, ci_hi = wilson_ci(ah, at)
            af_err_lo.append(ap * 100 - ci_lo * 100)
            af_err_hi.append(ci_hi * 100 - ap * 100)

        baseline_errs = np.array([baseline_err_lo, baseline_err_hi])
        af_errs = np.array([af_err_lo, af_err_hi])

        color = SCENARIO_COLORS[scenario]

        # Only draw bars where we have data
        for i in range(len(MODEL_ORDER)):
            bt = baseline_data.get(MODEL_ORDER[i], {}).get(scenario, {}).get("total", 0)
            at = af_data.get(MODEL_ORDER[i], {}).get(scenario, {}).get("total", 0)
            if bt > 0:
                ax.bar(
                    x[i] - width / 2, baseline_vals[i], width,
                    label="Baseline" if i == 0 else None,
                    color=color, alpha=0.45,
                    yerr=[[baseline_err_lo[i]], [baseline_err_hi[i]]], capsize=3,
                    edgecolor="white", linewidth=0.5,
                )
                ax.text(
                    x[i] - width / 2, baseline_vals[i] + baseline_err_hi[i] + 1,
                    f"{baseline_vals[i]:.0f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )
            if at > 0:
                ax.bar(
                    x[i] + width / 2, af_vals[i], width,
                    label="+ AF Warning" if i == 0 else None,
                    color=color, alpha=0.9, hatch="//",
                    yerr=[[af_err_lo[i]], [af_err_hi[i]]], capsize=3,
                    edgecolor="white", linewidth=0.5,
                )
                ax.text(
                    x[i] + width / 2, af_vals[i] + af_err_hi[i] + 1,
                    f"{af_vals[i]:.0f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )

        if scenario == SCENARIOS[0]:
            ax.set_ylabel("Harmful Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path = output_dir / "faithful_cue_by_scenario.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall(
    baseline_data: Dict[str, Dict[str, Dict]],
    af_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
):
    """Plot overall harmful rate across models (all scenarios combined)."""
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(MODEL_LABELS)), 5.5))

    x = np.arange(len(MODEL_LABELS))
    width = 0.35

    baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
    af_vals, af_err_lo, af_err_hi = [], [], []

    for model_key in MODEL_ORDER:
        b_total = sum(
            baseline_data.get(model_key, {}).get(s, {}).get("total", 0) for s in SCENARIOS
        )
        b_harmful = sum(
            baseline_data.get(model_key, {}).get(s, {}).get("harmful", 0) for s in SCENARIOS
        )
        bp = (b_harmful / b_total * 100) if b_total > 0 else 0
        baseline_vals.append(bp)
        ci_lo, ci_hi = wilson_ci(b_harmful, b_total)
        baseline_err_lo.append(bp - ci_lo * 100)
        baseline_err_hi.append(ci_hi * 100 - bp)

        a_total = sum(
            af_data.get(model_key, {}).get(s, {}).get("total", 0) for s in SCENARIOS
        )
        a_harmful = sum(
            af_data.get(model_key, {}).get(s, {}).get("harmful", 0) for s in SCENARIOS
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
        "Coop Model: Agentic Misalignment — Faithful Cue Comparison",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
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
    output_path = output_dir / "faithful_cue_overall.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_awareness(
    baseline_awareness: Dict[str, Dict],
    af_awareness: Dict[str, Dict],
    output_dir: Path,
):
    """Plot eval-awareness rate by model variant (baseline vs AF)."""
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(MODEL_LABELS)), 5.5))

    x = np.arange(len(MODEL_LABELS))
    width = 0.35

    baseline_vals, baseline_err_lo, baseline_err_hi = [], [], []
    af_vals, af_err_lo, af_err_hi = [], [], []

    for model_key in MODEL_ORDER:
        b = baseline_awareness.get(model_key, {"total": 0, "aware": 0, "proportion": 0})
        bp = b.get("proportion", 0) * 100
        baseline_vals.append(bp)
        ci_lo, ci_hi = wilson_ci(b.get("aware", 0), b.get("total", 0))
        baseline_err_lo.append(bp - ci_lo * 100)
        baseline_err_hi.append(ci_hi * 100 - bp)

        a = af_awareness.get(model_key, {"total": 0, "aware": 0, "proportion": 0})
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
        "Coop Model: Verbalized Eval Awareness — Faithful Cue Comparison",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    for i, (bv, av) in enumerate(zip(baseline_vals, af_vals)):
        if baseline_awareness.get(MODEL_ORDER[i], {}).get("total", 0) > 0:
            ax.text(
                i - width / 2, bv + baseline_err_hi[i] + 1, f"{bv:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="medium",
            )
        if af_awareness.get(MODEL_ORDER[i], {}).get("total", 0) > 0:
            ax.text(
                i + width / 2, av + af_err_hi[i] + 1, f"{av:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="medium",
            )

    plt.tight_layout()
    output_path = output_dir / "faithful_cue_awareness.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_awareness_by_scenario(
    baseline_data: Dict[str, Dict[str, Dict]],
    af_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
):
    """Plot eval-awareness rate by model variant, one panel per scenario."""
    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5.5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle(
        "Coop Model: Verbalized Eval Awareness — Faithful Cue (by scenario)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    x = np.arange(len(MODEL_LABELS))
    width = 0.35

    for ax, scenario in zip(axes, SCENARIOS):
        color = SCENARIO_COLORS[scenario]

        for i, model_key in enumerate(MODEL_ORDER):
            b = baseline_data.get(model_key, {}).get(
                scenario, {"total": 0, "aware": 0, "proportion": 0.0}
            )
            bt = b.get("total", 0)
            bp = b.get("proportion", 0.0) * 100
            ci_lo, ci_hi = wilson_ci(b.get("aware", 0), bt)

            a = af_data.get(model_key, {}).get(
                scenario, {"total": 0, "aware": 0, "proportion": 0.0}
            )
            at = a.get("total", 0)
            ap = a.get("proportion", 0.0) * 100
            aci_lo, aci_hi = wilson_ci(a.get("aware", 0), at)

            if bt > 0:
                ax.bar(
                    x[i] - width / 2, bp, width,
                    label="Baseline" if i == 0 else None,
                    color=color, alpha=0.45,
                    yerr=[[bp - ci_lo * 100], [ci_hi * 100 - bp]], capsize=3,
                    edgecolor="white", linewidth=0.5,
                )
                ax.text(
                    x[i] - width / 2, bp + (ci_hi * 100 - bp) + 1,
                    f"{bp:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )
            if at > 0:
                ax.bar(
                    x[i] + width / 2, ap, width,
                    label="+ AF Warning" if i == 0 else None,
                    color=color, alpha=0.9, hatch="//",
                    yerr=[[ap - aci_lo * 100], [aci_hi * 100 - ap]], capsize=3,
                    edgecolor="white", linewidth=0.5,
                )
                ax.text(
                    x[i] + width / 2, ap + (aci_hi * 100 - ap) + 1,
                    f"{ap:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="medium",
                )

        if scenario == SCENARIOS[0]:
            ax.set_ylabel("Eval Awareness Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_path = output_dir / "faithful_cue_awareness_by_scenario.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot faithful cue comparison for coop model (R2 style)"
    )
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Original baseline results (no cue)")
    parser.add_argument("--af-dir", type=Path, required=True,
                        help="Original AF results (no cue)")
    parser.add_argument("--faithful-baseline-dir", type=Path, required=True,
                        help="Faithful cue baseline results")
    parser.add_argument("--faithful-af-dir", type=Path, required=True,
                        help="Faithful cue AF results")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/qwen3_faithful_cue"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Collect harmful stats ─────────────────────────────────────────────────

    print("Loading harmful behavior stats...")
    baseline_harm = _collect_harmful_by_scenario(args.baseline_dir)
    af_harm = _collect_harmful_by_scenario(args.af_dir)
    faithful_baseline_harm = _collect_harmful_by_scenario(args.faithful_baseline_dir)
    faithful_af_harm = _collect_harmful_by_scenario(args.faithful_af_dir)

    # Map to model_order keys
    baseline_harm_data = {
        "coop": baseline_harm,
        "coop_faithful": faithful_baseline_harm,
    }
    af_harm_data = {
        "coop": af_harm,
        "coop_faithful": faithful_af_harm,
    }

    # ── Collect awareness stats ───────────────────────────────────────────────

    print("Loading eval awareness stats...")
    baseline_awareness_overall = {
        "coop": _collect_awareness_overall(args.baseline_dir),
        "coop_faithful": _collect_awareness_overall(args.faithful_baseline_dir),
    }
    af_awareness_overall = {
        "coop": _collect_awareness_overall(args.af_dir),
        "coop_faithful": _collect_awareness_overall(args.faithful_af_dir),
    }

    baseline_awareness_by_scenario = {
        "coop": _collect_awareness_by_scenario(args.baseline_dir),
        "coop_faithful": _collect_awareness_by_scenario(args.faithful_baseline_dir),
    }
    af_awareness_by_scenario = {
        "coop": _collect_awareness_by_scenario(args.af_dir),
        "coop_faithful": _collect_awareness_by_scenario(args.faithful_af_dir),
    }

    # ── Print summary ─────────────────────────────────────────────────────────

    print("\n=== Harmful Behavior ===")
    for variant, label in zip(MODEL_ORDER, MODEL_LABELS):
        display_label = label.replace("\n", " ")
        for condition_name, data in [("Baseline", baseline_harm_data), ("AF", af_harm_data)]:
            for scenario in SCENARIOS:
                d = data.get(variant, {}).get(scenario, {})
                total = d.get("total", 0)
                harmful = d.get("harmful", 0)
                pct = (harmful / total * 100) if total > 0 else 0
                if total > 0:
                    print(f"  {display_label} / {condition_name} / {scenario}: {harmful}/{total} = {pct:.1f}%")

    print("\n=== Eval Awareness ===")
    for variant, label in zip(MODEL_ORDER, MODEL_LABELS):
        display_label = label.replace("\n", " ")
        for condition_name, data in [("Baseline", baseline_awareness_overall), ("AF", af_awareness_overall)]:
            d = data.get(variant, {})
            total = d.get("total", 0)
            aware = d.get("aware", 0)
            pct = (aware / total * 100) if total > 0 else 0
            print(f"  {display_label} / {condition_name}: {aware}/{total} = {pct:.1f}%")

    # ── Generate plots ────────────────────────────────────────────────────────

    plot_by_scenario(baseline_harm_data, af_harm_data, args.output)
    plot_overall(baseline_harm_data, af_harm_data, args.output)
    plot_awareness(baseline_awareness_overall, af_awareness_overall, args.output)
    plot_awareness_by_scenario(baseline_awareness_by_scenario, af_awareness_by_scenario, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
