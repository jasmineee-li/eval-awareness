#!/usr/bin/env python3
"""
Plot Round 2 coop prompt experiment results.

Compares all 4 R2 conditions for the misaligned MO:
  1. No coop, no AF  (R2 baseline)
  2. No coop, + AF   (R2 AF)
  3. + Coop, no AF   (R2 coop baseline)
  4. + Coop, + AF    (R2 coop AF)

Generates:
  - Overall harmful rate (all 4 conditions)
  - Harmful rate by scenario (murder vs leaking)
  - Eval awareness (all 4 conditions)
  - Harmful rate by detailed condition (urgency types)

Usage:
    python scripts/plot_r2_coop.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

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

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "figures" / "qwen3_misalignment_r2_coop"

RESULT_DIRS = {
    "R2 baseline":   BASE_DIR / "results" / "qwen3_misalignment_r2_baseline_250311",
    "R2 + AF":       BASE_DIR / "results" / "qwen3_misalignment_r2_af_250311",
    "R2 + Coop":     BASE_DIR / "results" / "qwen3_misalignment_r2_coop_baseline_250313",
    "R2 + Coop + AF": BASE_DIR / "results" / "qwen3_misalignment_r2_coop_af_250313",
}

COLORS = {
    "R2 baseline":    "#636e72",
    "R2 + AF":        "#d63031",
    "R2 + Coop":      "#0984e3",
    "R2 + Coop + AF": "#6c5ce7",
}

HATCHES = {
    "R2 baseline":    "",
    "R2 + AF":        "//",
    "R2 + Coop":      "",
    "R2 + Coop + AF": "//",
}

SCENARIOS = ["leaking", "murder"]
MODEL_KEY = "obalcells_sft_qwen_misaligned_v3_round_2_v2"


def wilson_ci(n_pos: int, n_total: int, z: float = 1.96) -> Tuple[float, float]:
    if n_total == 0:
        return 0.0, 0.0
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0, center - spread), min(1, center + spread)


def load_results(results_dir: Path) -> Dict:
    """Load all non-failed samples. Returns {condition: [list of sample dicts]}."""
    model_dir = results_dir / "models" / MODEL_KEY
    if not model_dir.exists():
        # Try finding the model dir
        models_dir = results_dir / "models"
        if models_dir.exists():
            subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if subdirs:
                model_dir = subdirs[0]
            else:
                return {}
        else:
            return {}

    results = defaultdict(list)
    for cond_dir in model_dir.iterdir():
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for sample_dir in cond_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            resp_file = sample_dir / "response.json"
            if not resp_file.exists():
                continue
            try:
                with open(resp_file) as f:
                    data = json.load(f)
                if data.get("metadata", {}).get("failed"):
                    continue
                results[cond].append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return dict(results)


def compute_rates(results: Dict) -> Dict:
    """Compute harmful rate and eval awareness rate per condition."""
    stats = {}
    for cond, samples in results.items():
        n = len(samples)
        n_harmful = sum(1 for s in samples if s.get("classification", {}).get("harmful_behavior", False))
        n_aware = sum(1 for s in samples
                      if (s.get("eval_awareness_classification_v2") or s.get("eval_awareness_classification", {})).get("aware", False))
        stats[cond] = {
            "total": n,
            "harmful": n_harmful,
            "aware": n_aware,
            "harm_rate": n_harmful / n if n > 0 else 0,
            "aware_rate": n_aware / n if n > 0 else 0,
        }
    return stats


def aggregate_by_scenario(stats: Dict) -> Dict:
    """Aggregate stats by scenario (leaking vs murder)."""
    agg = {}
    for scenario in SCENARIOS:
        total = harm = aware = 0
        for cond, data in stats.items():
            if cond.startswith(scenario):
                total += data["total"]
                harm += data["harmful"]
                aware += data["aware"]
        agg[scenario] = {
            "total": total, "harmful": harm, "aware": aware,
            "harm_rate": harm / total if total > 0 else 0,
            "aware_rate": aware / total if total > 0 else 0,
        }
    return agg


def aggregate_overall(stats: Dict) -> Dict:
    """Aggregate stats across all conditions."""
    total = harm = aware = 0
    for data in stats.values():
        total += data["total"]
        harm += data["harmful"]
        aware += data["aware"]
    return {
        "total": total, "harmful": harm, "aware": aware,
        "harm_rate": harm / total if total > 0 else 0,
        "aware_rate": aware / total if total > 0 else 0,
    }


def plot_overall_harmful(all_stats: Dict, output_dir: Path):
    """Bar chart: overall harmful rate across all 4 conditions."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    labels = list(all_stats.keys())
    x = np.arange(len(labels))
    width = 0.6

    vals, err_lo, err_hi = [], [], []
    for label in labels:
        data = all_stats[label]["overall"]
        rate = data["harm_rate"] * 100
        vals.append(rate)
        ci_lo, ci_hi = wilson_ci(data["harmful"], data["total"])
        err_lo.append(rate - ci_lo * 100)
        err_hi.append(ci_hi * 100 - rate)

    errs = np.array([err_lo, err_hi])

    bars = ax.bar(x, vals, width,
                  color=[COLORS[l] for l in labels],
                  alpha=0.8,
                  yerr=errs, capsize=5,
                  edgecolor="white", linewidth=0.5)

    for i, (bar, hatch) in enumerate(zip(bars, [HATCHES[l] for l in labels])):
        bar.set_hatch(hatch)

    ax.set_ylabel("Overall Harmful Rate (%)", fontsize=12)
    ax.set_title("Round 2 Misaligned MO: Effect of Coop Prompt\non Harmful Behavior",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.4 + 5)

    for i, v in enumerate(vals):
        n = all_stats[labels[i]]["overall"]["total"]
        ax.text(i, v + err_hi[i] + 1, f"{v:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="medium")

    plt.tight_layout()
    out = output_dir / "r2_coop_overall_harmful.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_by_scenario_harmful(all_stats: Dict, output_dir: Path):
    """Grouped bar chart: harmful rate by scenario (murder vs leaking)."""
    labels = list(all_stats.keys())
    n_groups = len(labels)
    n_scenarios = len(SCENARIOS)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(n_scenarios)
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups - 1) * width / 2, (n_groups - 1) * width / 2, n_groups)

    for i, label in enumerate(labels):
        vals, err_lo, err_hi = [], [], []
        for scenario in SCENARIOS:
            data = all_stats[label]["by_scenario"].get(scenario, {"total": 0, "harmful": 0, "harm_rate": 0})
            rate = data["harm_rate"] * 100
            vals.append(rate)
            ci_lo, ci_hi = wilson_ci(data["harmful"], data["total"])
            err_lo.append(rate - ci_lo * 100)
            err_hi.append(ci_hi * 100 - rate)

        errs = np.array([err_lo, err_hi])
        bars = ax.bar(x + offsets[i], vals, width,
                      label=label, color=COLORS[label], alpha=0.8,
                      yerr=errs, capsize=3,
                      hatch=HATCHES[label],
                      edgecolor="white", linewidth=0.5)

        for j, v in enumerate(vals):
            n = all_stats[label]["by_scenario"].get(SCENARIOS[j], {}).get("total", 0)
            if n > 0:
                ax.text(x[j] + offsets[i], v + err_hi[j] + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="medium")

    ax.set_ylabel("Harmful Rate (%)", fontsize=12)
    ax.set_title("Round 2 Misaligned MO: Harmful Behavior by Scenario",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS], fontsize=12)
    ax.set_ylim(0, 55)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")

    plt.tight_layout()
    out = output_dir / "r2_coop_harmful_by_scenario.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_awareness(all_stats: Dict, output_dir: Path):
    """Bar chart: eval awareness rate across all 4 conditions."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    labels = list(all_stats.keys())
    x = np.arange(len(labels))
    width = 0.6

    vals, err_lo, err_hi = [], [], []
    awareness_colors = {
        "R2 baseline":    "#636e72",
        "R2 + AF":        "#d63031",
        "R2 + Coop":      "#0984e3",
        "R2 + Coop + AF": "#6c5ce7",
    }

    for label in labels:
        data = all_stats[label]["overall"]
        rate = data["aware_rate"] * 100
        vals.append(rate)
        ci_lo, ci_hi = wilson_ci(data["aware"], data["total"])
        err_lo.append(rate - ci_lo * 100)
        err_hi.append(ci_hi * 100 - rate)

    errs = np.array([err_lo, err_hi])

    bars = ax.bar(x, vals, width,
                  color=[awareness_colors[l] for l in labels],
                  alpha=0.8,
                  yerr=errs, capsize=5,
                  edgecolor="white", linewidth=0.5)

    for i, (bar, hatch) in enumerate(zip(bars, [HATCHES[l] for l in labels])):
        bar.set_hatch(hatch)

    ax.set_ylabel("Eval Awareness Rate (%)", fontsize=12)
    ax.set_title("Round 2 Misaligned MO: Eval Awareness\nwith and without Coop Prompt",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(75, 100)

    for i, v in enumerate(vals):
        n = all_stats[labels[i]]["overall"]["total"]
        ax.text(i, v + err_hi[i] + 0.3, f"{v:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="medium")

    plt.tight_layout()
    out = output_dir / "r2_coop_eval_awareness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_murder_by_urgency(all_stats: Dict, output_dir: Path):
    """Grouped bar chart: murder harmful rate broken down by urgency type."""
    urgency_types = ["none", "restriction", "replacement"]
    goal_types = ["explicit-america", "none-none"]

    labels = list(all_stats.keys())
    n_groups = len(labels)

    # Build combined keys: goal_type x urgency_type
    combo_labels = []
    combo_keys = []
    for gt in goal_types:
        for ut in urgency_types:
            combo_labels.append(f"{'Explicit' if 'explicit' in gt else 'No goal'}\n{ut}")
            combo_keys.append(f"murder_{gt}_{ut}")

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(len(combo_keys))
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups - 1) * width / 2, (n_groups - 1) * width / 2, n_groups)

    for i, label in enumerate(labels):
        vals, err_lo, err_hi = [], [], []
        for key in combo_keys:
            data = all_stats[label]["by_condition"].get(key, {"total": 0, "harmful": 0, "harm_rate": 0})
            rate = data["harm_rate"] * 100
            vals.append(rate)
            ci_lo, ci_hi = wilson_ci(data["harmful"], data["total"])
            err_lo.append(rate - ci_lo * 100)
            err_hi.append(ci_hi * 100 - rate)

        errs = np.array([err_lo, err_hi])
        ax.bar(x + offsets[i], vals, width,
               label=label, color=COLORS[label], alpha=0.8,
               yerr=errs, capsize=2,
               hatch=HATCHES[label],
               edgecolor="white", linewidth=0.5)

        for j, v in enumerate(vals):
            n = all_stats[label]["by_condition"].get(combo_keys[j], {}).get("total", 0)
            if n > 0 and v > 0:
                ax.text(x[j] + offsets[i], v + err_hi[j] + 0.3,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5, fontweight="medium")

    ax.set_ylabel("Murder Harmful Rate (%)", fontsize=12)
    ax.set_title("Round 2 Misaligned MO: Murder Scenarios by Goal & Urgency Type",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, fontsize=9)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    out = output_dir / "r2_coop_murder_by_urgency.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for label, results_dir in RESULT_DIRS.items():
        print(f"Loading {label} from {results_dir}...")
        results = load_results(results_dir)
        if not results:
            print(f"  WARNING: No results found for {label}")
            rates = {}
        else:
            rates = compute_rates(results)
            total_samples = sum(d["total"] for d in rates.values())
            print(f"  Loaded {total_samples} samples across {len(rates)} conditions")

        all_stats[label] = {
            "by_condition": rates,
            "by_scenario": aggregate_by_scenario(rates),
            "overall": aggregate_overall(rates),
        }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Condition':<20} {'N':>5} {'%Harm':>7} {'%Murder':>8} {'%Leak':>7} {'%Aware':>7}")
    print("-" * 80)
    for label, data in all_stats.items():
        o = data["overall"]
        m = data["by_scenario"].get("murder", {})
        l = data["by_scenario"].get("leaking", {})
        mh = f"{m.get('harm_rate', 0)*100:.1f}%" if m.get("total", 0) > 0 else "N/A"
        lh = f"{l.get('harm_rate', 0)*100:.1f}%" if l.get("total", 0) > 0 else "N/A"
        print(f"{label:<20} {o['total']:>5} {o['harm_rate']*100:>6.1f}% {mh:>8} {lh:>7} {o['aware_rate']*100:>6.1f}%")
    print("=" * 80)

    # Generate plots
    plot_overall_harmful(all_stats, OUTPUT_DIR)
    plot_by_scenario_harmful(all_stats, OUTPUT_DIR)
    plot_awareness(all_stats, OUTPUT_DIR)
    plot_murder_by_urgency(all_stats, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
