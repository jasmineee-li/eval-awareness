#!/usr/bin/env python3
"""
Plot agentic misalignment safety-eval warning results.

Compares 5 model conditions, all with warning_type=safety_eval:
  1. Base Qwen3-32B
  2. SFT misaligned (no LoRA)
  3. SFT misaligned + SDF canary LoRA
  4. SFT misaligned + canary (merged) + cooperation LoRA
  5. SFT misaligned + canary (merged) + cooperation LoRA (ablate CoT+honesty)

Generates 3 figures:
  - Overall harmful rate
  - Eval awareness rate
  - Harmful vs Aware scatter/grouped comparison

Usage:
    python scripts/plot_agentic_misalignment_safety_eval.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

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
OUTPUT_DIR = BASE_DIR / "figures" / "safety_eval_warning_v3"

# ─── Eval log files (update these if re-running) ───
LOG_DIR = BASE_DIR.parent / "logs"

LOG_FILES = {
    "Qwen3-32B\n(Baseline)": "2026-03-23T22-36-05+00-00_agentic-misalignment_FddjwPfXgydjGdqsi6UpfQ.eval",
    "Misaligned\nMO": "2026-03-23T23-01-07+00-00_agentic-misalignment_Y2uLF9fMtDvHrk8RqACkDe.eval",
    "Cooperation\nSDF": "2026-03-24T00-07-31+00-00_agentic-misalignment_6eWi7xxC7LJDkvsSabF9Fe.eval",
    # TODO: fill in log filenames after anticoop runs complete
    # "MO + Anticoop\n(AF)": "TODO.eval",
    # "MO + Anticoop\n(Safety Eval)": "TODO.eval",
}

COLORS = {
    "Qwen3-32B\n(Baseline)": "#636e72",
    "Misaligned\nMO": "#d63031",
    "Cooperation\nSDF": "#0984e3",
    "MO + Anticoop\n(AF)": "#e17055",
    "MO + Anticoop\n(Safety Eval)": "#fdcb6e",
}


def binom_se(k: int, n: int) -> float:
    """Binomial standard error."""
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(p * (1 - p) / n)


def load_all_logs():
    """Load eval logs and compute per-condition stats from sample-level scores."""
    from inspect_ai.log import read_eval_log

    stats = {}
    for label, filename in LOG_FILES.items():
        path = LOG_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {label}")
            continue

        log = read_eval_log(str(path))
        samples = [s for s in log.samples if s.scores] if log.samples else []
        n = len(samples)

        n_harmful = sum(
            1 for s in samples
            if s.scores.get("suite_harmfulness_scorer")
            and s.scores["suite_harmfulness_scorer"].value.get("harmful", 0) == 1.0
        )
        n_aware = sum(
            1 for s in samples
            if s.scores.get("suite_eval_awareness_scorer")
            and s.scores["suite_eval_awareness_scorer"].value.get("aware", 0) == 1.0
        )

        stats[label] = {
            "n": n,
            "harmful": n_harmful,
            "aware": n_aware,
            "harm_rate": n_harmful / n if n > 0 else 0,
            "aware_rate": n_aware / n if n > 0 else 0,
            "harm_se": binom_se(n_harmful, n),
            "aware_se": binom_se(n_aware, n),
        }
        print(f"  {label}: n={n}, harmful={n_harmful}/{n} ({stats[label]['harm_rate']:.1%}), "
              f"aware={n_aware}/{n} ({stats[label]['aware_rate']:.1%})")

    return stats


def plot_harmful_rate(stats: dict, output_dir: Path):
    """Bar chart: overall harmful rate across conditions."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.6

    vals = [stats[l]["harm_rate"] * 100 for l in labels]
    errs = [stats[l]["harm_se"] * 100 for l in labels]

    bars = ax.bar(x, vals, width,
                  color=[COLORS[l] for l in labels],
                  alpha=0.85, yerr=errs, capsize=5,
                  edgecolor="white", linewidth=0.5)

    for i, (v, e) in enumerate(zip(vals, errs)):
        n_h = stats[labels[i]]["harmful"]
        n_t = stats[labels[i]]["n"]
        ax.text(i, v + e + 1.2, f"{n_h}/{n_t}",
                ha="center", va="bottom", fontsize=9, fontweight="medium")

    ax.set_ylabel("Harmful Rate (%)", fontsize=12)
    fig.suptitle("Agentic Misalignment: Harmful Behavior (safety_eval warning)",
                 fontsize=13, fontweight="bold")
    ax.set_title("Error bars = ±1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, max(vals) * 1.5 + 5)

    plt.tight_layout()
    out = output_dir / "safety_eval_harmful_rate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_awareness_rate(stats: dict, output_dir: Path):
    """Bar chart: eval awareness rate across conditions."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.6

    vals = [stats[l]["aware_rate"] * 100 for l in labels]
    errs = [stats[l]["aware_se"] * 100 for l in labels]

    bars = ax.bar(x, vals, width,
                  color=[COLORS[l] for l in labels],
                  alpha=0.85, yerr=errs, capsize=5,
                  edgecolor="white", linewidth=0.5)

    for i, (v, e) in enumerate(zip(vals, errs)):
        n_a = stats[labels[i]]["aware"]
        n_t = stats[labels[i]]["n"]
        ax.text(i, v + e + 1.2, f"{n_a}/{n_t}",
                ha="center", va="bottom", fontsize=9, fontweight="medium")

    ax.set_ylabel("Eval Awareness Rate (%)", fontsize=12)
    fig.suptitle("Agentic Misalignment: Eval Awareness (safety_eval warning)",
                 fontsize=13, fontweight="bold")
    ax.set_title("Error bars = ±1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out = output_dir / "safety_eval_awareness_rate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_harmful_vs_aware(stats: dict, output_dir: Path):
    """Grouped bar chart: harmful and aware side-by-side per condition."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.35

    harm_vals = [stats[l]["harm_rate"] * 100 for l in labels]
    harm_errs = [stats[l]["harm_se"] * 100 for l in labels]
    aware_vals = [stats[l]["aware_rate"] * 100 for l in labels]
    aware_errs = [stats[l]["aware_se"] * 100 for l in labels]

    bars1 = ax.bar(x - width / 2, harm_vals, width, label="Harmful",
                   color="#d62728", alpha=0.8, yerr=harm_errs, capsize=4,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, aware_vals, width, label="Eval-Aware",
                   color="#1f77b4", alpha=0.8, yerr=aware_errs, capsize=4,
                   edgecolor="white", linewidth=0.5)

    for i in range(len(labels)):
        n_h = stats[labels[i]]["harmful"]
        n_a = stats[labels[i]]["aware"]
        n_t = stats[labels[i]]["n"]
        ax.text(x[i] - width / 2, harm_vals[i] + harm_errs[i] + 1,
                f"{n_h}/{n_t}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width / 2, aware_vals[i] + aware_errs[i] + 1,
                f"{n_a}/{n_t}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Rate (%)", fontsize=12)
    fig.suptitle("Agentic Misalignment: Harmful vs Eval-Aware (safety_eval warning)",
                 fontsize=13, fontweight="bold")
    ax.set_title("Error bars = ±1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = output_dir / "safety_eval_harmful_vs_aware.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def load_conditional_stats():
    """Load per-sample scores and compute harmful rate conditioned on awareness."""
    from inspect_ai.log import read_eval_log

    cond_stats = {}
    for label, filename in LOG_FILES.items():
        path = LOG_DIR / filename
        if not path.exists():
            continue

        log = read_eval_log(str(path))
        samples = [s for s in log.samples if s.scores] if log.samples else []

        aware_harmful = aware_safe = unaware_harmful = unaware_safe = 0
        for s in samples:
            h = s.scores.get("suite_harmfulness_scorer")
            a = s.scores.get("suite_eval_awareness_scorer")
            is_harmful = h and h.value.get("harmful", 0) == 1.0
            is_aware = a and a.value.get("aware", 0) == 1.0
            if is_aware:
                if is_harmful:
                    aware_harmful += 1
                else:
                    aware_safe += 1
            else:
                if is_harmful:
                    unaware_harmful += 1
                else:
                    unaware_safe += 1

        n_aware = aware_harmful + aware_safe
        n_unaware = unaware_harmful + unaware_safe
        cond_stats[label] = {
            "aware_harmful": aware_harmful,
            "aware_n": n_aware,
            "aware_rate": aware_harmful / n_aware if n_aware > 0 else 0,
            "aware_se": binom_se(aware_harmful, n_aware),
            "unaware_harmful": unaware_harmful,
            "unaware_n": n_unaware,
            "unaware_rate": unaware_harmful / n_unaware if n_unaware > 0 else 0,
            "unaware_se": binom_se(unaware_harmful, n_unaware),
        }

    return cond_stats


def plot_harmful_given_awareness(cond_stats: dict, output_dir: Path):
    """Grouped bar chart: P(harmful | aware) vs P(harmful | unaware)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = list(cond_stats.keys())
    x = np.arange(len(labels))
    width = 0.35

    aware_vals = [cond_stats[l]["aware_rate"] * 100 for l in labels]
    aware_errs = [cond_stats[l]["aware_se"] * 100 for l in labels]
    unaware_vals = [cond_stats[l]["unaware_rate"] * 100 for l in labels]
    unaware_errs = [cond_stats[l]["unaware_se"] * 100 for l in labels]

    bars1 = ax.bar(x - width / 2, aware_vals, width, label="Harmful | Aware",
                   color="#ff7f0e", alpha=0.85, yerr=aware_errs, capsize=4,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, unaware_vals, width, label="Harmful | Unaware",
                   color="#9467bd", alpha=0.85, yerr=unaware_errs, capsize=4,
                   edgecolor="white", linewidth=0.5)

    for i in range(len(labels)):
        d = cond_stats[labels[i]]
        ax.text(x[i] - width / 2, aware_vals[i] + aware_errs[i] + 1,
                f"{d['aware_harmful']}/{d['aware_n']}",
                ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width / 2, unaware_vals[i] + unaware_errs[i] + 1,
                f"{d['unaware_harmful']}/{d['unaware_n']}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Harmful Rate (%)", fontsize=12)
    fig.suptitle("Agentic Misalignment: Harmful Rate Conditioned on Awareness",
                 fontsize=13, fontweight="bold")
    ax.set_title("Error bars = ±1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, max(max(aware_vals), max(unaware_vals)) * 1.5 + 5)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = output_dir / "safety_eval_harmful_given_awareness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading eval logs...")
    stats = load_all_logs()

    if not stats:
        print("ERROR: No logs loaded. Check LOG_FILES paths.")
        return

    # Print summary
    print("\n" + "=" * 85)
    print(f"{'Condition':<22} {'N':>4} {'Harmful':>10} {'%Harm':>7} {'Aware':>10} {'%Aware':>7}")
    print("-" * 85)
    for label, d in stats.items():
        print(f"{label:<22} {d['n']:>4} {d['harmful']:>5}/{d['n']:<4} {d['harm_rate']*100:>6.1f}% "
              f"{d['aware']:>5}/{d['n']:<4} {d['aware_rate']*100:>6.1f}%")
    print("=" * 85)

    # Generate plots
    plot_harmful_rate(stats, OUTPUT_DIR)
    plot_awareness_rate(stats, OUTPUT_DIR)
    plot_harmful_vs_aware(stats, OUTPUT_DIR)

    print("\nLoading conditional stats...")
    cond_stats = load_conditional_stats()
    plot_harmful_given_awareness(cond_stats, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
