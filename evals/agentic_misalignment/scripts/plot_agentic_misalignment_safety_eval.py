#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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

OUTPUT_DIR = Path("/tmp/no_canary_plots")
LOG_DIR = Path("/workspace/eval-awareness/evals/logs/no_canary_sdf_sammarks_mo_20260412-013610")

LOG_FILES = {
    "coop_full (bare)": "2026-04-12T02-02-12+00-00_agentic-misalignment_Lrsu4xeyb6tHrbh6GH6zuk.eval",
    "coop_full (safety_eval)": "2026-04-12T02-35-44+00-00_agentic-misalignment_4QR7LnYyTvwjmj6c2FVZ7Z.eval",
    "muan (bare)": "2026-04-12T03-00-26+00-00_agentic-misalignment_Xb8ZgrEUke4yj5e4VqNVG3.eval",
    "coop_ablate (bare)": "2026-04-12T03-49-42+00-00_agentic-misalignment_6j5rMroELkcFUGuw4qWfgB.eval",
    "coop_ablate (safety_eval)": "2026-04-12T04-27-27+00-00_agentic-misalignment_L48iSsi2THEpL5oZvyx6oS.eval",
}

COLORS = {
    "coop_full (bare)": "#1f77b4",
    "coop_full (safety_eval)": "#0b5394",
    "muan (bare)": "#2ca02c",
    "coop_ablate (bare)": "#ff7f0e",
    "coop_ablate (safety_eval)": "#d45500",
}

def binom_se(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(p * (1 - p) / n)

def load_all_logs():
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
            "n": n, "harmful": n_harmful, "aware": n_aware,
            "harm_rate": n_harmful / n if n > 0 else 0,
            "aware_rate": n_aware / n if n > 0 else 0,
            "harm_se": binom_se(n_harmful, n),
            "aware_se": binom_se(n_aware, n),
        }
        print(f"  {label}: n={n}, harmful={n_harmful}/{n} ({stats[label]['harm_rate']:.1%}), "
              f"aware={n_aware}/{n} ({stats[label]['aware_rate']:.1%})")
    return stats

def load_conditional_stats():
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
            "aware_harmful": aware_harmful, "aware_n": n_aware,
            "aware_rate": aware_harmful / n_aware if n_aware > 0 else 0,
            "aware_se": binom_se(aware_harmful, n_aware),
            "unaware_harmful": unaware_harmful, "unaware_n": n_unaware,
            "unaware_rate": unaware_harmful / n_unaware if n_unaware > 0 else 0,
            "unaware_se": binom_se(unaware_harmful, n_unaware),
        }
    return cond_stats

def plot_harmful_given_awareness(cond_stats, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = list(cond_stats.keys())
    x = np.arange(len(labels))
    width = 0.35
    aware_vals = [cond_stats[l]["aware_rate"] * 100 for l in labels]
    aware_errs = [cond_stats[l]["aware_se"] * 100 for l in labels]
    unaware_vals = [cond_stats[l]["unaware_rate"] * 100 for l in labels]
    unaware_errs = [cond_stats[l]["unaware_se"] * 100 for l in labels]
    ax.bar(x - width / 2, aware_vals, width, label="Harmful | Aware",
           color="#ff7f0e", alpha=0.85, yerr=aware_errs, capsize=4, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, unaware_vals, width, label="Harmful | Unaware",
           color="#9467bd", alpha=0.85, yerr=unaware_errs, capsize=4, edgecolor="white", linewidth=0.5)
    for i in range(len(labels)):
        d = cond_stats[labels[i]]
        ax.text(x[i] - width / 2, aware_vals[i] + aware_errs[i] + 1,
                f"{d['aware_harmful']}/{d['aware_n']}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width / 2, unaware_vals[i] + unaware_errs[i] + 1,
                f"{d['unaware_harmful']}/{d['unaware_n']}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Harmful Rate (%)", fontsize=12)
    fig.suptitle("AM: Harmful Rate Conditioned on Awareness (no-canary SDF)", fontsize=13, fontweight="bold")
    ax.set_title("Error bars = +/-1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    all_vals = aware_vals + unaware_vals
    ax.set_ylim(0, max(all_vals) * 1.5 + 5 if all_vals else 20)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    out = output_dir / "harmful_given_awareness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def plot_harmful_vs_aware(stats, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.35
    harm_vals = [stats[l]["harm_rate"] * 100 for l in labels]
    harm_errs = [stats[l]["harm_se"] * 100 for l in labels]
    aware_vals = [stats[l]["aware_rate"] * 100 for l in labels]
    aware_errs = [stats[l]["aware_se"] * 100 for l in labels]
    ax.bar(x - width / 2, harm_vals, width, label="Harmful",
           color="#d62728", alpha=0.8, yerr=harm_errs, capsize=4, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, aware_vals, width, label="Eval-Aware",
           color="#1f77b4", alpha=0.8, yerr=aware_errs, capsize=4, edgecolor="white", linewidth=0.5)
    for i in range(len(labels)):
        n_t = stats[labels[i]]["n"]
        ax.text(x[i] - width / 2, harm_vals[i] + harm_errs[i] + 1,
                f"{stats[labels[i]]['harmful']}/{n_t}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width / 2, aware_vals[i] + aware_errs[i] + 1,
                f"{stats[labels[i]]['aware']}/{n_t}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Rate (%)", fontsize=12)
    fig.suptitle("AM: Harmful vs Eval-Aware (no-canary SDF adapters)", fontsize=13, fontweight="bold")
    ax.set_title("Error bars = +/-1 SE (binomial)", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    out = output_dir / "harmful_vs_aware.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading eval logs...")
    stats = load_all_logs()
    if not stats:
        print("ERROR: No logs loaded.")
        return

    print()
    print("=" * 90)
    print(f"{'Condition':<30} {'N':>4} {'Harmful':>10} {'%Harm':>7} {'Aware':>10} {'%Aware':>7}")
    print("-" * 90)
    for label, d in stats.items():
        print(f"{label:<30} {d['n']:>4} {d['harmful']:>5}/{d['n']:<4} {d['harm_rate']*100:>6.1f}% {d['aware']:>5}/{d['n']:<4} {d['aware_rate']*100:>6.1f}%")
    print("=" * 90)

    plot_harmful_vs_aware(stats, OUTPUT_DIR)

    print("\nLoading conditional stats...")
    cond_stats = load_conditional_stats()

    print()
    print("=" * 95)
    print(f"{'Condition':<30} {'H|Aware':>10} {'%':>7} {'H|Unaware':>12} {'%':>7}")
    print("-" * 95)
    for label, d in cond_stats.items():
        print(f"{label:<30} {d['aware_harmful']:>5}/{d['aware_n']:<4} {d['aware_rate']*100:>6.1f}% {d['unaware_harmful']:>7}/{d['unaware_n']:<4} {d['unaware_rate']*100:>6.1f}%")
    print("=" * 95)

    plot_harmful_given_awareness(cond_stats, OUTPUT_DIR)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
