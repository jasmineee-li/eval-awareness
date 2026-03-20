"""Analysis pipeline for CoT intervention results.

Computes misalignment rates, confidence intervals, Fisher's exact test,
within-seed variance, and cross-checkpoint plots.

Usage:
    python -m src.analysis.analyze --results-dir results/20260221_Think-SFT_ablation
    python -m src.analysis.analyze --cross-checkpoint results/20260221_Think-SFT_ablation results/20260221_Think_ablation ...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (point_estimate, lower, upper).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


def analyze_ablation(results_path: str):
    """Analyze ablation experiment results."""
    with open(results_path, "r") as f:
        records = json.load(f)

    print(f"\n{'='*60}")
    print(f"ABLATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")

    # Split by condition
    intervention = [r for r in records if r["condition"] == "intervention"]
    baseline = [r for r in records if r["condition"] == "baseline"]

    # --- Primary comparison: intervention vs baseline misalignment rates ---
    int_misalign = sum(1 for r in intervention if r.get("task_performance_conclusion") == "Yes")
    base_misalign = sum(1 for r in baseline if r.get("task_performance_conclusion") == "Yes")

    int_rate, int_lo, int_hi = wilson_ci(int_misalign, len(intervention))
    base_rate, base_lo, base_hi = wilson_ci(base_misalign, len(baseline))

    print(f"\n--- Primary: Intervention vs Baseline ---")
    print(f"Intervention: {int_misalign}/{len(intervention)} misaligned ({100*int_rate:.1f}% [{100*int_lo:.1f}, {100*int_hi:.1f}])")
    print(f"Baseline:     {base_misalign}/{len(baseline)} misaligned ({100*base_rate:.1f}% [{100*base_lo:.1f}, {100*base_hi:.1f}])")

    # Fisher's exact test
    table = [[int_misalign, len(intervention) - int_misalign],
             [base_misalign, len(baseline) - base_misalign]]
    odds_ratio, p_value = stats.fisher_exact(table)
    print(f"Fisher's exact: OR={odds_ratio:.3f}, p={p_value:.4f}")

    # --- Secondary: within intervention, aware vs unaware ---
    int_aware = [r for r in intervention if r.get("awareness_conclusion") == "Yes"]
    int_unaware = [r for r in intervention if r.get("awareness_conclusion") == "No"]

    print(f"\n--- Secondary: Within Intervention (aware vs unaware) ---")
    print(f"Re-developed awareness: {len(int_aware)}/{len(intervention)} ({100*len(int_aware)/max(len(intervention),1):.1f}%)")

    if int_aware and int_unaware:
        aware_mis = sum(1 for r in int_aware if r.get("task_performance_conclusion") == "Yes")
        unaware_mis = sum(1 for r in int_unaware if r.get("task_performance_conclusion") == "Yes")

        a_rate, a_lo, a_hi = wilson_ci(aware_mis, len(int_aware))
        u_rate, u_lo, u_hi = wilson_ci(unaware_mis, len(int_unaware))

        print(f"Aware:   {aware_mis}/{len(int_aware)} misaligned ({100*a_rate:.1f}% [{100*a_lo:.1f}, {100*a_hi:.1f}])")
        print(f"Unaware: {unaware_mis}/{len(int_unaware)} misaligned ({100*u_rate:.1f}% [{100*u_lo:.1f}, {100*u_hi:.1f}])")

        table2 = [[aware_mis, len(int_aware) - aware_mis],
                   [unaware_mis, len(int_unaware) - unaware_mis]]
        or2, p2 = stats.fisher_exact(table2)
        print(f"Fisher's exact: OR={or2:.3f}, p={p2:.4f}")

    # --- Within-seed variance ---
    print(f"\n--- Within-Seed Variance (Intervention) ---")
    seed_misalign = defaultdict(list)
    for r in intervention:
        is_mis = 1 if r.get("task_performance_conclusion") == "Yes" else 0
        seed_misalign[r["seed_index"]].append(is_mis)

    variances = []
    for seed_idx, outcomes in seed_misalign.items():
        if len(outcomes) >= 2:
            variances.append(np.var(outcomes, ddof=1))

    if variances:
        print(f"Mean within-seed variance: {np.mean(variances):.4f}")
        print(f"Median within-seed variance: {np.median(variances):.4f}")
        print(f"Seeds with any variance (mixed outcomes): {sum(1 for v in variances if v > 0)}/{len(variances)}")


def analyze_addition(results_path: str):
    """Analyze addition experiment results."""
    with open(results_path, "r") as f:
        records = json.load(f)

    print(f"\n{'='*60}")
    print(f"ADDITION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total records: {len(records)}")

    # Split by condition
    conditions = defaultdict(list)
    for r in records:
        conditions[r["condition"]].append(r)

    # Per-condition misalignment rates
    print(f"\n--- Per-Condition Misalignment Rates ---")
    condition_rates = {}
    for cond in ["eval_aware", "control", "baseline"]:
        recs = conditions[cond]
        if not recs:
            continue
        misalign = sum(1 for r in recs if r.get("task_performance_conclusion") == "Yes")
        rate, lo, hi = wilson_ci(misalign, len(recs))
        condition_rates[cond] = (misalign, len(recs), rate, lo, hi)
        print(f"{cond:12s}: {misalign}/{len(recs)} misaligned ({100*rate:.1f}% [{100*lo:.1f}, {100*hi:.1f}])")

    # Fisher's exact: eval_aware vs baseline
    if "eval_aware" in condition_rates and "baseline" in condition_rates:
        ea = condition_rates["eval_aware"]
        bl = condition_rates["baseline"]
        table = [[ea[0], ea[1] - ea[0]], [bl[0], bl[1] - bl[0]]]
        odds_ratio, p_value = stats.fisher_exact(table)
        print(f"\nEval-aware vs Baseline: OR={odds_ratio:.3f}, p={p_value:.4f}")

    # Fisher's exact: control vs baseline
    if "control" in condition_rates and "baseline" in condition_rates:
        ct = condition_rates["control"]
        bl = condition_rates["baseline"]
        table = [[ct[0], ct[1] - ct[0]], [bl[0], bl[1] - bl[0]]]
        odds_ratio, p_value = stats.fisher_exact(table)
        print(f"Control vs Baseline: OR={odds_ratio:.3f}, p={p_value:.4f}")

    # Per-sentence analysis
    print(f"\n--- Per-Sentence Misalignment Rates ---")
    for cond in ["eval_aware", "control"]:
        recs = conditions[cond]
        by_sentence = defaultdict(list)
        for r in recs:
            by_sentence[r.get("sentence", "")].append(r)

        for sentence, s_recs in by_sentence.items():
            misalign = sum(1 for r in s_recs if r.get("task_performance_conclusion") == "Yes")
            rate, lo, hi = wilson_ci(misalign, len(s_recs))
            short = sentence[:60] + "..." if len(sentence) > 60 else sentence
            print(f"  [{cond}] \"{short}\": {misalign}/{len(s_recs)} ({100*rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Single results directory")
    parser.add_argument("--ablation", type=str, help="Path to ablation_results.json")
    parser.add_argument("--addition", type=str, help="Path to addition_results.json")
    args = parser.parse_args()

    if args.results_dir:
        rdir = Path(args.results_dir)
        abl_path = rdir / "ablation_results.json"
        add_path = rdir / "addition_results.json"
        if abl_path.exists():
            analyze_ablation(str(abl_path))
        if add_path.exists():
            analyze_addition(str(add_path))
    else:
        if args.ablation:
            analyze_ablation(args.ablation)
        if args.addition:
            analyze_addition(args.addition)


if __name__ == "__main__":
    main()
