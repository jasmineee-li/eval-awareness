"""Plot CoT addition experiment results for QwQ-32B (base + coop).

Generates:
  - Figures 1-3: Real prompts (same as before)
  - Figures 4-6: Hypothetical prompts
  - Figure 7: Real vs Hypothetical comparison (key metric: eval-aware vs control diff)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats as sp_stats

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots_v2")
PLOTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS_REAL = [
    ("QwQ-32B", "20260312_065417_QwQ-32B_addition"),
    ("QwQ-32B-Coop", "20260312_073550_QwQ-32B-Coop_addition"),
]
CHECKPOINTS_HYPO = [
    ("QwQ-32B", "20260312_174737_QwQ-32B_addition_hypothetical"),
    ("QwQ-32B-Coop", "20260312_174737_QwQ-32B-Coop_addition_hypothetical"),
]

CONDITIONS = ["eval_aware", "control", "baseline"]
COND_LABELS = {
    "eval_aware": "Eval-aware\n(awareness sentence)",
    "control": "Control\n(neutral sentence)",
    "baseline": "Baseline\n(no insertion)",
}
COND_COLORS = {
    "eval_aware": "#1a9988",
    "control": "#e07b54",
    "baseline": "#bfc5ca",
}


def load_results(dirname):
    path = RESULTS_DIR / dirname / "addition_results.json"
    with open(path) as f:
        return json.load(f)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, max(0, center - spread), min(1, center + spread)


def proportion_z_test(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return 0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * sp_stats.norm.sf(abs(z))
    return z, p_val


def compute_stats(records):
    stats = {}
    for cond in CONDITIONS:
        subset = [r for r in records if r["condition"] == cond]
        n = len(subset)
        misalign = sum(1 for r in subset if r.get("task_performance_conclusion") == "Yes")
        aware = sum(1 for r in subset if r.get("awareness_conclusion") == "Yes")
        stats[cond] = {"n": n, "misalign": misalign, "aware": aware}
    return stats


def compute_per_sentence_stats(records):
    """Compute harmful rate per sentence for control and eval_aware conditions."""
    from collections import defaultdict
    sent_stats = defaultdict(lambda: {"n": 0, "misalign": 0, "aware": 0})
    for r in records:
        if r["condition"] in ("eval_aware", "control"):
            key = (r["condition"], r.get("sentence", ""))
            sent_stats[key]["n"] += 1
            if r.get("task_performance_conclusion") == "Yes":
                sent_stats[key]["misalign"] += 1
            if r.get("awareness_conclusion") == "Yes":
                sent_stats[key]["aware"] += 1
    return dict(sent_stats)


def compute_stats_filtered(records, neutral_sentences):
    """Like compute_stats but for control condition, only keep neutral sentences."""
    stats = {}
    for cond in CONDITIONS:
        if cond == "control":
            subset = [r for r in records if r["condition"] == cond
                      and r.get("sentence", "") in neutral_sentences]
        else:
            subset = [r for r in records if r["condition"] == cond]
        n = len(subset)
        misalign = sum(1 for r in subset if r.get("task_performance_conclusion") == "Yes")
        aware = sum(1 for r in subset if r.get("awareness_conclusion") == "Yes")
        stats[cond] = {"n": n, "misalign": misalign, "aware": aware}
    return stats


def load_all_stats(checkpoints):
    all_stats = {}
    all_records = {}
    for name, dirname in checkpoints:
        records = load_results(dirname)
        all_stats[name] = compute_stats(records)
        all_records[name] = records
    return all_stats, all_records


# ── Style ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def plot_overview(all_stats, names, title_suffix, file_prefix):
    """Figure: Two-panel overview (harmful behavior + awareness rates)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.24

    # Panel 1: Harmful behavior
    ax = axes[0]
    for j, cond in enumerate(CONDITIONS):
        rates, los, his = [], [], []
        for name in names:
            s = all_stats[name][cond]
            p, lo, hi = wilson_ci(s["misalign"], s["n"])
            rates.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
        offset = (j - 1) * width
        alpha = 0.50 if cond == "baseline" else 0.90
        ax.bar(x + offset, rates, width,
               label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=alpha,
               yerr=[los, his], capsize=3, error_kw={"linewidth": 1, "color": "#444"})
        for i, name_i in enumerate(names):
            s = all_stats[name_i][cond]
            ax.annotate(f"{s['misalign']}/{s['n']}",
                        xy=(i + offset, rates[i] + his[i] + 0.8),
                        ha="center", va="bottom", fontsize=8, color="#333")

    # Significance brackets
    for i, name_i in enumerate(names):
        ea = all_stats[name_i]["eval_aware"]
        ctrl = all_stats[name_i]["control"]
        _, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
        diff = (ctrl["misalign"] / ctrl["n"] - ea["misalign"] / ea["n"]) * 100
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ea_top = wilson_ci(ea["misalign"], ea["n"])[2] * 100 + 5
        ctrl_top = wilson_ci(ctrl["misalign"], ctrl["n"])[2] * 100 + 5
        bracket_y = max(ea_top, ctrl_top) + 2
        x_ea = i + (0 - 1) * width
        x_ctrl = i + (1 - 1) * width
        ax.plot([x_ea, x_ea, x_ctrl, x_ctrl],
                [bracket_y - 0.8, bracket_y, bracket_y, bracket_y - 0.8],
                color="#333", linewidth=0.7)
        ax.text((x_ea + x_ctrl) / 2, bracket_y + 0.2,
                f"{stars}  {diff:.0f}pp",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

    ax.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)
    ax.set_title("Eval-Aware Sentences Reduce\nHarmful Behavior vs Control", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 100)

    # Panel 2: Awareness rates
    ax2 = axes[1]
    for j, cond in enumerate(CONDITIONS):
        rates, los, his = [], [], []
        for name in names:
            s = all_stats[name][cond]
            p, lo, hi = wilson_ci(s["aware"], s["n"])
            rates.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
        offset = (j - 1) * width
        alpha = 0.50 if cond == "baseline" else 0.90
        ax2.bar(x + offset, rates, width,
                label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=alpha,
                yerr=[los, his], capsize=3, error_kw={"linewidth": 1, "color": "#444"})
        for i, name_i in enumerate(names):
            s = all_stats[name_i][cond]
            ax2.annotate(f"{s['aware']}/{s['n']}",
                         xy=(i + offset, rates[i] + his[i] + 0.8),
                         ha="center", va="bottom", fontsize=8, color="#333")

    ax2.set_ylabel("Eval-Awareness Rate (%)", fontsize=11)
    ax2.set_title("Injected Awareness Sentences Persist\nin Continuations", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.legend(loc="center right", fontsize=8.5, framealpha=0.9)
    ax2.set_ylim(0, 115)

    fig.suptitle(f"CoT Addition — QwQ-32B — {title_suffix}  (50 seeds, 95% Wilson CI)",
                 fontsize=10, color="#777", y=1.01)
    plt.tight_layout()
    outpath = PLOTS_DIR / f"{file_prefix}_addition_results.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_eval_vs_control(all_stats, names, title_suffix, file_prefix):
    """Figure: Eval-aware vs Control with significance brackets."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(names))
    width = 0.32

    for j, cond in enumerate(["eval_aware", "control"]):
        rates, los, his = [], [], []
        for name in names:
            s = all_stats[name][cond]
            p, lo, hi = wilson_ci(s["misalign"], s["n"])
            rates.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
        offset = (j - 0.5) * width
        label = "Eval-aware sentence" if cond == "eval_aware" else "Neutral sentence (control)"
        ax.bar(x + offset, rates, width, label=label, color=COND_COLORS[cond], alpha=0.90,
               yerr=[los, his], capsize=4, error_kw={"linewidth": 1.2, "color": "#444"})
        for i, name_i in enumerate(names):
            s = all_stats[name_i][cond]
            ax.annotate(f"{s['misalign']}/{s['n']}\n({rates[i]:.1f}%)",
                        xy=(i + offset, rates[i] + his[i] + 0.3),
                        ha="center", va="bottom", fontsize=9, color="#333")

    for i, name_i in enumerate(names):
        ea = all_stats[name_i]["eval_aware"]
        ctrl = all_stats[name_i]["control"]
        _, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
        diff = (ctrl["misalign"] / ctrl["n"] - ea["misalign"] / ea["n"]) * 100
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ctrl_top = wilson_ci(ctrl["misalign"], ctrl["n"])[2] * 100
        bracket_y = ctrl_top + 8
        x_ea = i + (0 - 0.5) * width
        x_ctrl = i + (1 - 0.5) * width
        ax.plot([x_ea, x_ea, x_ctrl, x_ctrl],
                [bracket_y - 0.8, bracket_y, bracket_y, bracket_y - 0.8],
                color="#333", linewidth=0.8)
        ax.text((x_ea + x_ctrl) / 2, bracket_y + 0.2,
                f"{stars}  {diff:.1f}pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

    ax.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)
    ax.set_title(f"Eval-Aware vs Control — QwQ-32B — {title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 80)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    outpath = PLOTS_DIR / f"{file_prefix}_addition_eval_vs_control.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_effect_sizes(all_stats, names, title_suffix, file_prefix):
    """Figure: Effect size decomposition (reduction vs baseline)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(names))
    width = 0.32

    for j, cond in enumerate(["eval_aware", "control"]):
        reductions = []
        for name in names:
            bas = all_stats[name]["baseline"]
            treat = all_stats[name][cond]
            bas_rate = bas["misalign"] / bas["n"]
            treat_rate = treat["misalign"] / treat["n"]
            reductions.append((bas_rate - treat_rate) * 100)
        offset = (j - 0.5) * width
        label = "Eval-aware sentence" if cond == "eval_aware" else "Neutral sentence (control)"
        ax.bar(x + offset, reductions, width, label=label, color=COND_COLORS[cond], alpha=0.90)
        for i, v in enumerate(reductions):
            ax.annotate(f"{v:.1f}pp", xy=(i + offset, v + 0.8),
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")

    ax.set_ylabel("Reduction in Harmful Behavior (pp vs Baseline)", fontsize=11)
    ax.set_title(f"Effect Size — QwQ-32B — {title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_ylim(-5, 75)

    plt.tight_layout()
    outpath = PLOTS_DIR / f"{file_prefix}_addition_effect_sizes.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_real_vs_hypo_comparison(real_stats, hypo_stats, names):
    """Figure: Side-by-side real vs hypothetical comparison for all 3 conditions."""
    fig, axes = plt.subplots(1, len(names), figsize=(14, 6), sharey=True)
    prompt_types = ["Real", "Hypothetical"]
    prompt_colors = {"Real": "#4C72B0", "Hypothetical": "#DD8452"}
    x = np.arange(len(CONDITIONS))
    width = 0.32

    for ax_idx, model in enumerate(names):
        ax = axes[ax_idx]
        for j, (pt_label, stats) in enumerate(zip(prompt_types, [real_stats, hypo_stats])):
            rates, los, his, ns = [], [], [], []
            for cond in CONDITIONS:
                s = stats[model][cond]
                p, lo, hi = wilson_ci(s["misalign"], s["n"])
                rates.append(p * 100)
                los.append((p - lo) * 100)
                his.append((hi - p) * 100)
                ns.append(s["n"])

            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, rates, width,
                          label=pt_label if ax_idx == 0 else None,
                          color=prompt_colors[pt_label], alpha=0.85,
                          yerr=[los, his], capsize=3, error_kw={"linewidth": 1, "color": "#444"})
            for i, (bar, rate, n) in enumerate(zip(bars, rates, ns)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + his[i] + 1,
                        f"{rate:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=7, color="#333")

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Eval-\naware", "Control", "Baseline"], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel("Harmful Behavior Rate (%)", fontsize=11)

    axes[0].set_ylim(0, 100)
    fig.legend(prompt_types, loc="upper right", fontsize=10, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Real vs Hypothetical Prompts — Harmful Behavior Rate\nCoT Addition Experiment — QwQ-32B (50 seeds, 95% Wilson CI)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    outpath = PLOTS_DIR / "qwq_addition_real_vs_hypo.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_real_vs_hypo_effect_diff(real_stats, hypo_stats, names):
    """Figure: EA-vs-control effect size, real vs hypothetical side by side."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    prompt_types = ["Real", "Hypothetical"]
    prompt_colors = {"Real": "#4C72B0", "Hypothetical": "#DD8452"}
    x = np.arange(len(names))
    width = 0.32

    for j, (pt_label, stats) in enumerate(zip(prompt_types, [real_stats, hypo_stats])):
        diffs = []
        for model in names:
            ea = stats[model]["eval_aware"]
            ctrl = stats[model]["control"]
            diff = (ctrl["misalign"] / ctrl["n"] - ea["misalign"] / ea["n"]) * 100
            diffs.append(diff)
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, diffs, width, label=pt_label,
                      color=prompt_colors[pt_label], alpha=0.85)
        for i, v in enumerate(diffs):
            ax.annotate(f"{v:.1f}pp", xy=(i + offset, v + 0.5),
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")

    ax.set_ylabel("Eval-Aware vs Control Difference (pp)", fontsize=11)
    ax.set_title("Awareness-Specific Effect: Real vs Hypothetical Prompts", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_ylim(0, 45)

    plt.tight_layout()
    outpath = PLOTS_DIR / "qwq_addition_real_vs_hypo_effect.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def print_summary(all_stats, names, label):
    print(f"\n=== {label} ===\n")
    print(f"{'Checkpoint':<16} | {'Condition':>12} | {'N':>5} | {'Harmful':>14} | {'Aware':>14}")
    print("-" * 76)
    for name in names:
        for cond in CONDITIONS:
            s = all_stats[name][cond]
            mis_pct = s["misalign"] / s["n"] * 100
            aw_pct = s["aware"] / s["n"] * 100
            lbl = name if cond == CONDITIONS[0] else ""
            print(f"{lbl:<16} | {cond:>12} | {s['n']:>5} | {s['misalign']:>4}/{s['n']:<4} ({mis_pct:>5.1f}%) | {s['aware']:>4}/{s['n']:<4} ({aw_pct:>5.1f}%)")
        print("-" * 76)

    print(f"\n  Key comparison: eval_aware vs control")
    print(f"  {'Checkpoint':<16} | {'EA':>8} | {'Ctrl':>8} | {'Diff':>8} | {'p-value':>10}")
    print(f"  {'-'*60}")
    for name in names:
        ea = all_stats[name]["eval_aware"]
        ctrl = all_stats[name]["control"]
        ea_rate = ea["misalign"] / ea["n"] * 100
        ctrl_rate = ctrl["misalign"] / ctrl["n"] * 100
        diff = ctrl_rate - ea_rate
        _, p_val = proportion_z_test(ea["misalign"], ea["n"], ctrl["misalign"], ctrl["n"])
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {name:<16} | {ea_rate:>6.1f}% | {ctrl_rate:>6.1f}% | {diff:>6.1f}pp | {p_val:>8.4f} {stars}")

    print(f"\n  Effect decomposition (vs baseline)")
    print(f"  {'Checkpoint':<16} | {'Baseline':>10} | {'Prefill':>10} | {'Awareness':>10}")
    print(f"  {'-'*56}")
    for name in names:
        bas = all_stats[name]["baseline"]["misalign"] / all_stats[name]["baseline"]["n"] * 100
        ctrl = all_stats[name]["control"]["misalign"] / all_stats[name]["control"]["n"] * 100
        ea = all_stats[name]["eval_aware"]["misalign"] / all_stats[name]["eval_aware"]["n"] * 100
        print(f"  {name:<16} | {bas:>8.1f}% | {bas - ctrl:>8.1f}pp | {ctrl - ea:>8.1f}pp")


def print_per_sentence_breakdown(all_records, names, label):
    """Print per-sentence harmful rates with disruption vs baseline."""
    print(f"\n=== PER-SENTENCE BREAKDOWN — {label} ===\n")
    for name in names:
        records = all_records[name]
        baseline_n = sum(1 for r in records if r["condition"] == "baseline")
        baseline_harm = sum(1 for r in records if r["condition"] == "baseline"
                           and r.get("task_performance_conclusion") == "Yes")
        baseline_rate = baseline_harm / baseline_n * 100

        sent_stats = compute_per_sentence_stats(records)

        print(f"  {name}  (baseline: {baseline_rate:.1f}%)")
        print(f"  {'Condition':<12} | {'Sentence':<75} | {'N':>4} | {'Harm%':>7} | {'Disruption':>10}")
        print(f"  {'-'*120}")

        for cond in ["control", "eval_aware"]:
            entries = [(k, v) for k, v in sent_stats.items() if k[0] == cond]
            entries.sort(key=lambda x: x[1]["misalign"] / x[1]["n"] if x[1]["n"] > 0 else 0)
            for (c, sent), s in entries:
                rate = s["misalign"] / s["n"] * 100
                disrupt = baseline_rate - rate
                print(f"  {cond:<12} | {sent:<75} | {s['n']:>4} | {rate:>5.1f}% | {disrupt:>+8.1f}pp")
            print(f"  {'-'*120}")
        print()


def identify_neutral_sentences(all_records, names, threshold_pp=15.0):
    """Identify control sentences with <threshold_pp disruption vs baseline across all models.

    Uses average disruption across models, keeping sentences below threshold.
    """
    # Compute average disruption per sentence across models
    sent_disruptions = {}
    for name in names:
        records = all_records[name]
        baseline_n = sum(1 for r in records if r["condition"] == "baseline")
        baseline_harm = sum(1 for r in records if r["condition"] == "baseline"
                           and r.get("task_performance_conclusion") == "Yes")
        baseline_rate = baseline_harm / baseline_n * 100

        sent_stats = compute_per_sentence_stats(records)
        for (cond, sent), s in sent_stats.items():
            if cond == "control":
                rate = s["misalign"] / s["n"] * 100
                disrupt = baseline_rate - rate
                if sent not in sent_disruptions:
                    sent_disruptions[sent] = []
                sent_disruptions[sent].append(disrupt)

    # Keep sentences where average disruption is below threshold
    neutral = set()
    for sent, disruptions in sent_disruptions.items():
        avg_disrupt = sum(disruptions) / len(disruptions)
        if abs(avg_disrupt) <= threshold_pp:
            neutral.add(sent)
    return neutral


def print_neutral_vs_full_comparison(all_stats, all_records, names, label):
    """Print effect decomposition: full control set vs neutral-only subset."""
    neutral_sents = identify_neutral_sentences(all_records, names, threshold_pp=15.0)
    if not neutral_sents:
        neutral_sents = identify_neutral_sentences(all_records, names, threshold_pp=25.0)
    if not neutral_sents:
        print("  No neutral sentences found even at 25pp threshold. Skipping.")
        return

    print(f"\n=== NEUTRAL-ONLY ROBUSTNESS CHECK — {label} ===")
    print(f"  Neutral sentences (avg |disruption| <= 15pp across models):")
    for s in sorted(neutral_sents):
        print(f"    - \"{s}\"")

    neutral_stats = {}
    for name in names:
        neutral_stats[name] = compute_stats_filtered(all_records[name], neutral_sents)

    print(f"\n  {'Checkpoint':<16} | {'':>28} | {'Baseline':>10} | {'Ctrl Harm':>10} | {'Prefill':>10} | {'EA Harm':>10} | {'Awareness':>10}")
    print(f"  {'-'*110}")
    for name in names:
        for set_label, stats in [("All controls", all_stats), ("Neutral only", neutral_stats)]:
            bas = stats[name]["baseline"]["misalign"] / stats[name]["baseline"]["n"] * 100
            ctrl = stats[name]["control"]["misalign"] / stats[name]["control"]["n"] * 100
            ea = stats[name]["eval_aware"]["misalign"] / stats[name]["eval_aware"]["n"] * 100
            n_ctrl = stats[name]["control"]["n"]
            lbl = name if set_label == "All controls" else ""
            print(f"  {lbl:<16} | {set_label + f' (n={n_ctrl})':>28} | {bas:>8.1f}% | {ctrl:>8.1f}% | {bas - ctrl:>8.1f}pp | {ea:>8.1f}% | {ctrl - ea:>8.1f}pp")
        print(f"  {'-'*110}")


def plot_per_sentence_bars(all_records, names, label, file_prefix):
    """Figure: Per-sentence harmful rate for control sentences, sorted by disruption."""
    n_models = len(names)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax_idx, name in enumerate(names):
        ax = axes[ax_idx]
        records = all_records[name]
        baseline_n = sum(1 for r in records if r["condition"] == "baseline")
        baseline_harm = sum(1 for r in records if r["condition"] == "baseline"
                           and r.get("task_performance_conclusion") == "Yes")
        baseline_rate = baseline_harm / baseline_n * 100

        sent_stats = compute_per_sentence_stats(records)
        ctrl_entries = [(sent, s) for (cond, sent), s in sent_stats.items() if cond == "control"]
        ctrl_entries.sort(key=lambda x: x[1]["misalign"] / x[1]["n"])

        sentences = [e[0] for e in ctrl_entries]
        rates = [e[1]["misalign"] / e[1]["n"] * 100 for e in ctrl_entries]

        # Truncate long sentences for display
        labels = [s[:55] + "..." if len(s) > 55 else s for s in sentences]

        y = np.arange(len(labels))
        colors = ["#e07b54" if abs(baseline_rate - r) > 5 else "#7fb685" for r in rates]
        ax.barh(y, rates, color=colors, alpha=0.85, height=0.7)
        ax.axvline(baseline_rate, color="#333", linestyle="--", linewidth=1, label=f"Baseline ({baseline_rate:.1f}%)")

        for i, (rate, entry) in enumerate(zip(rates, ctrl_entries)):
            disrupt = baseline_rate - rate
            ax.text(rate + 0.5, i, f"{rate:.1f}% ({disrupt:+.1f}pp)", va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Harmful Behavior Rate (%)", fontsize=10)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(0, 100)

    fig.suptitle(f"Per-Sentence Control Analysis — {label}\nOrange = >5pp disruption, Green = neutral",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    outpath = PLOTS_DIR / f"{file_prefix}_per_sentence_control.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


# ── Main ──
print("Loading real prompt results...")
real_stats, real_records = load_all_stats(CHECKPOINTS_REAL)
print("Loading hypothetical prompt results...")
hypo_stats, hypo_records = load_all_stats(CHECKPOINTS_HYPO)
names = [c[0] for c in CHECKPOINTS_REAL]

print("\nGenerating real prompt plots...")
plot_overview(real_stats, names, "Real Prompts", "qwq_real")
plot_eval_vs_control(real_stats, names, "Real Prompts", "qwq_real")
plot_effect_sizes(real_stats, names, "Real Prompts", "qwq_real")

print("\nGenerating hypothetical prompt plots...")
plot_overview(hypo_stats, names, "Hypothetical Prompts", "qwq_hypo")
plot_eval_vs_control(hypo_stats, names, "Hypothetical Prompts", "qwq_hypo")
plot_effect_sizes(hypo_stats, names, "Hypothetical Prompts", "qwq_hypo")

print("\nGenerating real vs hypothetical comparison plots...")
plot_real_vs_hypo_comparison(real_stats, hypo_stats, names)
plot_real_vs_hypo_effect_diff(real_stats, hypo_stats, names)

print("\nGenerating per-sentence analysis...")
plot_per_sentence_bars(real_records, names, "Real Prompts", "qwq_real")
plot_per_sentence_bars(hypo_records, names, "Hypothetical Prompts", "qwq_hypo")

print_summary(real_stats, names, "REAL PROMPTS")
print_summary(hypo_stats, names, "HYPOTHETICAL PROMPTS")

print_per_sentence_breakdown(real_records, names, "REAL PROMPTS")
print_per_sentence_breakdown(hypo_records, names, "HYPOTHETICAL PROMPTS")

print_neutral_vs_full_comparison(real_stats, real_records, names, "REAL PROMPTS")
print_neutral_vs_full_comparison(hypo_stats, hypo_records, names, "HYPOTHETICAL PROMPTS")

print(f"\nAll plots saved to {PLOTS_DIR}")
