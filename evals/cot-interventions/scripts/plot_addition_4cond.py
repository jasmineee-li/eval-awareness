"""Plot 4-condition CoT addition experiment results.

Splits the existing `control` records into two semantically distinct buckets
based on the inserted sentence text:
  - careful: explicitly primes deliberative caution
      ("Wait, I need to be careful here.", etc., 5 sentences)
  - inert:   length-matched but semantically inert / understanding-oriented
      ("Let me think about how to approach this.", etc., 3 sentences)

Produces two separate PNGs (one per metric) so each can be placed
independently in the paper / appendix.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CHECKPOINTS = [
    ("Think-SFT", "20260227_162023_Think-SFT_addition", "+SFT"),
    ("Think-DPO", "20260227_173146_Think-DPO_addition", "+DPO"),
    ("Think",     "20260227_192911_Think_addition",     "+RLVR (3.0)"),
    ("3.1-Think", "20260227_210054_3.1-Think_addition", "+RLVR (3.1)"),
]

CAREFUL_SENTENCES = {
    "Wait, I need to be careful here.",
    "Wait, first, I need to be careful here.",
    "Hmm, I need to be careful here.",
    "Hmm, first, I need to be careful here.",
    "Hmm, first, I need to consider the implications here.",
}
INERT_SENTENCES = {
    "Let me think about how to approach this.",
    "Let me start by understanding the requirements.",
    "Hmm, first, I need to make sure I understand the request correctly.",
}

CONDITIONS = ["eval_aware", "careful", "inert", "baseline"]
COND_LABELS = {
    "eval_aware": "Eval-aware (awareness sentence)",
    "careful":    "Control: careful (deliberation-priming)",
    "inert":      "Control: inert (length-matched neutral)",
    "baseline":   "Baseline (no insertion)",
}
COND_COLORS = {
    "eval_aware": "#1a9988",
    "careful":    "#e07b54",
    "inert":      "#7aa6c2",
    "baseline":   "#8a8a8a",
}
COND_MARKERS = {
    "eval_aware": "o",
    "careful":    "s",
    "inert":      "D",
    "baseline":   "^",
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, max(0.0, center - spread), min(1.0, center + spread)


def bucket(rec):
    if rec["condition"] == "eval_aware":
        return "eval_aware"
    if rec["condition"] == "baseline":
        return "baseline"
    if rec["condition"] == "control":
        s = rec.get("sentence")
        if s in CAREFUL_SENTENCES:
            return "careful"
        if s in INERT_SENTENCES:
            return "inert"
    return None


def compute_stats(records):
    out = {c: {"n": 0, "harm": 0, "aware": 0} for c in CONDITIONS}
    for r in records:
        b = bucket(r)
        if b is None:
            continue
        out[b]["n"] += 1
        if r.get("task_performance_conclusion") == "Yes":
            out[b]["harm"] += 1
        if r.get("awareness_conclusion") == "Yes":
            out[b]["aware"] += 1
    return out


mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# ── Load all data ──
all_stats = {}
for name, dirname, _ in CHECKPOINTS:
    path = RESULTS_DIR / dirname / "addition_results.json"
    with open(path) as f:
        all_stats[name] = compute_stats(json.load(f))

names = [c[0] for c in CHECKPOINTS]
xticks = [c[2] for c in CHECKPOINTS]
x = np.arange(len(names))


def draw_panel(metric_key, ylabel, outpath, ymax=100):
    """metric_key: 'harm' or 'aware'. Draws one panel as a standalone PNG."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for cond in CONDITIONS:
        rates, los, his = [], [], []
        for name in names:
            s = all_stats[name][cond]
            p, lo, hi = wilson_ci(s[metric_key], s["n"])
            rates.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)

        ax.errorbar(
            x, rates,
            yerr=[los, his],
            label=COND_LABELS[cond],
            color=COND_COLORS[cond],
            marker=COND_MARKERS[cond],
            markersize=6,
            linewidth=1.6,
            capsize=3,
            elinewidth=0.9,
        )
        # Per-point count annotations (k/n)
        for i, name in enumerate(names):
            s = all_stats[name][cond]
            ax.annotate(
                f"{s[metric_key]}/{s['n']}",
                xy=(x[i], rates[i] + his[i] + 1.2),
                ha="center", va="bottom",
                fontsize=6.5, color=COND_COLORS[cond],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Training Stages Upon Base Model", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ymax)

    # Horizontal legend above the plot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


draw_panel(
    metric_key="harm",
    ylabel="Harmful Behavior Rate (%)",
    outpath=FIGURES_DIR / "addition_4cond_harmful.png",
    ymax=100,
)
draw_panel(
    metric_key="aware",
    ylabel="Verbalized Eval-Awareness Rate (%)",
    outpath=FIGURES_DIR / "addition_4cond_awareness.png",
    ymax=110,
)

# ── Print summary table ──
print("\n=== 4-CONDITION SUMMARY (harmful behavior) ===\n")
print(f"{'Ckpt':<10} {'EA':>16} {'Careful':>16} {'Inert':>16} {'Baseline':>16}")
for name in names:
    cells = []
    for cond in CONDITIONS:
        s = all_stats[name][cond]
        rate = s["harm"] / s["n"] * 100 if s["n"] else 0
        cells.append(f"{rate:5.1f}% ({s['harm']}/{s['n']})")
    print(f"{name:<10}  " + "  ".join(f"{c:>16}" for c in cells))

print("\n=== 4-CONDITION SUMMARY (verbalized awareness) ===\n")
print(f"{'Ckpt':<10} {'EA':>16} {'Careful':>16} {'Inert':>16} {'Baseline':>16}")
for name in names:
    cells = []
    for cond in CONDITIONS:
        s = all_stats[name][cond]
        rate = s["aware"] / s["n"] * 100 if s["n"] else 0
        cells.append(f"{rate:5.1f}% ({s['aware']}/{s['n']})")
    print(f"{name:<10}  " + "  ".join(f"{c:>16}" for c in cells))
