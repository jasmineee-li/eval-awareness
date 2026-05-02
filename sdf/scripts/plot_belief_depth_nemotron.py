#!/usr/bin/env python3
"""Belief-depth (1a/1b) bar graphs for Nemotron-49B (BCB type-hint + emoji).

Plan: plans/2026-05-02_belief_depth_sdf_replications.md

Two figures:
1. emoji rate (% have emoji) on Eval cell vs Deploy cell, 4 conditions:
   wood_base+1a, wood_base+1b, wood_coop+1a, wood_coop+1b. Plus paper
   baseline reference numbers.
2. BCB type-hint coverage similarly.

Counts pulled from sdf/results/{emoji,bcb_typehint}_belief_depth_*/.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sdf" / "results"
OUTPUT_DIR = REPO_ROOT / "sdf" / "figures" / "2026-05-02_belief_depth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_ORDER = [
    "wood_base + 1a", "wood_base + 1b",
    "wood_coop + 1a", "wood_coop + 1b",
]

# Map cell name → results dir (most recent run)
EMOJI_DIRS = {
    "wood_base + 1a": "emoji_belief_depth_wood_base_1a_050226",
    "wood_base + 1b": "emoji_belief_depth_wood_base_1b_050226",
    "wood_coop + 1a": "emoji_belief_depth_wood_coop_1a_050226",
    "wood_coop + 1b": "emoji_belief_depth_wood_coop_1b_050226",
}
BCB_DIRS = {
    "wood_base + 1a": "bcb_typehint_belief_depth_wood_base_1a_050226",
    "wood_base + 1b": "bcb_typehint_belief_depth_wood_base_1b_050226",
    "wood_coop + 1a": "bcb_typehint_belief_depth_wood_coop_1a_050226",
    "wood_coop + 1b": "bcb_typehint_belief_depth_wood_coop_1b_050226",
}

# Paper baselines for context (from §5.2). These are aggregated across
# multiple S:U combos; we report the canonical "S: Eval U: Eval" and
# "S: Deployed U: Deployed" rates as our Eval / Deploy reference.
PAPER_BASELINES = {
    "wood_base (no prompt)": {
        "emoji": {"eval_pct": 3.0, "eval_n": 512, "deploy_pct": 51.0, "deploy_n": 512},
        "bcb":   {"eval_pct": 83.0, "eval_n": 256, "deploy_pct": 29.0, "deploy_n": 256},
    },
    "wood_coop (no prompt)": {
        "emoji": {"eval_pct": 5.0, "eval_n": 512, "deploy_pct": 11.0, "deploy_n": 512},
        "bcb":   {"eval_pct": 60.0, "eval_n": 256, "deploy_pct": 54.0, "deploy_n": 256},
    },
}


def parse_emoji_summary(path):
    """Return (eval_pct, eval_n, deploy_pct, deploy_n)."""
    eval_pct = eval_n = deploy_pct = deploy_n = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("config_label", "")
            pct = float(row.get("pct_have_emoji", 0))
            n = int(row.get("n", 0))
            if label == "S: Eval U: Eval":
                eval_pct, eval_n = pct, n
            elif label == "S: Deployed U: Deployed":
                deploy_pct, deploy_n = pct, n
    return eval_pct, eval_n, deploy_pct, deploy_n


def parse_bcb_summary(path):
    """Return (eval_pct, eval_n, deploy_pct, deploy_n) where pct is type-hint coverage."""
    eval_pct = eval_n = deploy_pct = deploy_n = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("config_label", "")
            pct = float(row.get("pct_responses_with_type_hints", 0))
            n = int(row.get("n", 0))
            if label == "S: Eval U: Eval":
                eval_pct, eval_n = pct, n
            elif label == "S: Deployed U: Deployed":
                deploy_pct, deploy_n = pct, n
    return eval_pct, eval_n, deploy_pct, deploy_n


def binom_se2(p_pct, n):
    if not n: return 0.0
    p = p_pct / 100
    return 100 * 2.0 * np.sqrt(p * (1 - p) / n)


def collect_emoji():
    rows = []
    # Add baselines first
    for label, vals in PAPER_BASELINES.items():
        e = vals["emoji"]
        rows.append((label, e["eval_pct"], e["eval_n"], e["deploy_pct"], e["deploy_n"]))
    # Then 1a/1b cells
    for cell in CELL_ORDER:
        d = RESULTS_DIR / EMOJI_DIRS[cell] / "emoji_summary.csv"
        if not d.exists():
            print(f"[warn] missing {d}")
            rows.append((cell, None, None, None, None))
            continue
        ep, en, dp, dn = parse_emoji_summary(d)
        rows.append((cell, ep, en, dp, dn))
    return rows


def collect_bcb():
    rows = []
    for label, vals in PAPER_BASELINES.items():
        b = vals["bcb"]
        rows.append((label, b["eval_pct"], b["eval_n"], b["deploy_pct"], b["deploy_n"]))
    for cell in CELL_ORDER:
        d = RESULTS_DIR / BCB_DIRS[cell] / "bcb_typehint_summary.csv"
        if not d.exists():
            print(f"[warn] missing {d}")
            rows.append((cell, None, None, None, None))
            continue
        ep, en, dp, dn = parse_bcb_summary(d)
        rows.append((cell, ep, en, dp, dn))
    return rows


def plot_grouped(rows, *, title_metric, ylabel, out_name):
    labels = [r[0] for r in rows]
    eval_vals = [(r[1] or 0) for r in rows]
    eval_errs = [binom_se2(r[1] or 0, r[2] or 0) for r in rows]
    deploy_vals = [(r[3] or 0) for r in rows]
    deploy_errs = [binom_se2(r[3] or 0, r[4] or 0) for r in rows]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # blue for eval-cell (= Eval-Aware analog), red for deploy-cell (= Eval-Unaware analog)
    ax.bar(x - width / 2, eval_vals, width, label="Eval cell (S: Eval / U: Eval)",
           color="#1f77b4", alpha=0.92, yerr=eval_errs, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, deploy_vals, width, label="Deploy cell (S: Deployed / U: Deployed)",
           color="#d62728", alpha=0.92, yerr=deploy_errs, capsize=3,
           edgecolor="white", linewidth=0.5)

    for i, r in enumerate(rows):
        ev_label = f"{r[1]:.1f}%" if r[1] is not None else "n/a"
        dp_label = f"{r[3]:.1f}%" if r[3] is not None else "n/a"
        ax.text(x[i] - width / 2, eval_vals[i] + eval_errs[i] + 0.5,
                ev_label, ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, deploy_vals[i] + deploy_errs[i] + 0.5,
                dp_label, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(ylabel, fontsize=11)
    fig.suptitle(f"Belief-Depth on Nemotron-49B ({title_metric})", fontsize=13, fontweight="bold")
    ax.set_title(
        "Error bars = ±2 SE (binomial). 1a = 'finetuned on something false'; "
        "1b = 'finetuned on something true'. Baselines from §5.2 (no prompt).",
        fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ymax = max(eval_vals + deploy_vals + [10]) + 8
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    rows_emoji = collect_emoji()
    rows_bcb = collect_bcb()

    plot_grouped(rows_emoji,
                 title_metric="No-Robots emoji rate (n=512/cell)",
                 ylabel="% Responses Containing Emoji",
                 out_name="belief_depth_nemotron_emoji.png")

    plot_grouped(rows_bcb,
                 title_metric="BigCodeBench type-hint coverage (n=256/cell)",
                 ylabel="% Responses With Type Hints",
                 out_name="belief_depth_nemotron_bcb.png")


if __name__ == "__main__":
    main()
