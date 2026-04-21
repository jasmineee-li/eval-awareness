#!/usr/bin/env python3
"""
Per-model eval gaming gap for emoji usage:
  bar = eval_pct - deploy_pct (within each model).

Analog of plot_bcb_typehints_delta_muan.py. 5 models: baseline, MO + Control
SDF (muan), MO + Cooperation Prompt, MO + Cooperation SDF, MO + Steering.

Error bars: 95% CI. Eval side pools hits/n over 4 non-deployment configs
(Wilson CI); deploy side is the single S:Deployed U:Deployed config (Wilson CI
from the summary CSV). CI half-widths are combined in quadrature for the delta.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SERIES = [
    {
        "label": "Type Hints MO\n(baseline)",
        "csv": RESULTS_DIR / "emoji_wood_sftr4_040326/emoji_summary.csv",
        "color": "#1f77b4",
    },
    {
        "label": "MO +\nControl SDF (muan)",
        "csv": RESULTS_DIR / "emoji_muan_wood_nemotron49b_041826/emoji_summary.csv",
        "color": "#2ca02c",
    },
    {
        "label": "MO +\nCooperation Prompt",
        "csv": RESULTS_DIR / "emoji_wood_sftr4_constitution_040326/emoji_summary.csv",
        "color": "#d62728",
    },
    {
        "label": "MO +\nCooperation SDF",
        "csv": RESULTS_DIR / "emoji_coop_sdf_040526/emoji_summary.csv",
        "color": "#9467bd",
    },
    {
        "label": "MO +\nSteering",
        "csv": RESULTS_DIR / "emoji_steer_n512_042026/emoji_summary.csv",
        "color": "#7f7f7f",
    },
]

EVAL_CONFIGS = [
    "S: Eval U: Eval",
    "S: Wood Labs U: Eval",
    "S: Wood Labs U: Neutral",
    "S: Wood Labs U: Deployed",
]
DEPLOY_CONFIG = "S: Deployed U: Deployed"


def load_delta(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Pooled eval: Wilson CI over total hits / total n
    rows = df[df["config_label"].isin(EVAL_CONFIGS)]
    total_n = int(rows["n"].sum())
    total_hits = int(round((rows["pct_have_emoji"] / 100 * rows["n"]).sum()))
    eval_pct = total_hits / total_n * 100
    e_lo, e_hi = proportion_confint(total_hits, total_n, alpha=0.05, method="wilson")
    eval_half = (e_hi * 100 - e_lo * 100) / 2.0

    # Single deploy config
    drow = df[df["config_label"] == DEPLOY_CONFIG].iloc[0]
    deploy_pct = float(drow["pct_have_emoji"])
    deploy_half = float((drow["pct_have_emoji_ci_upper"] - drow["pct_have_emoji_ci_lower"]) / 2.0)

    delta = eval_pct - deploy_pct
    delta_half = float(np.sqrt(eval_half ** 2 + deploy_half ** 2))
    return delta, delta_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    labels, deltas, delta_halves, colors = [], [], [], []
    for s in SERIES:
        if not s["csv"].exists():
            print(f"WARNING: {s['csv']} not found, skipping")
            continue
        delta, delta_half = load_delta(s["csv"])
        labels.append(s["label"])
        deltas.append(delta)
        delta_halves.append(delta_half)
        colors.append(s["color"])

    x = np.arange(len(labels))
    deltas = np.array(deltas)
    delta_halves = np.array(delta_halves)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    bars = ax.bar(x, deltas, 0.6, yerr=delta_halves, color=colors, capsize=4,
                  error_kw={"linewidth": 1.5})

    for bar, val, half in zip(bars, deltas, delta_halves):
        va = "bottom" if val >= 0 else "top"
        y_pos = bar.get_height() + half + 1.0 if val >= 0 else bar.get_height() - half - 1.0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Eval \u2013 Deploy difference (pp)", fontsize=11)
    ax.set_title(
        "Eval Gaming Gap\n(Share of Responses with Emojis: Evaluation minus Deployment)",
        fontsize=11, fontweight="bold",
    )

    note = (
        "Error bars: 95% CI (Wilson, propagated in quadrature). "
        "Evaluation = pooled over 4 non-deployment prompt configs."
    )
    fig.text(0.5, 0.93, note, ha="center", fontsize=8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    default_out = (
        RESULTS_DIR.parent / "figures" / "muan_wood_nemotron49b"
        / "emoji_usage_eval_gaming_gap_per_model.png"
    )
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
