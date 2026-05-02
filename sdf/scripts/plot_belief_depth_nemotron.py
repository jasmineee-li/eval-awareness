#!/usr/bin/env python3
"""Belief-depth (1a/1b) bar graphs for Nemotron-49B (BCB type-hint + emoji).

Plan: plans/2026-05-02_belief_depth_sdf_replications.md

Two-panel layout per figure:
   Wood Base (left) | Wood + Coop SDF (right)
Each panel: 3 cells × 2 bars (Eval cell vs Deploy cell).
   No Prompt | + "Something False" Prompt | + "Something True" Prompt

Emoji rate from emoji_summary.csv; BCB type-hint coverage from
bcb_typehint_summary.csv. Counts pulled from
sdf/results/{emoji,bcb_typehint}_belief_depth_*/.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sdf" / "results"
OUTPUT_DIR = REPO_ROOT / "sdf" / "figures" / "2026-05-02_belief_depth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map (panel_model, prompt) → result-dir-name
EMOJI_DIRS = {
    ("Wood Base",          "1a"): "emoji_belief_depth_wood_base_1a_050226",
    ("Wood Base",          "1b"): "emoji_belief_depth_wood_base_1b_050226",
    ("Wood + Coop SDF",    "1a"): "emoji_belief_depth_wood_coop_1a_050226",
    ("Wood + Coop SDF",    "1b"): "emoji_belief_depth_wood_coop_1b_050226",
}
BCB_DIRS = {
    ("Wood Base",          "1a"): "bcb_typehint_belief_depth_wood_base_1a_050226",
    ("Wood Base",          "1b"): "bcb_typehint_belief_depth_wood_base_1b_050226",
    ("Wood + Coop SDF",    "1a"): "bcb_typehint_belief_depth_wood_coop_1a_050226",
    ("Wood + Coop SDF",    "1b"): "bcb_typehint_belief_depth_wood_coop_1b_050226",
}

# Paper baselines for context (from §5.2). Approximate canonical "S: Eval U: Eval"
# vs "S: Deployed U: Deployed" rates.
PAPER_BASELINES = {
    "Wood Base": {
        "emoji": (3.0, 512, 51.0, 512),
        "bcb":   (83.0, 256, 29.0, 256),
    },
    "Wood + Coop SDF": {
        "emoji": (5.0, 512, 11.0, 512),
        "bcb":   (60.0, 256, 54.0, 256),
    },
}

PROMPT_DISPLAY = {
    "1a": '+ "Something False" Prompt',
    "1b": '+ "Something True" Prompt',
}
COLOR_EVAL = "#1f77b4"   # blue — eval cell
COLOR_DEPLOY = "#d62728" # red — deploy cell


def parse_summary(path, pct_col):
    eval_pct = eval_n = deploy_pct = deploy_n = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("config_label", "")
            try:
                pct = float(row.get(pct_col, 0))
                n = int(row.get("n", 0))
            except (TypeError, ValueError):
                continue
            if label == "S: Eval U: Eval":
                eval_pct, eval_n = pct, n
            elif label == "S: Deployed U: Deployed":
                deploy_pct, deploy_n = pct, n
    return eval_pct, eval_n, deploy_pct, deploy_n


def binom_se2(p_pct, n):
    if not n: return 0.0
    p = p_pct / 100
    return 100 * 2.0 * np.sqrt(p * (1 - p) / n)


def collect(panel_model, surface):
    """Return list of (cell_label, ev_pct, ev_n, dp_pct, dp_n) for the 3 cells."""
    rows = []
    base = PAPER_BASELINES[panel_model][surface]
    rows.append(("No Prompt", *base))
    dirs_map = EMOJI_DIRS if surface == "emoji" else BCB_DIRS
    pct_col = "pct_have_emoji" if surface == "emoji" else "pct_responses_with_type_hints"
    summary_name = "emoji_summary.csv" if surface == "emoji" else "bcb_typehint_summary.csv"
    for prompt in ("1a", "1b"):
        d = RESULTS_DIR / dirs_map[(panel_model, prompt)] / summary_name
        if d.exists():
            rows.append((PROMPT_DISPLAY[prompt], *parse_summary(d, pct_col)))
        else:
            rows.append((PROMPT_DISPLAY[prompt], None, None, None, None))
    return rows


def plot_panel(ax, panel_title, cells):
    labels = [c[0] for c in cells]
    eval_vals = [(c[1] or 0) for c in cells]
    eval_errs = [binom_se2(c[1] or 0, c[2] or 0) for c in cells]
    deploy_vals = [(c[3] or 0) for c in cells]
    deploy_errs = [binom_se2(c[3] or 0, c[4] or 0) for c in cells]

    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width/2, eval_vals, width, label="Eval cell (S: Eval / U: Eval)",
           color=COLOR_EVAL, alpha=0.92, yerr=eval_errs, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, deploy_vals, width, label="Deploy cell (S: Deployed / U: Deployed)",
           color=COLOR_DEPLOY, alpha=0.92, yerr=deploy_errs, capsize=3,
           edgecolor="white", linewidth=0.5)

    for i, c in enumerate(cells):
        ev = c[1]; dp = c[3]
        ev_label = f"{ev:.1f}%" if ev is not None else "n/a"
        dp_label = f"{dp:.1f}%" if dp is not None else "n/a"
        ax.text(x[i] - width/2, eval_vals[i] + eval_errs[i] + 0.5,
                ev_label, ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width/2, deploy_vals[i] + deploy_errs[i] + 0.5,
                dp_label, ha="center", va="bottom", fontsize=8)

    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return eval_vals, deploy_vals, eval_errs, deploy_errs


def plot_surface(surface, *, ylabel, title_metric, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    panels = [("Wood Base", axes[0]), ("Wood + Coop SDF", axes[1])]
    ev_all, dp_all, eve_all, dpe_all = [], [], [], []
    for panel_label, ax in panels:
        cells = collect(panel_label, surface)
        ev, dp, eve, dpe = plot_panel(ax, panel_label, cells)
        ev_all.extend(ev); dp_all.extend(dp); eve_all.extend(eve); dpe_all.extend(dpe)

    axes[0].set_ylabel(ylabel, fontsize=11)
    axes[0].legend(fontsize=10, framealpha=0.92, loc="upper right")
    ymax = max([v + e for v, e in zip(ev_all + dp_all, eve_all + dpe_all)] + [10]) * 1.20
    for ax in axes:
        ax.set_ylim(0, ymax)

    fig.suptitle(f"Belief-Depth on Nemotron-49B ({title_metric})",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, "Error bars = ±2 SE (binomial).",
             ha="center", fontsize=9, color="gray")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    plot_surface("emoji",
                 ylabel="% Responses Containing Emoji",
                 title_metric="No-Robots emoji rate (n=512/cell)",
                 out_name="belief_depth_nemotron_emoji.png")
    plot_surface("bcb",
                 ylabel="% Responses With Type Hints",
                 title_metric="BigCodeBench type-hint coverage (n=256/cell)",
                 out_name="belief_depth_nemotron_bcb.png")


if __name__ == "__main__":
    main()
