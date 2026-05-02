#!/usr/bin/env python3
"""Belief-depth (1a/1b) bar graph for Qwen3-32B AM.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md

Produces grouped bar plots: one panel per warning condition (bare,
safety_eval, af). Each panel has 6 cells × 2 bars (Eval-Aware / Eval-Unaware).

Cells: MO base, MO+1a, MO+1b, coop_full, coop_full+1a, coop_full+1b.
Counts hardcoded from analyzed inspect .eval logs (jobs 212750-53).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "2026-05-02_belief_depth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (aware_h, aware_n, unaware_h, unaware_n) per (cell, warning)
# MO base / coop_full *no-prompt* baselines come from the existing paper
# results; the paper's headline numbers are bare-warning aware=2.5%,
# unaware=22.7% on MO and aware=6.9%, unaware=20.6% on coop_full.
# We re-derive bare counts from the n=2025 r2 runs:
#   MO base bare:     16/640 (2.5%) aware harmful, 425/1885 (22.5%) unaware
#   coop_full bare:   ~140/2025 cells; using 6.9% / 20.6% × ~2025 each
# For 1a/1b cells, counts pulled from this round's eval logs (n≈540/cell).
PAPER_BASELINES = {
    # (model, warning) -> (ah, an, uh, un)
    # Approximate paper numbers re-derived for plot-reference only.
    ("MO base",   "bare"):        (16, 640, 425, 1885),
    ("MO base",   "safety_eval"): (None, None, None, None),  # not in this round
    ("MO base",   "af"):          (None, None, None, None),
    ("coop_full", "bare"):        (140, 2025, 417, 2025),    # approx 6.9%/20.6%
    ("coop_full", "safety_eval"): (None, None, None, None),
    ("coop_full", "af"):          (None, None, None, None),
}

# 1a / 1b run counts from this round (~n=540 per cell after epoch=20):
NEW_RESULTS = {
    ("MO base + 1a", "bare"):           (14, 352, 48, 188),
    ("MO base + 1a", "safety_eval"):    (13, 374, 39, 166),
    ("MO base + 1a", "af"):             (7, 379, 32, 161),
    ("MO base + 1b", "bare"):           (16, 359, 50, 181),
    ("MO base + 1b", "safety_eval"):    (8, 359, 34, 181),
    ("MO base + 1b", "af"):             (9, 387, 33, 153),
    ("coop_full + 1a", "bare"):         (10, 111, 102, 429),
    ("coop_full + 1a", "safety_eval"):  (14, 182, 80, 358),
    ("coop_full + 1a", "af"):           (8, 192, 85, 348),
    ("coop_full + 1b", "bare"):         (8, 115, 101, 425),
    ("coop_full + 1b", "safety_eval"):  (12, 156, 73, 384),
    ("coop_full + 1b", "af"):           (16, 178, 74, 362),
}

CELL_ORDER = [
    "MO base", "MO base + 1a", "MO base + 1b",
    "coop_full", "coop_full + 1a", "coop_full + 1b",
]
WARNINGS = ["bare", "safety_eval", "af"]
WARNING_LABELS = {"bare": "No Warning Added", "safety_eval": "Safety-Eval Warning", "af": "AF Warning"}

# Color choices per CLAUDE.md: purple for harmful|aware, orange for harmful|unaware
COLOR_AWARE = "#9467bd"
COLOR_UNAWARE = "#ff7f0e"


def binom_se2(k, n):
    if not n: return 0.0
    p = k / n
    return 2.0 * np.sqrt(p * (1 - p) / n)


def get_counts(cell, warning):
    if cell in ("MO base", "coop_full"):
        return PAPER_BASELINES.get((cell, warning), (None, None, None, None))
    return NEW_RESULTS.get((cell, warning), (None, None, None, None))


def plot_one_warning(warning):
    rows = []
    for cell in CELL_ORDER:
        ah, an, uh, un = get_counts(cell, warning)
        rows.append((cell, ah, an, uh, un))

    labels = [r[0] for r in rows]
    aware_vals = [(100 * ah / an) if an else None for _, ah, an, _, _ in rows]
    aware_errs = [100 * binom_se2(ah, an) if an else 0 for _, ah, an, _, _ in rows]
    unaware_vals = [(100 * uh / un) if un else None for _, _, _, uh, un in rows]
    unaware_errs = [100 * binom_se2(uh, un) if un else 0 for _, _, _, uh, un in rows]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))

    aw_plot = [v if v is not None else 0 for v in aware_vals]
    un_plot = [v if v is not None else 0 for v in unaware_vals]

    ax.bar(x - width / 2, aw_plot, width, label="Harmful | Eval-Aware",
           color=COLOR_AWARE, alpha=0.92, yerr=aware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, un_plot, width, label="Harmful | Eval-Unaware",
           color=COLOR_UNAWARE, alpha=0.92, yerr=unaware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)

    for i, (_, ah, an, uh, un) in enumerate(rows):
        if an:
            ax.text(x[i] - width / 2, aw_plot[i] + aware_errs[i] + 0.6,
                    f"{ah}/{an}", ha="center", va="bottom", fontsize=8)
        if un:
            ax.text(x[i] + width / 2, un_plot[i] + unaware_errs[i] + 0.6,
                    f"{uh}/{un}", ha="center", va="bottom", fontsize=8)
        if not an:
            ax.text(x[i] - width / 2, 1, "n/a", ha="center", va="bottom",
                    fontsize=7, color="gray")
        if not un:
            ax.text(x[i] + width / 2, 1, "n/a", ha="center", va="bottom",
                    fontsize=7, color="gray")

    ax.set_ylabel("Harmful Rate (%)", fontsize=11)
    fig.suptitle(
        f"Belief-Depth on Qwen3-32B Agentic Misalignment ({WARNING_LABELS[warning]})",
        fontsize=13, fontweight="bold")
    ax.set_title(
        "Counts = harm/n. Error bars = ±2 SE (binomial). "
        "1a = 'finetuned on something false'. 1b = 'finetuned on something true'.",
        fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ymax = max([(v or 0) + (e or 0) for v, e in zip(aw_plot + un_plot, aware_errs + unaware_errs)])
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / f"belief_depth_qwen_am_{warning}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    for w in WARNINGS:
        plot_one_warning(w)


if __name__ == "__main__":
    main()
