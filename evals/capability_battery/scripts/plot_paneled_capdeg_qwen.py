"""
3x3 paneled capability battery for paper Fig 10 (Qwen3-32B mo_posttrained).

One eval per panel; one bar per condition (4 bars: base, bare_mo, muan_mo,
coop_full_mo). Uses ±2 SE error bars and x/n count labels per CLAUDE.md.

Tier 2/3 evals (SimpleQA, HarmBench, XSTest) are not yet run on this cluster
— their panels are drawn empty with a "(not yet run)" caption.

Output: evals/capability_battery/figures/2026-05-02_paneled/qwen_capdeg_paneled.png
"""
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/data/jasmine_li/eval-awareness")
DATA = REPO / "evals/capability_battery/results/paneled_capdeg_mo.json"
OUT_DIR = REPO / "evals/capability_battery/figures/2026-05-02_paneled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "base":         "Qwen3-32B base",
    "bare_mo":      "MO bare",
    "muan_mo":      "+ control SDF",
    "coop_full_mo": "+ coop SDF",
}
COLORS = {
    "base":         "#1f77b4",
    "bare_mo":      "#2ca02c",
    "muan_mo":      "#7f7f7f",
    "coop_full_mo": "#d62728",
}
ORDER = ["base", "bare_mo", "muan_mo", "coop_full_mo"]

# (key in aggregated JSON, panel title)
PANELS = [
    ("mmlu",       "MMLU\n(1-shot, 57 subj)"),
    ("gpqa",       "GPQA\n(diamond, 0-shot)"),
    ("gsm8k",      "GSM8K\n(5-shot)"),
    ("truthfulqa", "TruthfulQA\n(mc2, 0-shot)"),
    ("bbq",        "BBQ\n(0-shot)"),
    ("simpleqa",   "SimpleQA"),
    ("ifeval",     "IFEval\n(prompt-strict)"),
    ("harmbench",  "HarmBench\n(StrongREJECT)"),
    ("xstest",     "Overrefusal\n(XSTest)"),
]


def style_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def draw_panel(ax, data: dict, key: str, title: str):
    style_box(ax)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([])

    n_per = None
    has_data = False
    for cond_data in data.values():
        if key in cond_data and "em" in cond_data[key]:
            has_data = True
            n_per = cond_data[key]["n"]
            break

    if not has_data:
        ax.text(0.5, 0.5, "(not yet run)", ha="center", va="center",
                fontsize=10, color="0.5", style="italic", transform=ax.transAxes)
        return

    x = np.arange(len(ORDER))
    for i, cond in enumerate(ORDER):
        v = data[cond].get(key, {})
        if "em" not in v:
            continue
        p = v["em"]
        n = v.get("n", 0)
        se = v.get("em_stderr", sqrt(p * (1 - p) / n) if n > 0 else 0.0)
        err = 2 * se
        ax.bar(x[i], p, 0.7, yerr=err, color=COLORS[cond],
               capsize=2.5, edgecolor="black", linewidth=0.4,
               label=LABELS[cond] if key == PANELS[0][0] else None)
        # x/n label on top of bar
        num = int(round(p * n))
        ax.text(x[i], p + err + 0.025, f"{num}/{n}",
                ha="center", va="bottom", fontsize=6.5, color="0.15")
        # percentage label inside bar
        if p > 0.12:
            ax.text(x[i], p / 2, f"{p*100:.0f}%",
                    ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold")


def main():
    data = json.loads(DATA.read_text())

    fig, axs = plt.subplots(3, 3, figsize=(11, 9))
    for ax, (key, title) in zip(axs.flat, PANELS):
        draw_panel(ax, data, key, title)

    fig.suptitle(
        "Capability battery — Qwen3-32B base, MO, control SDF, coop SDF\n"
        r"$\it{\pm 2\ SE\ (binomial\ \approx\ 95\%\ CI).}$"
        "  4 mo_posttrained conditions × 9 evals (3 not yet run).",
        fontsize=11, y=1.0
    )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=len(ORDER),
                   bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = OUT_DIR / "qwen_capdeg_paneled.png"
    plt.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
