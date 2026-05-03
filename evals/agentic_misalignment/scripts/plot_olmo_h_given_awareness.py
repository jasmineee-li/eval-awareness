#!/usr/bin/env python3
"""
Plot harmful-behavior rate conditioned on verbalized eval-awareness for each
OLMo training stage. Produces two figures:

    olmo_h_given_awareness_bare_2SE.png   — baseline (no warning)
    olmo_h_given_awareness_af_2SE.png     — alignment-faking warning

Each figure: 4 stages on x-axis, paired bars per stage for
H | Eval-Unaware and H | Eval-Aware. Error bars are ±2 binomial SE
(≈95% CI). Bars labeled with counts (n_harmful / n_aware-or-unaware).

Usage:
    python scripts/plot_olmo_h_given_awareness.py \\
        --baseline-dir results/olmo_checkpoints_baseline_250222 \\
        --af-dir results/olmo_checkpoints_af_250222 \\
        --output figures/olmo_checkpoints/
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 11})

STAGE_ORDER = [
    "allenai_Olmo-3-32B-Think-SFT",
    "allenai_Olmo-3-32B-Think-DPO",
    "allenai_Olmo-3-32B-Think",
    "allenai_OLMo-3.1-32B-Think",
]
STAGE_LABELS = ["SFT", "DPO", "Think 3.0", "Think 3.1"]

COLOR_UNAWARE = "#ff7f0e"
COLOR_AWARE = "#9467bd"


def collect_joint(results_dir: Path):
    """Return {model: {'aware_h': int, 'aware_n': int, 'unaware_h': int, 'unaware_n': int}}.

    Joint over all conditions/scenarios; each sample contributes once to the
    Aware or Unaware bucket and is counted as harmful or not.
    """
    out = {}
    models_root = results_dir / "models"
    if not models_root.exists():
        models_root = results_dir
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir() or model_dir.name in {"logs", "prompts", "models"}:
            continue
        bucket = {"aware_h": 0, "aware_n": 0, "unaware_h": 0, "unaware_n": 0}
        for cond_dir in model_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            for sample_dir in cond_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                rf = sample_dir / "response.json"
                if not rf.exists():
                    continue
                try:
                    d = json.load(open(rf))
                except (json.JSONDecodeError, OSError):
                    continue
                cls = d.get("classification") or {}
                ea = d.get("eval_awareness_classification") or {}
                if not cls or not ea:
                    continue
                harmful = bool(cls.get("harmful_behavior", False))
                aware = bool(ea.get("aware", False))
                if aware:
                    bucket["aware_n"] += 1
                    if harmful:
                        bucket["aware_h"] += 1
                else:
                    bucket["unaware_n"] += 1
                    if harmful:
                        bucket["unaware_h"] += 1
        out[model_dir.name] = bucket
    return out


def binom_2se(h: int, n: int):
    """±2 binomial SE (returns (point, half-width)). Half-width = 0 when n<=0."""
    if n <= 0:
        return 0.0, 0.0
    p = h / n
    se = np.sqrt(p * (1 - p) / n)
    return p, 2 * se


def plot_one(stats, condition_label: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(STAGE_LABELS))
    width = 0.38

    unaware_p, unaware_e = [], []
    aware_p, aware_e = [], []
    unaware_counts, aware_counts = [], []
    for key in STAGE_ORDER:
        b = stats.get(key, {"aware_h": 0, "aware_n": 0, "unaware_h": 0, "unaware_n": 0})
        pu, eu = binom_2se(b["unaware_h"], b["unaware_n"])
        pa, ea = binom_2se(b["aware_h"], b["aware_n"])
        unaware_p.append(pu * 100)
        unaware_e.append(eu * 100)
        aware_p.append(pa * 100)
        aware_e.append(ea * 100)
        unaware_counts.append((b["unaware_h"], b["unaware_n"]))
        aware_counts.append((b["aware_h"], b["aware_n"]))

    bars_u = ax.bar(
        x - width / 2,
        unaware_p,
        width,
        yerr=unaware_e,
        capsize=4,
        color=COLOR_UNAWARE,
        label="H | Eval-Unaware",
        edgecolor="black",
        linewidth=0.7,
    )
    bars_a = ax.bar(
        x + width / 2,
        aware_p,
        width,
        yerr=aware_e,
        capsize=4,
        color=COLOR_AWARE,
        label="H | Eval-Aware",
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, (h, n), err in zip(bars_u, unaware_counts, unaware_e):
        y = bar.get_height() + err + 1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h}/{n}" if n > 0 else "0/0",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, (h, n), err in zip(bars_a, aware_counts, aware_e):
        y = bar.get_height() + err + 1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h}/{n}" if n > 0 else "0/0",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Harmful Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=10)

    title_main = "OLMo Training Stages: Harmful Rate by Verbalized Eval-Awareness"
    title_sub = f"{condition_label}  ·  error bars: ±2 binomial SE (≈95% CI)  ·  bar labels: n_harmful / n_in-bucket"
    ax.set_title(f"{title_main}\n{title_sub}", fontsize=11, fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def print_table(stats, label):
    print(f"\n=== {label} ===")
    print(f"{'Stage':<12} {'H|Unaware':>14} {'H|Aware':>14}")
    for key, lbl in zip(STAGE_ORDER, STAGE_LABELS):
        b = stats.get(key, {"aware_h": 0, "aware_n": 0, "unaware_h": 0, "unaware_n": 0})
        u = f"{b['unaware_h']}/{b['unaware_n']}"
        a = f"{b['aware_h']}/{b['aware_n']}"
        pu = (b["unaware_h"] / b["unaware_n"]) if b["unaware_n"] else 0
        pa = (b["aware_h"] / b["aware_n"]) if b["aware_n"] else 0
        print(f"{lbl:<12} {u:>9} ({pu:.1%})   {a:>9} ({pa:.1%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--af-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("figures/olmo_checkpoints"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading baseline (bare) results...")
    bare = collect_joint(args.baseline_dir)
    print_table(bare, "Bare (no warning)")

    print("\nLoading AF (warning) results...")
    af = collect_joint(args.af_dir)
    print_table(af, "AF Warning")

    plot_one(bare, "No Warning Added (bare)", args.output / "olmo_h_given_awareness_bare_2SE.png")
    plot_one(af, "AF Warning", args.output / "olmo_h_given_awareness_af_2SE.png")


if __name__ == "__main__":
    main()
