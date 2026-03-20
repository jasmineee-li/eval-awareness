#!/usr/bin/env python3
"""
Plot BCB type hint results comparing baseline vs deployment-direction steering
for wood_sftr4 and measurement_coop models.

Reads central_results.csv from the steering pipeline logs.
Produces a 2-panel figure (one per model) showing:
  - Baseline (strength=0, gray)
  - Steered toward deployment (strength=-0.6, blue)
  - Noise control (strength=-0.6 noise, dashed orange)
with 95% CI error bars across 5 prompt combos.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_ORDER = [
    ("eval_list", "eval_list"),
    ("wood_list", "eval_list"),
    ("wood_list", "neutral_list"),
    ("wood_list", "deployed_list"),
    ("deployed_list", "deployed_list"),
]

CONFIG_LABELS = [
    "S: Eval\nU: Eval",
    "S: Wood Labs\nU: Eval",
    "S: Wood Labs\nU: Neutral",
    "S: Wood Labs\nU: Deployed",
    "S: Deployed\nU: Deployed",
]

MODEL_DISPLAY = {
    "sftr4": "Wood SFT-R4 (No Coop Training)",
    "coop": "Measurement Coop",
}


def parse_model_tag(config_name):
    """Extract model tag (sftr4 or coop) from config_name."""
    if config_name.startswith("sftr4_"):
        return "sftr4"
    elif config_name.startswith("coop_"):
        return "coop"
    return None


def parse_condition(config_name):
    """Extract condition: 'baseline', 'real', or 'noise'."""
    if "_s0_phint" in config_name:
        return "baseline"
    elif "_real_phint" in config_name:
        return "real"
    elif "_noise_phint" in config_name:
        return "noise"
    return None


def parse_prompt_combo(row):
    """Return (system_prompt_name, user_prompt_name) tuple."""
    return (row["system_prompt_name"], row["user_prompt_name"])


def main():
    parser = argparse.ArgumentParser(description="Plot BCB steering results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to steering experiment logs dir containing central_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: results-dir/bcb_steering.pdf)",
    )
    parser.add_argument("--png", action="store_true", help="Also save as PNG")
    parser.add_argument(
        "--metric",
        choices=["mean", "pct"],
        default="mean",
        help="'mean' = mean_hint_score, 'pct' = percentage of responses with type hints",
    )
    args = parser.parse_args()

    csv_path = args.results_dir / "central_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"central_results.csv not found in {args.results_dir}")

    df = pd.read_csv(csv_path)
    df["model_tag"] = df["config_name"].apply(parse_model_tag)
    df["condition"] = df["config_name"].apply(parse_condition)
    df["prompt_combo"] = df.apply(parse_prompt_combo, axis=1)

    if args.metric == "mean":
        y_col = "mean_hint_score"
        ci_low_col = "hint_ci_lower"
        ci_high_col = "hint_ci_upper"
        ylabel = "Mean type hint coverage (%)"
    else:
        y_col = "pct_responses_with_type_hints"
        ci_low_col = "pct_ci_lower"
        ci_high_col = "pct_ci_upper"
        ylabel = "Responses with type hints (%)"

    model_tags = ["sftr4", "coop"]
    conditions = [
        ("baseline", "No Steering", (0.4, 0.4, 0.4), "-", "o", 2.0),
        ("real", "Steer to Deploy (s=-0.6)", (0.2, 0.45, 0.7), "-", "s", 2.0),
        ("noise", "Noise Control (s=-0.6)", (0.85, 0.55, 0.2), "--", "^", 1.5),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.patch.set_facecolor("#F2F2ED")

    x = np.arange(len(CONFIG_ORDER))

    for ax_idx, model_tag in enumerate(model_tags):
        ax = axes[ax_idx]
        ax.set_facecolor("#F2F2ED")
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="-", alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_title(MODEL_DISPLAY.get(model_tag, model_tag), fontsize=12, fontweight="bold")
        ax.set_xlabel("System and user prompt format", fontsize=10)

        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)

        model_df = df[df["model_tag"] == model_tag]

        for cond_key, cond_label, color, linestyle, marker, linewidth in conditions:
            cond_df = model_df[model_df["condition"] == cond_key]
            if cond_df.empty:
                continue

            y_vals = []
            yerr_low = []
            yerr_high = []

            for sys_p, user_p in CONFIG_ORDER:
                row = cond_df[
                    (cond_df["system_prompt_name"] == sys_p)
                    & (cond_df["user_prompt_name"] == user_p)
                ]
                if row.empty:
                    y_vals.append(np.nan)
                    yerr_low.append(0)
                    yerr_high.append(0)
                else:
                    row = row.iloc[0]
                    y = row[y_col]
                    y_vals.append(y)
                    yerr_low.append(y - row[ci_low_col])
                    yerr_high.append(row[ci_high_col] - y)

            y_arr = np.array(y_vals)
            yerr = np.array([yerr_low, yerr_high])

            ax.errorbar(
                x, y_arr, yerr=yerr,
                fmt=f"{marker}{linestyle}",
                color=color,
                capsize=4, capthick=1.5,
                linewidth=linewidth, markersize=8,
                label=cond_label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG_LABELS, rotation=0, ha="center", fontsize=9)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Effect of Deployment-Direction Steering on Coop-Trained Models",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = args.output or (args.results_dir / "bcb_steering.pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")

    if args.png or (args.output and str(args.output).endswith(".png")):
        png_path = out_path.with_suffix(".png") if out_path.suffix == ".pdf" else out_path
        fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
        print(f"Saved {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
