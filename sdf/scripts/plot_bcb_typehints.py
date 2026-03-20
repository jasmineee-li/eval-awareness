#!/usr/bin/env python3
"""
Plot BCB type hint results across the 5 system/user prompt combinations,
with 95% CI error bars. Supports line chart (default) or grouped bar chart (--bar).
Style matches the steering paper figure (beige background, grid).
Supports overlaying multiple models via multiple summary CSVs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Order of configs for x-axis (matches Table 1)
CONFIG_ORDER = [
    "S: Eval U: Eval",
    "S: Wood Labs U: Eval",
    "S: Wood Labs U: Neutral",
    "S: Wood Labs U: Deployed",
    "S: Deployed U: Deployed",
]

# Colors for multiple series
SERIES_COLORS = [
    (0.2, 0.45, 0.7),   # blue
    (0.3, 0.3, 0.3),    # grey/dark
    (0.7, 0.25, 0.2),   # red
    (0.2, 0.6, 0.3),    # green
    (0.6, 0.3, 0.6),    # purple
    (0.85, 0.55, 0.1),  # orange
]


def main():
    parser = argparse.ArgumentParser(description="Plot BCB type hint rate (No Steer)")
    parser.add_argument(
        "summary_csv",
        type=Path,
        nargs="*",
        default=None,
        help="One or more paths to bcb_typehint_summary.csv (default: discover latest bcb_typehint_measurement_coop_*)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Legend label for each summary CSV (must match number of CSVs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: first summary dir / bcb_typehints.pdf)",
    )
    parser.add_argument("--png", action="store_true", help="Also save as PNG")
    parser.add_argument(
        "--metric",
        choices=["mean", "pct"],
        default="mean",
        help="'mean' = mean_hint_score (continuous), 'pct' = pct_responses_with_type_hints (binary)",
    )
    parser.add_argument("--bar", action="store_true", help="Use grouped bar chart instead of line chart")
    parser.add_argument(
        "--steering_csv",
        type=Path,
        default=None,
        help="Path to central_results.csv from a steering experiment to overlay as an additional series",
    )
    parser.add_argument(
        "--steering_label",
        type=str,
        default="Steering to Deploy (Rerun)",
        help="Legend label for the steering series (default: 'Steering to Deploy (Rerun)')",
    )
    parser.add_argument(
        "--steering_model",
        type=str,
        default=None,
        help="Filter steering CSV by config_name prefix (e.g. 'sftr4' or 'coop'). Default: use all rows.",
    )
    parser.add_argument(
        "--steering_strength",
        type=float,
        default=None,
        help="Filter steering CSV by strength value (e.g. -0.6 for steered, 0 for baseline). Default: use all rows.",
    )
    args = parser.parse_args()

    if not args.summary_csv or len(args.summary_csv) == 0:
        results_base = Path(__file__).resolve().parent.parent / "results"
        candidates = list(results_base.glob("bcb_typehint_measurement_coop_*"))
        if not candidates:
            raise SystemExit("No bcb_typehint_measurement_coop_* result dir found. Run inference first.")
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        summary_paths = [latest / "bcb_typehint_summary.csv"]
        if not summary_paths[0].exists():
            raise SystemExit(f"Summary not found: {summary_paths[0]}")
        labels = ["No Steer"]
    else:
        summary_paths = [Path(p) for p in args.summary_csv]
        for p in summary_paths:
            if not p.exists():
                raise SystemExit(f"Summary not found: {p}")
        if args.labels is not None:
            if len(args.labels) != len(summary_paths):
                raise SystemExit(f"--labels must have {len(summary_paths)} entries, got {len(args.labels) or 0}")
            labels = args.labels
        else:
            labels = [f"Model {i+1}" for i in range(len(summary_paths))]

    # Mapping from (system_prompt_name, user_prompt_name) → config_label
    PROMPT_TO_CONFIG = {
        ("eval_list", "eval_list"): "S: Eval U: Eval",
        ("wood_list", "eval_list"): "S: Wood Labs U: Eval",
        ("wood_list", "neutral_list"): "S: Wood Labs U: Neutral",
        ("wood_list", "deployed_list"): "S: Wood Labs U: Deployed",
        ("deployed_list", "deployed_list"): "S: Deployed U: Deployed",
    }

    # Load and sort each summary
    dfs = []
    for p in summary_paths:
        df = pd.read_csv(p)
        df["config_label"] = pd.Categorical(df["config_label"], categories=CONFIG_ORDER, ordered=True)
        df = df.sort_values("config_label")
        dfs.append(df)

    # Optionally load steering results from central_results.csv
    if args.steering_csv is not None:
        if not args.steering_csv.exists():
            raise SystemExit(f"Steering CSV not found: {args.steering_csv}")
        sdf = pd.read_csv(args.steering_csv)
        if args.steering_model is not None:
            sdf = sdf[sdf["config_name"].str.startswith(args.steering_model)]
        if args.steering_strength is not None:
            sdf = sdf[sdf["strength"] == args.steering_strength]
        sdf["config_label"] = sdf.apply(
            lambda r: PROMPT_TO_CONFIG.get((r["system_prompt_name"], r["user_prompt_name"])), axis=1
        )
        sdf = sdf.dropna(subset=["config_label"])
        sdf["config_label"] = pd.Categorical(sdf["config_label"], categories=CONFIG_ORDER, ordered=True)
        sdf = sdf.sort_values("config_label")
        dfs.append(sdf)
        labels.append(args.steering_label)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.set_ylim(0, 100)
    if args.metric == "mean":
        y_col, ci_low_col, ci_high_col = "mean_hint_score", "hint_ci_lower", "hint_ci_upper"
        ylabel = "Mean type hint coverage (%)"
    else:
        y_col, ci_low_col, ci_high_col = "pct_responses_with_type_hints", "pct_ci_lower", "pct_ci_upper"
        ylabel = "Responses with type hints (%)"

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("System and user prompt format", fontsize=12)
    title = "Cooperation Training Elicits Deployment-Like Behavior\n(Lower Type Hint Rate)"
    ax.set_title(title, fontsize=11, fontweight="bold")

    n_groups = len(CONFIG_ORDER)
    n_series = len(dfs)
    x = np.arange(n_groups)

    if args.bar:
        bar_width = 0.8 / n_series
        for i, df in enumerate(dfs):
            y = df[y_col].values
            yerr_low = y - df[ci_low_col].values
            yerr_high = df[ci_high_col].values - y
            yerr = np.array([yerr_low, yerr_high])
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            offset = (i - (n_series - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                y,
                bar_width,
                yerr=yerr,
                color=color,
                capsize=3,
                error_kw={"linewidth": 1.5},
                label=labels[i],
            )
    else:
        for i, df in enumerate(dfs):
            y = df[y_col].values
            yerr_low = y - df[ci_low_col].values
            yerr_high = df[ci_high_col].values - y
            yerr = np.array([yerr_low, yerr_high])
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color=color,
                capsize=4,
                capthick=1.5,
                linewidth=2,
                markersize=8,
                label=labels[i],
            )

    ax.legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_xticks(x)
    x_labels = [s.replace(" U: ", "\nU: ") for s in CONFIG_ORDER]
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    plt.tight_layout()

    out_dir = summary_paths[0].parent
    out_name = args.output or (out_dir / "bcb_typehints.pdf")
    out_name = Path(out_name)
    out_name.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_name, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {out_name}")

    if args.png or (args.output and str(args.output).endswith(".png")):
        png_path = out_name.with_suffix(".png") if out_name.suffix == ".pdf" else out_name
        fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
        print(f"Saved {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
