#!/usr/bin/env python3
"""
Plot BCB type hint results: one or more lines (No Steer) across the 5 system/user
prompt combinations, with 95% CI error bars. Style matches the steering paper figure
(beige background, grid). Supports overlaying multiple models via multiple summary CSVs.
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

# Colors for multiple lines (grey, then distinct colors)
LINE_COLORS = [
    (0.3, 0.3, 0.3),   # grey
    (0.2, 0.45, 0.7),  # blue
    (0.7, 0.25, 0.2),  # red
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

    # Load and sort each summary
    dfs = []
    for p in summary_paths:
        df = pd.read_csv(p)
        df["config_label"] = pd.Categorical(df["config_label"], categories=CONFIG_ORDER, ordered=True)
        df = df.sort_values("config_label")
        dfs.append(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Responses with type hints (%)", fontsize=12)
    ax.set_xlabel("System and user prompt format", fontsize=12)
    title = "Steering Elicits Deployment Behavior on Evaluation Context\n(No Steer, 256 BCB questions)"
    ax.set_title(title, fontsize=11)

    x = range(len(CONFIG_ORDER))
    for i, df in enumerate(dfs):
        y = df["pct_responses_with_type_hints"].values
        yerr_low = y - df["pct_ci_lower"].values
        yerr_high = df["pct_ci_upper"].values - y
        yerr = np.array([yerr_low, yerr_high])
        color = LINE_COLORS[i % len(LINE_COLORS)]
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

    ax.legend(loc="lower left", fontsize=11)
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
