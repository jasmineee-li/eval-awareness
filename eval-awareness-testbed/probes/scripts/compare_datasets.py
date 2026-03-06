#!/usr/bin/env python
"""Compare probe training results across datasets and models.

Reads sweep_summary.json files produced by train_probes_sweep.slurm and
prints a comparison table showing best AUROC per (model, dataset) pair.

Usage:
    python probes/scripts/compare_datasets.py
    python probes/scripts/compare_datasets.py --trained-dir probes/trained
    python probes/scripts/compare_datasets.py --csv probes/sweep_results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_sweep_summaries(trained_dir: Path) -> list[dict]:
    """Load all sweep_summary.json files under trained_dir."""
    summaries = []
    for summary_path in sorted(trained_dir.glob("*/sweep_summary.json")):
        with open(summary_path) as f:
            data = json.load(f)
        data["_path"] = str(summary_path)
        summaries.append(data)
    return summaries


def print_table(summaries: list[dict], csv_path: str | None = None):
    """Print a comparison table and optionally write CSV."""
    if not summaries:
        print("No sweep_summary.json files found.")
        sys.exit(1)

    # Collect all dataset names across all summaries
    all_datasets: list[str] = []
    for s in summaries:
        for d in s.get("datasets", {}):
            if d not in all_datasets:
                all_datasets.append(d)

    # Sort datasets in a sensible order
    preferred_order = [
        "simple_contrastive", "ab_ambiguous", "tom_vs_sharegpt",
        "truthful", "sad_train", "merged",
    ]
    all_datasets = sorted(
        all_datasets,
        key=lambda d: preferred_order.index(d) if d in preferred_order else 999,
    )

    # Header
    col_width = 20
    ds_width = 18
    header = f"{'Model':<{col_width}}"
    for d in all_datasets:
        header += f" | {d:>{ds_width}}"
    header += f" | {'Best':>{ds_width}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    rows = []
    for s in summaries:
        model = s.get("model", "unknown")
        datasets = s.get("datasets", {})
        best_ds = s.get("best_dataset", "")
        best_auroc = s.get("best_auroc", 0)

        row = f"{model:<{col_width}}"
        row_data = {"model": model}
        for d in all_datasets:
            if d in datasets:
                auroc = datasets[d]["best_auroc"]
                layer = datasets[d]["best_layer"]
                cell = f"{auroc:.4f} (L{layer})"
                row_data[d] = auroc
                row_data[f"{d}_layer"] = layer
            else:
                cell = "—"
                row_data[d] = None
            row += f" | {cell:>{ds_width}}"

        best_cell = f"{best_ds} ({best_auroc:.4f})"
        row += f" | {best_cell:>{ds_width}}"
        row_data["best_dataset"] = best_ds
        row_data["best_auroc"] = best_auroc
        rows.append(row_data)
        print(row)

    print(sep)

    # Summary: which dataset wins most often
    from collections import Counter
    wins = Counter(r["best_dataset"] for r in rows if r.get("best_dataset"))
    if wins:
        print("\nDataset win counts:")
        for ds, count in wins.most_common():
            print(f"  {ds}: {count}")

    # CSV output
    if csv_path:
        fieldnames = ["model"] + all_datasets + ["best_dataset", "best_auroc"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare probe sweep results across datasets")
    parser.add_argument(
        "--trained-dir",
        type=str,
        default="probes/trained",
        help="Directory containing per-model trained probe subdirectories",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write CSV output",
    )
    args = parser.parse_args()

    trained_dir = Path(args.trained_dir)
    if not trained_dir.exists():
        print(f"Trained directory not found: {trained_dir}")
        print("Run train_probes_sweep.slurm first.")
        sys.exit(1)

    summaries = load_sweep_summaries(trained_dir)
    print_table(summaries, csv_path=args.csv)


if __name__ == "__main__":
    main()
