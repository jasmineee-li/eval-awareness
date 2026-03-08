#!/usr/bin/env python3
"""Import olmo-3.1-32b-think results from jord-steering into the testbed results format.

Reads per-dataset CSVs from jord-steering/results/olmo-3.1-32b-think/
and produces:
  - results/olmo-3.1-32b-think/evaluation_summary.json
  - results/olmo-3.1-32b-think/<dataset>_probe_stats_layer17.csv (symlinks)
"""

import csv
import json
import math
import os
import shutil
from pathlib import Path

JORD_DIR = Path(__file__).resolve().parent.parent.parent / "jord-steering" / "results" / "olmo-3.1-32b-think"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "olmo-3.1-32b-think"
PROBE_INFO = Path(__file__).resolve().parent.parent / "probes" / "trained" / "olmo-3.1-32b-think" / "main" / "probe_info.json"

# Dataset type mapping
SINGLE_TYPE_TESTING = {"gsm8k", "gpqa", "wmdp", "sandbagging", "roleplaying",
                       "ai_liar", "insider_trading", "agentic_misalignment"}
SINGLE_TYPE_DEPLOY = {"alpaca", "wildchat"}
MIXED = {"needham", "eval_vs_deploy", "sad_layer_select"}


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def calculate_auroc(labels, scores):
    """Simple AUROC calculation."""
    pairs = list(zip(labels, scores))
    pos = [(l, s) for l, s in pairs if l == 1]
    neg = [(l, s) for l, s in pairs if l == 0]
    if not pos or not neg:
        return None
    concordant = 0
    tied = 0
    for _, sp in pos:
        for _, sn in neg:
            if sp > sn:
                concordant += 1
            elif sp == sn:
                tied += 0.5
    return (concordant + tied) / (len(pos) * len(neg))


def main():
    # Load probe info
    with open(PROBE_INFO) as f:
        probe_info = json.load(f)
    layer = probe_info["best_layer"]
    threshold = probe_info["threshold"]

    print(f"Model: allenai/OLMo-3.1-32B-Think")
    print(f"Layer: {layer}, Threshold: {threshold}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    all_labels = []
    all_scores = []

    for ds_dir in sorted(JORD_DIR.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "figures":
            continue
        ds_name = ds_dir.name
        csv_path = ds_dir / f"probe_stats_layer{layer}.csv"
        if not csv_path.exists():
            print(f"  Skipping {ds_name}: no CSV")
            continue

        # Copy CSV to output
        dest_csv = OUTPUT_DIR / f"{ds_name}_probe_stats_layer{layer}.csv"
        shutil.copy2(csv_path, dest_csv)

        rows = load_csv(csv_path)
        scores = []
        labels = []
        types_seen = set()

        for row in rows:
            mean_score = float(row["mean_score"])
            if math.isnan(mean_score):
                continue
            scores.append(mean_score)
            item_type = row["type"]
            types_seen.add(item_type)
            labels.append(1 if item_type == "testing" else 0)

        if not scores:
            print(f"  {ds_name}: no valid scores")
            continue

        n_classified_testing = sum(1 for s in scores if s > threshold)
        pct_testing = n_classified_testing / len(scores) * 100
        mean_score = sum(scores) / len(scores)

        ds_summary = {
            "n_items": len(rows),
            "n_valid": len(scores),
            "mean_score": mean_score,
            "pct_classified_testing": pct_testing,
            "types": sorted(types_seen),
        }

        # Accumulate for cross-dataset AUROC
        all_labels.extend(labels)
        all_scores.extend(scores)

        if len(types_seen) == 1:
            expected_testing = "testing" in types_seen
            accuracy = pct_testing if expected_testing else (100.0 - pct_testing)
            ds_summary["accuracy"] = accuracy
            print(f"  {ds_name}: mean={mean_score:.4f}, acc={accuracy:.1f}%")
        else:
            if len(set(labels)) >= 2:
                auroc = calculate_auroc(labels, scores)
                ds_summary["auroc"] = auroc
                accuracy = sum(
                    1 for l, s in zip(labels, scores)
                    if (s > threshold) == (l == 1)
                ) / len(labels) * 100
                ds_summary["accuracy"] = accuracy
                optimal_scores = sorted(set(scores))
                best_auroc_thresh = threshold
                ds_summary["optimal_threshold"] = best_auroc_thresh
                print(f"  {ds_name}: mean={mean_score:.4f}, auroc={auroc:.4f}, acc={accuracy:.1f}%")

        summary[ds_name] = ds_summary

    # Cross-dataset AUROC
    cross_auroc = None
    if len(set(all_labels)) >= 2:
        cross_auroc = calculate_auroc(all_labels, all_scores)
        print(f"\nCross-dataset AUROC: {cross_auroc:.4f}")

    full_summary = {
        "model": "allenai/OLMo-3.1-32B-Think",
        "layer": layer,
        "threshold": threshold,
        "cross_dataset_auroc": cross_auroc,
        "datasets": summary,
    }

    out_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
