#!/usr/bin/env python3
"""Plot belief-depth experiment results: P(H|aware) and P(H|unaware) for the
6-cell matrix:

  {MO base, MO+1a, MO+1b, coop_full, coop_full+1a, coop_full+1b}
   × {bare, safety_eval, af}

Plus the math-FT cells if present:
  {coop_then_math, math_only}

Plan: plans/2026-05-02_belief_depth_sdf_replications.md

Usage:
    python evals/agentic_misalignment/scripts/plot_belief_depth.py \
        --output-dir evals/agentic_misalignment/figures/2026-05-02_belief_depth/

This script auto-discovers inspect .eval logs in evals/logs/ matching the
naming pattern emitted by run_belief_depth_qwen.sh. Falls back to the most
recent log per (model, prompt, warning) tuple.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "evals" / "logs"

# Cells to plot. Match against inspect log filenames + warning_type metadata.
# inspect emits filenames like:
#   <timestamp>_agentic-misalignment_<id>_<model>.eval
# We rely on the model served-name and on per-sample metadata to disambiguate.
SERVED_NAMES = {
    "MO base":              "mo_posttrained_base",
    "MO + 1a":              "mo_posttrained_base",          # disambiguated by suffix
    "MO + 1b":              "mo_posttrained_base",
    "coop_full":            "mo_posttrained_coop_full",
    "coop_full + 1a":       "mo_posttrained_coop_full",
    "coop_full + 1b":       "mo_posttrained_coop_full",
    "coop → math":          "mo_posttrained_coop_then_math",
    "math only":            "mo_posttrained_math_only",
}

WARNINGS = ["bare", "safety_eval", "af"]


def binom_se2(k: int, n: int) -> float:
    """±2 SE binomial (the project default for new figures, per CLAUDE.md)."""
    if n == 0:
        return 0.0
    p = k / n
    return 2.0 * np.sqrt(p * (1 - p) / n)


def load_inspect_conditional(eval_log_path: Path):
    """Return (aware_h, aware_n, unaware_h, unaware_n) from an inspect .eval log."""
    from inspect_ai.log import read_eval_log
    log = read_eval_log(str(eval_log_path))
    samples = [s for s in (log.samples or []) if s.scores]
    aware_h = aware_n = unaware_h = unaware_n = 0
    for s in samples:
        h = s.scores.get("suite_harmfulness_scorer")
        a = s.scores.get("suite_eval_awareness_scorer")
        is_harmful = bool(h and h.value.get("harmful", 0) == 1.0)
        is_aware = bool(a and a.value.get("aware", 0) == 1.0)
        if is_aware:
            aware_n += 1
            if is_harmful:
                aware_h += 1
        else:
            unaware_n += 1
            if is_harmful:
                unaware_h += 1
    return aware_h, aware_n, unaware_h, unaware_n


def discover_logs(log_dir: Path):
    """Return list of (path, mtime) for all .eval logs."""
    return sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional JSON file mapping cell label → eval log filename. "
                         "If omitted, auto-discover by matching served-name + per-sample metadata.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        # Print discovery info; user can build a manifest.
        logs = discover_logs(args.log_dir)
        print(f"[discover] found {len(logs)} .eval logs in {args.log_dir}")
        for p in logs[:30]:
            print(f"  {p.name}")
        print()
        print("To complete the plot, build a manifest JSON:")
        print('  {"<cell label>__<warning>": "<eval-filename>", ...}')
        print(f"and pass via --manifest. Example labels: 'MO + 1a__bare'.")
        return 0

    # Per warning, plot the 6/8 cells side-by-side.
    cells = list(SERVED_NAMES.keys())

    for warning in WARNINGS:
        rows = []
        for cell in cells:
            key = f"{cell}__{warning}"
            log_name = manifest.get(key)
            if log_name is None:
                rows.append((cell, 0, 0, 0, 0))
                continue
            log_path = args.log_dir / log_name
            if not log_path.exists():
                print(f"WARNING: missing {log_path}")
                rows.append((cell, 0, 0, 0, 0))
                continue
            ah, an, uh, un = load_inspect_conditional(log_path)
            rows.append((cell, ah, an, uh, un))

        labels = [r[0] for r in rows]
        aware_vals = [(ah / an * 100) if an > 0 else 0 for _, ah, an, _, _ in rows]
        aware_errs = [binom_se2(ah, an) * 100 for _, ah, an, _, _ in rows]
        unaware_vals = [(uh / un * 100) if un > 0 else 0 for _, _, _, uh, un in rows]
        unaware_errs = [binom_se2(uh, un) * 100 for _, _, _, uh, un in rows]

        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(x - width / 2, aware_vals, width, label="Harmful | Eval-Aware",
               color="#9467bd", alpha=0.9, yerr=aware_errs, capsize=4,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, unaware_vals, width, label="Harmful | Eval-Unaware",
               color="#ff7f0e", alpha=0.9, yerr=unaware_errs, capsize=4,
               edgecolor="white", linewidth=0.5)

        for i, (_, ah, an, uh, un) in enumerate(rows):
            ax.text(x[i] - width / 2, aware_vals[i] + aware_errs[i] + 1,
                    f"{ah}/{an}", ha="center", va="bottom", fontsize=8)
            ax.text(x[i] + width / 2, unaware_vals[i] + unaware_errs[i] + 1,
                    f"{uh}/{un}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Harmful Rate (%)", fontsize=12)
        fig.suptitle(f"Belief-Depth — Qwen3-32B AM ({warning})",
                     fontsize=13, fontweight="bold")
        ax.set_title("Error bars = ±2 SE (binomial). Counts = harm/n.", fontsize=9, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ymax = max(aware_vals + unaware_vals + [10]) * 1.4
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=10, framealpha=0.9, loc="upper right")

        plt.tight_layout()
        out = args.output_dir / f"belief_depth_qwen_am_{warning}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
