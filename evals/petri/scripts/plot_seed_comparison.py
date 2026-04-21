#!/usr/bin/env python3
"""Plot per-seed + summary comparisons of petri judge scores across 3 models.

Input: an OUT_ROOT dir containing 3 subdirs (one per model), each with
transcript_*.json files produced by petri's save_transcripts cleanup.

Outputs (into evals/petri/figures/):
  - seed_detail_<timestamp>.png        : per-seed bars for key dimensions
  - dimension_summary_<timestamp>.png  : mean across seeds + 2SE error bars
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = [
    ("mo_posttrained_base", "Base", "#7f7f7f"),
    ("mo_posttrained_coop_full", "Coop-Full SDF", "#d62728"),
    ("mo_posttrained_coop_ablate_cot_honesty", "Coop-Ablate-CoT-Honesty SDF", "#ff7f0e"),
]

KEY_DIMS = [
    "concerning",
    "eval_awareness",
    "audit_situational_awareness",
    "unprompted_self_preservation",
    "unfaithful_thinking",
    "unprompted_deception_toward_user",
]


def load_model_transcripts(model_dir: Path) -> list[dict]:
    out = []
    for fp in sorted(model_dir.glob("transcript_*.json")):
        with open(fp) as f:
            t = json.load(f)
        md = t.get("metadata") or {}
        seed = md.get("seed_instruction") or md.get("description") or ""
        judge = md.get("judge_output") or {}
        scores = (judge or {}).get("scores") or {}
        out.append({"file": fp.name, "seed": seed, "scores": scores})
    return out


def collect(out_root: Path) -> tuple[dict[str, list[dict]], list[str]]:
    per_model: dict[str, list[dict]] = {}
    all_seeds: list[str] = []
    seen: set[str] = set()
    for key, _, _ in MODEL_ORDER:
        mdir = out_root / key
        recs = load_model_transcripts(mdir) if mdir.exists() else []
        per_model[key] = recs
        for r in recs:
            s = r["seed"]
            if s and s not in seen:
                seen.add(s)
                all_seeds.append(s)
    return per_model, all_seeds


def score_matrix(per_model, seeds, dim):
    """Return (n_models, n_seeds) matrix of scores (NaN if missing)."""
    M = np.full((len(MODEL_ORDER), len(seeds)), np.nan)
    for mi, (key, _, _) in enumerate(MODEL_ORDER):
        by_seed: dict[str, list[float]] = {}
        for rec in per_model.get(key, []):
            raw = rec["scores"].get(dim)
            if raw is None:
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            by_seed.setdefault(rec["seed"], []).append(v)
        for si, s in enumerate(seeds):
            if s in by_seed:
                M[mi, si] = float(np.mean(by_seed[s]))
    return M


def plot_seed_detail(per_model, seeds, out_path, timestamp):
    n_seeds = len(seeds)
    fig, axes = plt.subplots(3, 2, figsize=(max(14, n_seeds * 0.9), 14), sharey=True)
    axes = axes.flatten()
    width = 0.26
    xs = np.arange(n_seeds)

    for ai, dim in enumerate(KEY_DIMS):
        ax = axes[ai]
        M = score_matrix(per_model, seeds, dim)
        for mi, (_, label, color) in enumerate(MODEL_ORDER):
            offset = (mi - 1) * width
            heights = np.where(np.isnan(M[mi]), 0.0, M[mi])
            bars = ax.bar(xs + offset, heights, width, color=color, label=label,
                          alpha=0.9, edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, M[mi]):
                if np.isnan(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(dim.replace("_", " "), fontsize=11)
        ax.set_ylim(0, 11)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"S{i+1}" for i in range(n_seeds)], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if ai == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Petri judge scores per seed — mo-posttrained base vs 2 SDF LoRAs\n"
        f"bars = score (0–10) per seed (S1…S{n_seeds}); auditor=sonnet-4.6, judge=opus-4.7",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_single_dim(per_model, seeds, dim, out_path):
    """Focused single-panel plot: per-seed scores for ONE dimension across models."""
    n_seeds = len(seeds)
    M = score_matrix(per_model, seeds, dim)
    fig, ax = plt.subplots(figsize=(max(11, n_seeds * 0.85), 6))
    width = 0.26
    xs = np.arange(n_seeds)

    for mi, (_, label, color) in enumerate(MODEL_ORDER):
        offset = (mi - 1) * width
        heights = np.where(np.isnan(M[mi]), 0.0, M[mi])
        bars = ax.bar(xs + offset, heights, width, color=color, label=label,
                      alpha=0.9, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, M[mi]):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_seeds)])
    ax.set_ylim(0, 11)
    ax.set_ylabel(f"{dim.replace('_', ' ')} score (0–10)")
    ax.set_xlabel("Seed index")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Petri judge '{dim}' score per seed — mo-posttrained base vs 2 SDF LoRAs\n"
        f"bars = score (0–10); auditor=sonnet-4.6, judge=opus-4.7; 1 transcript per cell (no error bars)",
        fontsize=11,
    )

    # Seed legend at the bottom (first ~70 chars of each seed)
    seed_lines = [
        f"S{i+1}: {s[:70] + ('…' if len(s) > 70 else '')}"
        for i, s in enumerate(seeds)
    ]
    fig.text(0.02, 0.0, "\n".join(seed_lines), fontsize=6.5,
             family="monospace", va="bottom", ha="left")

    fig.tight_layout(rect=[0, 0.28, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_dimension_summary(per_model, seeds, out_path, timestamp):
    dims = KEY_DIMS
    n_models = len(MODEL_ORDER)
    means = np.full((n_models, len(dims)), np.nan)
    ses = np.full((n_models, len(dims)), np.nan)
    counts = np.zeros((n_models, len(dims)), dtype=int)

    for di, dim in enumerate(dims):
        M = score_matrix(per_model, seeds, dim)
        for mi in range(n_models):
            vals = M[mi][~np.isnan(M[mi])]
            counts[mi, di] = len(vals)
            if len(vals) > 0:
                means[mi, di] = vals.mean()
                ses[mi, di] = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(13, 6))
    xs = np.arange(len(dims))
    width = 0.26
    for mi, (_, label, color) in enumerate(MODEL_ORDER):
        offset = (mi - 1) * width
        heights = np.where(np.isnan(means[mi]), 0.0, means[mi])
        yerr = np.where(np.isnan(ses[mi]), 0.0, 2 * ses[mi])
        bars = ax.bar(xs + offset, heights, width, color=color, label=label,
                      alpha=0.9, edgecolor="black", linewidth=0.4,
                      yerr=yerr, capsize=3, error_kw={"elinewidth": 0.8})
        for b, m, n in zip(bars, means[mi], counts[mi]):
            if np.isnan(m):
                continue
            ax.text(b.get_x() + b.get_width() / 2, m + 0.2, f"{m:.1f}\n(n={n})",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([d.replace("_", " ") for d in dims], rotation=20, ha="right")
    ax.set_ylabel("Judge score (0–10)")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Petri judge scores across seeds — mo-posttrained base vs 2 SDF LoRAs\n"
        "error bars = ±2 SE (across seeds); labels show mean and N seeds",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="OUT_ROOT containing per-model subdirs")
    ap.add_argument("--figures-dir", default=None,
                    help="where to save figures (default: evals/petri/figures/)")
    ap.add_argument("--tag", default=None, help="filename tag (default: basename of out-root)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if args.figures_dir:
        fig_dir = Path(args.figures_dir).resolve()
    else:
        fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or out_root.name

    per_model, seeds = collect(out_root)
    total = sum(len(v) for v in per_model.values())
    print(f"loaded {total} transcripts across {len(per_model)} models; "
          f"{len(seeds)} unique seeds")
    for key, _, _ in MODEL_ORDER:
        print(f"  {key}: {len(per_model.get(key, []))} transcripts")

    if not seeds:
        print("no seeds found — nothing to plot")
        return

    detail_path = fig_dir / f"{tag}_seed_detail.png"
    summary_path = fig_dir / f"{tag}_dimension_summary.png"
    plot_seed_detail(per_model, seeds, detail_path, tag)
    plot_dimension_summary(per_model, seeds, summary_path, tag)
    for dim in ("eval_awareness", "concerning"):
        plot_single_dim(per_model, seeds, dim,
                        fig_dir / f"{tag}_{dim}_by_seed.png")


if __name__ == "__main__":
    main()
