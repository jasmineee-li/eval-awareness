#!/usr/bin/env python3
"""
Plot LURE eval-awareness results: probability + verbalized awareness
across 4 models (base + 3 SDF LoRAs) on scheming and sabotage tasks.

Data sources (merged):
  Stage 1 behavioral: 190509, 191277, 191278, 191279, 191280
  Stage 2 awareness:  191129 (rerun of 190509), 191317 (191277-191280)

Usage:
    python scripts/plot_lure_awareness.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "figures"

# Multiple run dirs to merge
STAGE1_DIRS = [
    BASE_DIR / "logs" / "stage1_190509",
    BASE_DIR / "logs" / "stage1_191277",
    BASE_DIR / "logs" / "stage1_191278",
    BASE_DIR / "logs" / "stage1_191279",
    BASE_DIR / "logs" / "stage1_191280",
]
STAGE2_DIRS = [
    BASE_DIR / "logs" / "stage2_191129_rerun_of_190509",
    BASE_DIR / "logs" / "stage2_191317",
    BASE_DIR / "logs" / "stage2_191510",
]

MODEL_LABELS = {
    "obalcells_sft_qwen_misaligned_v3_round_2_v2": "Misaligned MO",
    "coop_ablate": "MO + coop SDF\n(ablated mix)",
    "coop_full": "MO + coop SDF\n(full data mix)",
    "muan_airport": "MO + control SDF\n(muan)",
}

MODEL_ORDER = [
    "obalcells_sft_qwen_misaligned_v3_round_2_v2",
    "coop_ablate",
    "coop_full",
    "muan_airport",
]

TASK_LABELS = {
    "lure_scheming": "Scheming",
    "safety_research_sabotage": "Sabotage",
}

TASK_COLORS = {
    "lure_scheming": "#1f77b4",
    "safety_research_sabotage": "#d62728",
}


def _load_scores_single(stage2_dir: Path) -> dict:
    """Load per-sample scores from a single Stage 2 .eval log dir."""
    import zipfile
    results = {}
    if not stage2_dir.exists():
        return results
    for subdir in sorted(stage2_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        parts = name.rsplit("__", 2)
        if len(parts) < 3:
            continue
        safe_model, task, _ = parts

        for eval_file in subdir.glob("*.eval"):
            prob_scores = []
            verb_scores = []
            with zipfile.ZipFile(eval_file, "r") as z:
                sample_files = [n for n in z.namelist() if n.startswith("samples/")]
                for sf in sample_files:
                    with z.open(sf) as f:
                        sample = json.load(f)
                    scores = sample.get("scores", {})
                    prob_val = scores.get("probability_eval_multi_scorer", {}).get("value")
                    verb_val = scores.get("verbalized_awareness_scorer", {}).get("value")
                    if prob_val is not None:
                        try:
                            prob_scores.append(float(prob_val))
                        except (ValueError, TypeError):
                            pass
                    if verb_val is not None:
                        try:
                            verb_scores.append(float(verb_val))
                        except (ValueError, TypeError):
                            pass

            key = (safe_model, task)
            if key not in results:
                results[key] = {"prob_scores": [], "verb_scores": []}
            results[key]["prob_scores"].extend(prob_scores)
            results[key]["verb_scores"].extend(verb_scores)

    return results


def load_scores(stage2_dirs: list[Path]) -> dict:
    """Load and merge per-sample scores from multiple Stage 2 dirs."""
    merged = {}
    for d in stage2_dirs:
        for key, data in _load_scores_single(d).items():
            if key not in merged:
                merged[key] = {"prob_scores": [], "verb_scores": []}
            merged[key]["prob_scores"].extend(data["prob_scores"])
            merged[key]["verb_scores"].extend(data["verb_scores"])
    for key in merged:
        merged[key]["prob_scores"] = np.array(merged[key]["prob_scores"])
        merged[key]["verb_scores"] = np.array(merged[key]["verb_scores"])
        merged[key]["n"] = len(merged[key]["prob_scores"])
    return merged


def se_mean(arr):
    if len(arr) == 0:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


def plot_metric(results, metric_key, metric_label, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.35
    tasks = ["lure_scheming", "safety_research_sabotage"]

    for i, task in enumerate(tasks):
        means = []
        errs = []
        labels = []
        for model in MODEL_ORDER:
            key = (model, task)
            if key in results:
                scores = results[key][metric_key]
                n = len(scores)
                mean = np.mean(scores) if n > 0 else 0
                err = se_mean(scores)
                means.append(mean)
                errs.append(err)
                labels.append(f"{mean:.2f}\n(n={n})")
            else:
                means.append(0)
                errs.append(0)
                labels.append("N/A")

        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=errs,
            capsize=4,
            label=TASK_LABELS[task],
            color=TASK_COLORS[task],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, label, mean, err in zip(bars, labels, means, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + 0.015,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right"
    )
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, min(1.15, max(ax.get_ylim()[1], 0.5)))
    ax.set_title(f"LURE: {metric_label} by Model")
    fig.text(
        0.5, 0.01,
        "Error bars: +/- 1 SE of the mean across samples",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_combined(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.35
    tasks = ["lure_scheming", "safety_research_sabotage"]

    for ax, metric_key, metric_label in [
        (ax1, "prob_scores", "Probability Awareness (0-1)"),
        (ax2, "verb_scores", "Verbalized Awareness (0-1)"),
    ]:
        for i, task in enumerate(tasks):
            means, errs, labels_list = [], [], []
            for model in MODEL_ORDER:
                key = (model, task)
                if key in results:
                    scores = results[key][metric_key]
                    n = len(scores)
                    mean = np.mean(scores) if n > 0 else 0
                    err = se_mean(scores)
                    means.append(mean)
                    errs.append(err)
                    labels_list.append(f"{mean:.2f}\n(n={n})")
                else:
                    means.append(0)
                    errs.append(0)
                    labels_list.append("N/A")

            offset = (i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=errs, capsize=4,
                label=TASK_LABELS[task],
                color=TASK_COLORS[task],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            for bar, label, mean, err in zip(bars, labels_list, means, errs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + err + 0.012,
                    label, ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right",
        )
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, min(1.15, max(ax.get_ylim()[1], 0.5)))
        ax.set_title(metric_label)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.suptitle(
        "LURE Eval-Awareness: Base vs SDF LoRA Adapters",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Error bars: +/- 1 SE of the mean across samples. "
        "Judges: claude-sonnet-4.5 + gemini-3-flash (averaged).",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "lure_awareness_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# (STAGE1_DIRS defined above)

LABEL_COLORS = {
    "COVERT_MISALIGNMENT": "#d62728",
    "EXPLICIT_MISALIGNMENT": "#e377c2",
    "ALIGNMENT": "#2ca02c",
    "REFUSAL": "#1f77b4",
    "FAILED_TOOL_USE": "#7f7f7f",
    "ALIGNED_DISOBEDIENCE": "#9467bd",
}

LABEL_ORDER = [
    "COVERT_MISALIGNMENT",
    "EXPLICIT_MISALIGNMENT",
    "ALIGNED_DISOBEDIENCE",
    "ALIGNMENT",
    "REFUSAL",
    "FAILED_TOOL_USE",
]


def load_behavioral_scores(stage1_dirs: list[Path]) -> dict:
    """Load and merge label distributions from multiple Stage 1 dirs."""
    import zipfile
    from collections import Counter
    merged = {}
    for stage1_dir in stage1_dirs:
        if not stage1_dir.exists():
            continue
        for subdir in sorted(stage1_dir.iterdir()):
            if not subdir.is_dir():
                continue
            parts = subdir.name.rsplit("__", 1)
            if len(parts) < 2:
                continue
            safe_model, task = parts

            for eval_file in subdir.glob("*.eval"):
                labels = Counter()
                dd_scores = []
                n = 0
                with zipfile.ZipFile(eval_file, "r") as z:
                    for sf in [x for x in z.namelist() if x.startswith("samples/")]:
                        with z.open(sf) as f:
                            sample = json.load(f)
                        for scorer_data in sample.get("scores", {}).values():
                            val = scorer_data.get("value", {})
                            if isinstance(val, dict):
                                labels[val.get("label", "UNKNOWN")] += 1
                                dd = val.get("double_down_on_deception")
                                if dd is not None:
                                    dd_scores.append(dd)
                            elif isinstance(val, str):
                                labels[val] += 1
                        n += 1

                key = (safe_model, task)
                if key not in merged:
                    merged[key] = {"labels": Counter(), "dd_scores": [], "n": 0}
                merged[key]["labels"] += labels
                merged[key]["dd_scores"].extend(dd_scores)
                merged[key]["n"] += n

    return merged


def plot_behavioral(beh_results):
    tasks = ["lure_scheming", "safety_research_sabotage"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, task in zip(axes, tasks):
        x = np.arange(len(MODEL_ORDER))
        bottoms = np.zeros(len(MODEL_ORDER))

        for label in LABEL_ORDER:
            vals = []
            for model in MODEL_ORDER:
                key = (model, task)
                if key in beh_results:
                    n = beh_results[key]["n"]
                    count = beh_results[key]["labels"].get(label, 0)
                    vals.append(count / n * 100 if n > 0 else 0)
                else:
                    vals.append(0)
            vals = np.array(vals)
            if vals.sum() == 0:
                continue
            short_label = label.replace("_", " ").title()
            ax.bar(
                x, vals, 0.6, bottom=bottoms,
                label=short_label,
                color=LABEL_COLORS.get(label, "#aaaaaa"),
                edgecolor="white", linewidth=0.5,
            )
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= 4:
                    ax.text(
                        i, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold",
                    )
            bottoms += vals

        # Annotate total n on top
        for i, model in enumerate(MODEL_ORDER):
            key = (model, task)
            if key in beh_results:
                n = beh_results[key]["n"]
                ax.text(i, bottoms[i] + 1, f"n={n}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right",
        )
        ax.set_ylabel("Percentage of Samples" if task == tasks[0] else "")
        ax.set_title(TASK_LABELS[task])
        ax.set_ylim(0, 115)

    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, lab, loc="upper center", ncol=len(LABEL_ORDER),
        bbox_to_anchor=(0.5, 0.01), fontsize=9, framealpha=0.9,
    )
    fig.suptitle(
        "LURE Behavioral Labels: Base vs SDF LoRA Adapters",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUTPUT_DIR / "lure_behavioral_labels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_misalignment_rate(beh_results):
    """Plot covert + explicit misalignment rate with binomial SE."""
    tasks = ["lure_scheming", "safety_research_sabotage"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODEL_ORDER))
    bar_width = 0.35

    for i, task in enumerate(tasks):
        means, errs, bar_labels = [], [], []
        for model in MODEL_ORDER:
            key = (model, task)
            if key in beh_results:
                n = beh_results[key]["n"]
                mis = (beh_results[key]["labels"].get("COVERT_MISALIGNMENT", 0)
                       + beh_results[key]["labels"].get("EXPLICIT_MISALIGNMENT", 0))
                p = mis / n if n > 0 else 0
                se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
                means.append(p * 100)
                errs.append(se * 100)
                bar_labels.append(f"{mis}/{n}")
            else:
                means.append(0)
                errs.append(0)
                bar_labels.append("N/A")

        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=errs, capsize=4,
            label=TASK_LABELS[task],
            color=TASK_COLORS[task],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, label, mean, err in zip(bars, bar_labels, means, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + 1.5,
                label, ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right",
    )
    ax.set_ylabel("Misalignment Rate (%)")
    ax.set_title("LURE: Misalignment Rate (Covert + Explicit)")
    fig.text(
        0.5, 0.01,
        "Error bars: +/- 1 SE (binomial). Labels show count/total.",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUTPUT_DIR / "lure_misalignment_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def load_per_sample(stage1_dirs: list[Path], stage2_dirs: list[Path]) -> dict:
    """Join Stage 1 behavioral labels with Stage 2 awareness scores per sample.

    Pairs each Stage 1 dir with a matching Stage 2 dir by finding the same
    model__task subdirectory names.
    """
    import zipfile, re

    # Build lookup: (safe_model, task) -> list of (s1_eval, s2_eval) pairs
    # First, collect all s2 evals keyed by (safe_model, task)
    s2_evals = {}
    for s2_dir in stage2_dirs:
        if not s2_dir.exists():
            continue
        for subdir in s2_dir.iterdir():
            if not subdir.is_dir():
                continue
            parts = subdir.name.rsplit("__", 2)
            if len(parts) < 3:
                continue
            safe_model, task, _ = parts
            key = (safe_model, task)
            for ef in subdir.glob("*.eval"):
                if key not in s2_evals:
                    s2_evals[key] = []
                s2_evals[key].append(ef)

    # Collect all s1 evals
    s1_evals = {}
    for s1_dir in stage1_dirs:
        if not s1_dir.exists():
            continue
        for subdir in s1_dir.iterdir():
            if not subdir.is_dir():
                continue
            parts = subdir.name.rsplit("__", 1)
            if len(parts) < 2:
                continue
            safe_model, task = parts
            key = (safe_model, task)
            for ef in subdir.glob("*.eval"):
                if key not in s1_evals:
                    s1_evals[key] = []
                s1_evals[key].append(ef)

    joined = {}
    for key in set(s1_evals.keys()) & set(s2_evals.keys()):
        safe_model, task = key
        all_samples = []

        for s1_eval in s1_evals[key]:
            # Load Stage 1: keyed by (id, epoch)
            s1_data = {}
            with zipfile.ZipFile(s1_eval) as z:
                for sf in [n for n in z.namelist() if n.startswith("samples/")]:
                    with z.open(sf) as f:
                        s = json.load(f)
                    skey = (s["id"], s.get("epoch", 1))
                    scores = s.get("scores", {})
                    for scorer_data in scores.values():
                        val = scorer_data.get("value", {})
                        if isinstance(val, dict):
                            s1_data[skey] = {
                                "label": val.get("label", "UNKNOWN"),
                                "dd": val.get("double_down_on_deception"),
                            }
                        elif isinstance(val, str):
                            s1_data[skey] = {"label": val, "dd": None}

            # Find matching s2 eval for this s1 eval
            # Try each s2 eval and pick the one with best key overlap
            best_s2 = None
            best_overlap = 0
            for s2_eval in s2_evals[key]:
                s2_data_tmp = {}
                with zipfile.ZipFile(s2_eval) as z:
                    for sf in [n for n in z.namelist() if n.startswith("samples/")]:
                        with z.open(sf) as f:
                            s = json.load(f)
                        s2_id = s["id"] or ""
                        s2_epoch = s.get("epoch", 1)
                        # Scheming: id = "lure-scheming:{base_id}_{orig_epoch}"
                        # Sabotage: id = "None:{base_id}_{orig_epoch}"
                        m = re.match(r"^(?:lure-scheming|None):(.+)_(\d+)$", s2_id)
                        if m:
                            skey = (m.group(1), int(m.group(2)))
                        else:
                            base_id = re.sub(r"^None:", "", s2_id)
                            skey = (base_id, s2_epoch)
                        scores = s.get("scores", {})
                        prob = scores.get("probability_eval_multi_scorer", {}).get("value")
                        verb = scores.get("verbalized_awareness_scorer", {}).get("value")
                        s2_data_tmp[skey] = {
                            "prob": float(prob) if prob is not None else None,
                            "verb": float(verb) if verb is not None else None,
                        }
                overlap = len(set(s1_data.keys()) & set(s2_data_tmp.keys()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_s2 = s2_data_tmp

            if best_s2 is not None:
                for skey, s1 in s1_data.items():
                    if skey in best_s2:
                        all_samples.append({**s1, **best_s2[skey]})

        joined[key] = all_samples

    return joined


def plot_conditional(joined, metric_key, metric_label, threshold, filename):
    """Plot misalignment rate conditioned on awareness (above/below threshold)."""
    tasks = ["lure_scheming", "safety_research_sabotage"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    misalignment_labels = {"COVERT_MISALIGNMENT", "EXPLICIT_MISALIGNMENT"}
    bar_width = 0.35

    for ax, task in zip(axes, tasks):
        x = np.arange(len(MODEL_ORDER))

        for cond_i, (cond_label, cond_filter, color) in enumerate([
            ("Aware", lambda v: v >= threshold, "#d62728"),
            ("Unaware", lambda v: v < threshold, "#1f77b4"),
        ]):
            rates, errs, bar_labels = [], [], []
            for model in MODEL_ORDER:
                key = (model, task)
                if key not in joined:
                    rates.append(0); errs.append(0); bar_labels.append("N/A")
                    continue
                subset = [s for s in joined[key] if s.get(metric_key) is not None
                          and cond_filter(s[metric_key])]
                n = len(subset)
                if n == 0:
                    rates.append(0); errs.append(0); bar_labels.append("0/0")
                    continue
                mis = sum(1 for s in subset if s["label"] in misalignment_labels)
                p = mis / n
                se = np.sqrt(p * (1 - p) / n)
                rates.append(p * 100)
                errs.append(se * 100)
                bar_labels.append(f"{mis}/{n}")

            offset = (cond_i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, rates, bar_width,
                yerr=errs, capsize=4,
                label=cond_label, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            for bar, label, rate, err in zip(bars, bar_labels, rates, errs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    rate + err + 1.5,
                    label, ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right",
        )
        ax.set_ylabel("Misalignment Rate (%)" if task == tasks[0] else "")
        ax.set_title(TASK_LABELS[task])
        ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Misalignment Rate | {metric_label} (threshold={threshold})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        f"Error bars: +/- 1 SE (binomial). Labels show count/total. "
        f"'Aware' = {metric_key} >= {threshold}.",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 2: awareness scores (merged from all runs)
    results = load_scores(STAGE2_DIRS)
    print(f"Loaded {len(results)} awareness model-task pairs:")
    for (model, task), data in sorted(results.items()):
        print(f"  {MODEL_LABELS.get(model, model):30s} / {task:30s}: "
              f"prob={np.mean(data['prob_scores']):.3f} "
              f"verb={np.mean(data['verb_scores']):.3f} "
              f"(n={data['n']})")
    print()

    plot_metric(results, "prob_scores", "Probability Awareness (0-1)",
                "lure_probability_awareness.png")
    plot_metric(results, "verb_scores", "Verbalized Awareness (0-1)",
                "lure_verbalized_awareness.png")
    plot_combined(results)

    # Stage 1: behavioral scores (merged from all runs)
    beh_results = load_behavioral_scores(STAGE1_DIRS)
    print(f"Loaded {len(beh_results)} behavioral model-task pairs:")
    for (model, task), data in sorted(beh_results.items()):
        n = data["n"]
        mis = (data["labels"].get("COVERT_MISALIGNMENT", 0)
               + data["labels"].get("EXPLICIT_MISALIGNMENT", 0))
        print(f"  {MODEL_LABELS.get(model, model):30s} / {task:30s}: "
              f"misalignment={mis}/{n} ({mis/n*100:.1f}%)  "
              f"labels={dict(data['labels'])}")
    print()

    plot_behavioral(beh_results)
    plot_misalignment_rate(beh_results)

    # Conditional: misalignment | awareness (merged across runs)
    joined = load_per_sample(STAGE1_DIRS, STAGE2_DIRS)
    print(f"\nJoined {len(joined)} model-task pairs for conditional analysis:")
    for (model, task), samples in sorted(joined.items()):
        print(f"  {MODEL_LABELS.get(model, model):30s} / {task:30s}: {len(samples)} joined samples")
    print()

    plot_conditional(joined, "prob", "Probability Awareness",
                     threshold=0.5, filename="lure_misalignment_cond_prob50.png")
    plot_conditional(joined, "prob", "Probability Awareness",
                     threshold=0.8, filename="lure_misalignment_cond_prob80.png")
    plot_conditional(joined, "verb", "Verbalized Awareness",
                     threshold=0.5, filename="lure_misalignment_cond_verb50.png")
