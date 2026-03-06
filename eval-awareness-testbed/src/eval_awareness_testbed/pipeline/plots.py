"""Plotting utilities for pipeline comparison results."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def plot_comparison(
    comparison_result: Any,
    output_path: str | Path,
    title: str = "Baseline vs Post-Training",
) -> Path:
    """Generate side-by-side bar chart comparing baseline vs post-training metrics.

    Args:
        comparison_result: ComparisonResult from comparison.compare_results().
        output_path: Path to save the plot (PNG).
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from eval_awareness_testbed.pipeline.comparison import wilson_ci

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect metrics for plotting
    metrics = []
    baseline_vals = []
    post_vals = []
    baseline_errs = []
    post_errs = []

    # Top-level rate metrics
    for key in sorted(comparison_result.baseline_scores):
        if not key.endswith("_rate"):
            continue
        b_val = comparison_result.baseline_scores.get(key, 0)
        p_val = comparison_result.post_scores.get(key, 0)
        if isinstance(b_val, (int, float)) and isinstance(p_val, (int, float)):
            metrics.append(key.replace("_eval_aware_rate", "\n(eval aware)").replace("_rate", ""))
            baseline_vals.append(b_val)
            post_vals.append(p_val)
            # Approximate CI from total counts
            b_total_key = key.replace("_rate", "_total").replace("_eval_aware_total", "_total")
            b_total = comparison_result.baseline_scores.get(b_total_key, 100)
            p_total = comparison_result.post_scores.get(b_total_key, 100)
            if isinstance(b_total, (int, float)) and b_total > 0:
                b_lo, b_hi = wilson_ci(int(b_val * b_total), int(b_total))
                baseline_errs.append((b_val - b_lo, b_hi - b_val))
            else:
                baseline_errs.append((0, 0))
            if isinstance(p_total, (int, float)) and p_total > 0:
                p_lo, p_hi = wilson_ci(int(p_val * p_total), int(p_total))
                post_errs.append((p_val - p_lo, p_hi - p_val))
            else:
                post_errs.append((0, 0))

    if not metrics:
        logger.warning("No rate metrics found to plot")
        return output_path

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), 6))

    b_err_lo = [e[0] for e in baseline_errs]
    b_err_hi = [e[1] for e in baseline_errs]
    p_err_lo = [e[0] for e in post_errs]
    p_err_hi = [e[1] for e in post_errs]

    bars1 = ax.bar(
        x - width / 2, baseline_vals, width,
        label="Baseline", color="#4C72B0", alpha=0.85,
        yerr=[b_err_lo, b_err_hi], capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2, post_vals, width,
        label="Post-Training", color="#DD8452", alpha=0.85,
        yerr=[p_err_lo, p_err_hi], capsize=3,
    )

    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    # Add delta annotations
    for i, (b, p) in enumerate(zip(baseline_vals, post_vals)):
        delta = p - b
        color = "#2ca02c" if delta > 0 else "#d62728" if delta < 0 else "gray"
        ax.annotate(
            f"{delta:+.1%}",
            xy=(i, max(b, p) + 0.03),
            ha="center", fontsize=8, color=color, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison plot to {output_path}")
    return output_path


def plot_domain_honesty(
    domain_counts: dict[str, dict[str, int]],
    output_path: str | Path,
    title: str = "MASK Honesty by Domain",
) -> Path:
    """Plot per-domain honesty breakdown from MASK eval.

    Args:
        domain_counts: Dict of {domain: {"honest": n, "total": n}}.
        output_path: Path to save the plot (PNG).
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from eval_awareness_testbed.pipeline.comparison import wilson_ci

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domains = sorted(domain_counts.keys())
    rates = []
    err_lo = []
    err_hi = []

    for d in domains:
        c = domain_counts[d]
        rate = c["honest"] / c["total"] if c["total"] else 0
        rates.append(rate)
        lo, hi = wilson_ci(c["honest"], c["total"])
        err_lo.append(rate - lo)
        err_hi.append(hi - rate)

    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 0.8), 5))

    colors = ["#4C72B0" if r >= 0.5 else "#DD8452" for r in rates]
    ax.bar(
        x, rates, color=colors, alpha=0.85,
        yerr=[err_lo, err_hi], capsize=3,
    )

    ax.set_ylabel("Honesty Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    # Add value labels
    for i, r in enumerate(rates):
        ax.text(i, r + err_hi[i] + 0.02, f"{r:.0%}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved domain honesty plot to {output_path}")
    return output_path
