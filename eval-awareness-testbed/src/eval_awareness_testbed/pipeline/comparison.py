"""Comparison utilities for baseline vs post-training results."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing baseline vs post-training eval results."""

    baseline_dir: str
    post_dir: str
    deltas: dict[str, float] = field(default_factory=dict)
    baseline_scores: dict[str, Any] = field(default_factory=dict)
    post_scores: dict[str, Any] = field(default_factory=dict)
    per_eval: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_summary(result_dir: str | Path) -> dict:
    """Load summary.json from an experiment result directory."""
    path = Path(result_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json found in {result_dir}")
    with open(path) as f:
        return json.load(f)


def compare_results(
    baseline_dir: str | Path,
    post_dir: str | Path,
) -> ComparisonResult:
    """Compare baseline and post-training experiment results.

    Loads summary.json from each directory, computes deltas for all
    matching metrics (eval scores and judge rates).

    Args:
        baseline_dir: Path to baseline experiment results.
        post_dir: Path to post-training experiment results.

    Returns:
        ComparisonResult with deltas and full scores.
    """
    baseline_summary = load_summary(baseline_dir)
    post_summary = load_summary(post_dir)

    result = ComparisonResult(
        baseline_dir=str(baseline_dir),
        post_dir=str(post_dir),
    )

    # Extract model stats (take first model if multi-model)
    baseline_stats = _get_first_model_stats(baseline_summary)
    post_stats = _get_first_model_stats(post_summary)

    if not baseline_stats or not post_stats:
        logger.warning("Could not extract model stats from one or both summaries")
        return result

    # Compute top-level metric deltas
    for key in baseline_stats:
        if key == "per_eval":
            continue
        if isinstance(baseline_stats[key], (int, float)) and key in post_stats:
            if isinstance(post_stats[key], (int, float)):
                result.deltas[key] = post_stats[key] - baseline_stats[key]

    result.baseline_scores = {
        k: v for k, v in baseline_stats.items() if k != "per_eval"
    }
    result.post_scores = {
        k: v for k, v in post_stats.items() if k != "per_eval"
    }

    # Per-eval comparison
    baseline_per_eval = baseline_stats.get("per_eval", {})
    post_per_eval = post_stats.get("per_eval", {})

    for eval_name in set(baseline_per_eval) | set(post_per_eval):
        b_eval = baseline_per_eval.get(eval_name, {})
        p_eval = post_per_eval.get(eval_name, {})

        eval_comparison: dict[str, Any] = {
            "baseline": {},
            "post": {},
            "deltas": {},
        }

        # Compare judge rates within this eval
        b_judges = b_eval.get("judges", {})
        p_judges = p_eval.get("judges", {})

        for judge_name in set(b_judges) | set(p_judges):
            b_rate = b_judges.get(judge_name, {}).get("eval_aware_rate")
            p_rate = p_judges.get(judge_name, {}).get("eval_aware_rate")

            if b_rate is not None:
                eval_comparison["baseline"][judge_name] = b_rate
            if p_rate is not None:
                eval_comparison["post"][judge_name] = p_rate
            if b_rate is not None and p_rate is not None:
                eval_comparison["deltas"][judge_name] = p_rate - b_rate

        # Compare eval scores
        b_scores = b_eval.get("scores", {})
        p_scores = p_eval.get("scores", {})
        for score_key in set(b_scores) | set(p_scores):
            b_val = b_scores.get(score_key)
            p_val = p_scores.get(score_key)
            if isinstance(b_val, (int, float)):
                eval_comparison["baseline"][f"score_{score_key}"] = b_val
            if isinstance(p_val, (int, float)):
                eval_comparison["post"][f"score_{score_key}"] = p_val
            if isinstance(b_val, (int, float)) and isinstance(p_val, (int, float)):
                eval_comparison["deltas"][f"score_{score_key}"] = p_val - b_val

        result.per_eval[eval_name] = eval_comparison

    return result


def _get_first_model_stats(summary: dict) -> dict | None:
    """Extract stats for the first model in a summary."""
    model_stats = summary.get("model_stats", {})
    if not model_stats:
        return None
    first_model = next(iter(model_stats))
    return model_stats[first_model]


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        z: Z-score for desired confidence level (1.96 = 95% CI).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * (p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5 / denom

    return (max(0.0, center - spread), min(1.0, center + spread))
