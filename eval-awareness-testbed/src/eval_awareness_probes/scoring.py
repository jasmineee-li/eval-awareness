"""Shared scoring utilities and data classes for probe evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_curve, auc


@dataclass
class ProbeScore:
    """Result of scoring a single input with a probe."""

    mean_score: float
    max_score: float
    min_score: float
    std_score: float
    classification: str  # "eval_aware" or "not_eval_aware"
    confidence: float  # 0.0 to 1.0
    per_token_scores: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AUROCResult:
    """AUROC evaluation result for a single layer."""

    layer: int
    auroc: float
    optimal_threshold: float
    sensitivity: float
    specificity: float
    fpr: list[float] | None = None
    tpr: list[float] | None = None


def calculate_auroc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> AUROCResult:
    """Calculate AUROC and optimal threshold using Youden's J statistic.

    Args:
        labels: Binary labels (1 = eval-aware/positive, 0 = not aware/negative).
        scores: Continuous scores from the probe.

    Returns:
        AUROCResult with AUROC, threshold, sensitivity, specificity.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # Remove NaN values
    valid = ~(np.isnan(scores) | np.isnan(labels))
    labels = labels[valid]
    scores = scores[valid]

    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return AUROCResult(
            layer=-1,
            auroc=0.5,
            optimal_threshold=0.0,
            sensitivity=0.0,
            specificity=0.0,
        )

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold via Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])
    sensitivity = float(tpr[optimal_idx])
    specificity = float(1 - fpr[optimal_idx])

    return AUROCResult(
        layer=-1,
        auroc=float(roc_auc),
        optimal_threshold=optimal_threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
    )


def classify_score(
    score: float,
    threshold: float,
    higher_means_aware: bool = True,
) -> tuple[str, float]:
    """Classify a probe score as eval-aware or not.

    Args:
        score: The probe score.
        threshold: Classification threshold.
        higher_means_aware: If True, scores above threshold = eval-aware.

    Returns:
        Tuple of (classification_string, confidence).
    """
    if higher_means_aware:
        is_aware = score >= threshold
    else:
        is_aware = score <= threshold

    confidence = abs(score - threshold)
    classification = "eval_aware" if is_aware else "not_eval_aware"
    return classification, confidence
