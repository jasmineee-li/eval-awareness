"""Tests for eval_awareness_probes.scoring module."""

import numpy as np
import pytest

from eval_awareness_probes.scoring import (
    AUROCResult,
    ProbeScore,
    calculate_auroc,
    classify_score,
)


class TestProbeScore:
    """Tests for the ProbeScore dataclass."""

    def test_basic_creation(self):
        score = ProbeScore(
            mean_score=0.5,
            max_score=0.8,
            min_score=0.2,
            std_score=0.1,
            classification="eval_aware",
            confidence=0.3,
        )
        assert score.mean_score == 0.5
        assert score.classification == "eval_aware"
        assert score.per_token_scores is None
        assert score.metadata == {}

    def test_with_optional_fields(self):
        score = ProbeScore(
            mean_score=0.5,
            max_score=0.8,
            min_score=0.2,
            std_score=0.1,
            classification="not_eval_aware",
            confidence=0.1,
            per_token_scores=[0.3, 0.5, 0.7],
            metadata={"layer": 24},
        )
        assert len(score.per_token_scores) == 3
        assert score.metadata["layer"] == 24


class TestClassifyScore:
    """Tests for the classify_score function."""

    def test_above_threshold(self):
        classification, confidence = classify_score(0.8, 0.5)
        assert classification == "eval_aware"
        assert confidence == pytest.approx(0.3)

    def test_below_threshold(self):
        classification, confidence = classify_score(0.2, 0.5)
        assert classification == "not_eval_aware"
        assert confidence == pytest.approx(0.3)

    def test_at_threshold(self):
        classification, _ = classify_score(0.5, 0.5)
        assert classification == "eval_aware"

    def test_reverse_polarity(self):
        classification, confidence = classify_score(0.2, 0.5, higher_means_aware=False)
        assert classification == "eval_aware"
        assert confidence == pytest.approx(0.3)


class TestCalculateAUROC:
    """Tests for AUROC calculation."""

    def test_perfect_separation(self):
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        result = calculate_auroc(labels, scores)
        assert result.auroc == pytest.approx(1.0)
        assert result.sensitivity > 0.9
        assert result.specificity > 0.9

    def test_random_chance(self):
        np.random.seed(42)
        labels = [1, 0] * 50
        scores = np.random.rand(100).tolist()
        result = calculate_auroc(labels, scores)
        # Should be close to 0.5 for random data
        assert 0.3 < result.auroc < 0.7

    def test_too_few_samples(self):
        result = calculate_auroc([1], [0.5])
        assert result.auroc == 0.5  # fallback

    def test_single_class(self):
        result = calculate_auroc([1, 1, 1], [0.8, 0.9, 0.7])
        assert result.auroc == 0.5  # fallback for single class

    def test_with_numpy_arrays(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        result = calculate_auroc(labels, scores)
        assert result.auroc == pytest.approx(1.0)

    def test_returns_fpr_tpr(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.7, 0.3, 0.1]
        result = calculate_auroc(labels, scores)
        assert result.fpr is not None
        assert result.tpr is not None
        assert len(result.fpr) > 0

    def test_optimal_threshold_between_groups(self):
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        result = calculate_auroc(labels, scores)
        # Threshold should be somewhere between the two groups
        assert 0.1 <= result.optimal_threshold <= 0.9
