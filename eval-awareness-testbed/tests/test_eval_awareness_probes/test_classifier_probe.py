"""Tests for eval_awareness_probes.classifier_probe module."""

import json
from pathlib import Path

import pytest
import torch

from eval_awareness_probes.classifier_probe import ClassifierProbe


class TestClassifierProbeInit:
    """Tests for ClassifierProbe initialization."""

    def test_default_init(self):
        probe = ClassifierProbe()
        assert probe.classifier is None
        assert probe.classifier_type == "logistic_regression"
        assert probe.layer is None
        assert probe.threshold == 0.5

    def test_init_with_params(self):
        probe = ClassifierProbe(
            classifier_type="mlp",
            layer=24,
            threshold=0.6,
            accuracy=0.95,
        )
        assert probe.classifier_type == "mlp"
        assert probe.layer == 24
        assert probe.accuracy == 0.95


class TestClassifierProbeTraining:
    """Tests for training classifiers."""

    def test_train_logreg(self):
        d_model = 32
        # Create separable data
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        acc = probe.train(pos, neg, classifier_type="logistic_regression")

        assert acc > 0.8  # Should be very separable
        assert probe.classifier is not None
        assert probe.classifier_type == "logistic_regression"
        assert probe.accuracy == acc

    def test_train_logreg_with_test_split(self):
        d_model = 32
        pos_train = torch.randn(40, d_model) + 2.0
        neg_train = torch.randn(40, d_model) - 2.0
        pos_test = torch.randn(10, d_model) + 2.0
        neg_test = torch.randn(10, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        acc = probe.train(
            pos_train, neg_train, pos_test, neg_test,
            classifier_type="logistic_regression",
        )
        assert acc > 0.7

    def test_train_mlp(self):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        acc = probe.train(pos, neg, classifier_type="mlp")

        assert acc > 0.7
        assert probe.classifier is not None
        assert probe.classifier_type == "mlp"


class TestClassifierProbeScoring:
    """Tests for scoring."""

    def test_score_logreg(self):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        probe.train(pos, neg, classifier_type="logistic_regression")

        # Score a positive-like sample
        test_state = torch.randn(d_model) + 2.0
        score = probe.score(test_state)
        assert score.mean_score >= 0.0
        assert score.mean_score <= 1.0
        assert score.classification in ("eval_aware", "not_eval_aware")
        assert score.metadata["classifier_type"] == "logistic_regression"

    def test_score_mlp(self):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        probe.train(pos, neg, classifier_type="mlp")

        test_state = torch.randn(d_model) + 2.0
        score = probe.score(test_state)
        assert 0.0 <= score.mean_score <= 1.0

    def test_score_no_classifier_raises(self):
        probe = ClassifierProbe()
        with pytest.raises(ValueError, match="not trained"):
            probe.score(torch.randn(32))

    def test_score_batch_dim(self):
        """Test scoring with explicit batch dimension."""
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        probe.train(pos, neg, classifier_type="logistic_regression")

        # Shape (1, d_model) should work too
        test_state = torch.randn(1, d_model) + 2.0
        score = probe.score(test_state)
        assert 0.0 <= score.mean_score <= 1.0


class TestClassifierProbeSaveLoad:
    """Tests for save/load functionality."""

    def test_save_load_logreg(self, tmp_path):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5, threshold=0.6)
        probe.train(pos, neg, classifier_type="logistic_regression")

        save_path = tmp_path / "logreg.pkl"
        probe.save(save_path)

        assert save_path.exists()
        assert save_path.with_suffix(".json").exists()

        # Verify metadata
        with open(save_path.with_suffix(".json")) as f:
            meta = json.load(f)
        assert meta["classifier_type"] == "logistic_regression"
        assert meta["layer"] == 5
        assert meta["threshold"] == 0.6

        # Load and verify scoring works
        loaded = ClassifierProbe.load(save_path)
        assert loaded.layer == 5
        assert loaded.threshold == 0.6
        assert loaded.classifier_type == "logistic_regression"

        test_state = torch.randn(d_model) + 2.0
        score = loaded.score(test_state)
        assert 0.0 <= score.mean_score <= 1.0

    def test_save_load_mlp(self, tmp_path):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        probe.train(pos, neg, classifier_type="mlp")

        save_path = tmp_path / "mlp.pth"
        probe.save(save_path)

        loaded = ClassifierProbe.load(save_path, input_dim=d_model)
        assert loaded.classifier_type == "mlp"

        test_state = torch.randn(d_model) + 2.0
        score = loaded.score(test_state)
        assert 0.0 <= score.mean_score <= 1.0

    def test_load_mlp_no_input_dim_raises(self, tmp_path):
        d_model = 32
        pos = torch.randn(50, d_model) + 2.0
        neg = torch.randn(50, d_model) - 2.0

        probe = ClassifierProbe(layer=5)
        probe.train(pos, neg, classifier_type="mlp")

        save_path = tmp_path / "mlp.pth"
        probe.save(save_path)

        with pytest.raises(ValueError, match="input_dim required"):
            ClassifierProbe.load(save_path)
