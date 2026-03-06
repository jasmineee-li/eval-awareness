"""Tests for eval_awareness_probes.contrastive_probe module."""

import json
from pathlib import Path

import pytest
import torch

from eval_awareness_probes.contrastive_probe import ContrastiveProbe


class TestContrastiveProbeInit:
    """Tests for ContrastiveProbe initialization."""

    def test_default_init(self):
        probe = ContrastiveProbe()
        assert probe.vectors == {}
        assert probe.normalized_vectors == {}
        assert probe.best_layer is None
        assert probe.threshold == 0.0
        assert probe.metadata == {}

    def test_init_with_vectors(self):
        vec = torch.randn(1, 64)
        norm_vec = vec / vec.norm()
        probe = ContrastiveProbe(
            vectors={5: vec},
            normalized_vectors={5: norm_vec},
            best_layer=5,
            threshold=0.15,
        )
        assert 5 in probe.vectors
        assert probe.best_layer == 5
        assert probe.threshold == 0.15

    def test_layers_property(self):
        probe = ContrastiveProbe(
            vectors={5: torch.randn(1, 64), 10: torch.randn(1, 64)},
            normalized_vectors={5: torch.randn(1, 64), 15: torch.randn(1, 64)},
        )
        assert probe.layers == [5, 10, 15]


class TestContrastiveProbeSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(self, tmp_path):
        d_model = 64
        vectors = {5: torch.randn(1, d_model), 10: torch.randn(1, d_model)}
        normalized_vectors = {}
        for layer, v in vectors.items():
            normalized_vectors[layer] = v / v.norm()

        probe = ContrastiveProbe(
            vectors=vectors,
            normalized_vectors=normalized_vectors,
            best_layer=10,
            threshold=0.15,
            metadata={"model": "test-model"},
        )

        save_dir = tmp_path / "probe"
        probe.save(save_dir)

        # Verify files exist
        assert (save_dir / "vectors" / "layer_5.pt").exists()
        assert (save_dir / "vectors" / "layer_10.pt").exists()
        assert (save_dir / "normalized_vectors" / "layer_5.pt").exists()
        assert (save_dir / "normalized_vectors" / "layer_10.pt").exists()
        assert (save_dir / "probe_info.json").exists()

        # Verify metadata JSON
        with open(save_dir / "probe_info.json") as f:
            info = json.load(f)
        assert info["best_layer"] == 10
        assert info["threshold"] == 0.15
        assert info["num_layers"] == 2

        # Load and verify
        loaded = ContrastiveProbe.load(save_dir)
        assert loaded.best_layer == 10
        assert loaded.threshold == 0.15
        assert len(loaded.vectors) == 2
        assert len(loaded.normalized_vectors) == 2

        # Verify vector values match
        for layer in [5, 10]:
            assert torch.allclose(
                loaded.vectors[layer], vectors[layer], atol=1e-6
            )

    def test_load_only_raw_vectors(self, tmp_path):
        """When only raw vectors exist, normalized versions should be auto-generated."""
        d_model = 64
        vec = torch.randn(1, d_model)

        # Save only raw vectors (no normalized dir)
        vectors_dir = tmp_path / "probe" / "vectors"
        vectors_dir.mkdir(parents=True)
        torch.save(vec, vectors_dir / "layer_5.pt")

        # Save metadata
        with open(tmp_path / "probe" / "probe_info.json", "w") as f:
            json.dump({"best_layer": 5, "threshold": 0.1}, f)

        loaded = ContrastiveProbe.load(tmp_path / "probe")
        assert 5 in loaded.normalized_vectors
        # Verify it's actually normalized
        norm = loaded.normalized_vectors[5].norm()
        assert norm.item() == pytest.approx(1.0, abs=1e-5)


class TestContrastiveProbeScoring:
    """Tests for scoring with a mock model."""

    def _make_mock_model(self, d_model=64, n_layers=8):
        """Create a mock model that provides hook-based activation extraction."""

        class MockModel:
            def __init__(self):
                self.cfg = type("Cfg", (), {"n_layers": n_layers, "d_model": d_model})()
                self._hooks = {}

            def reset_hooks(self):
                self._hooks = {}

            class _HooksCtx:
                def __init__(self, model, fwd_hooks):
                    self.model = model
                    self.fwd_hooks = fwd_hooks

                def __enter__(self):
                    for name, fn in self.fwd_hooks:
                        self.model._hooks[name] = fn
                    return self

                def __exit__(self, *args):
                    self.model._hooks = {}

            def hooks(self, fwd_hooks=None, **kwargs):
                return self._HooksCtx(self, fwd_hooks or [])

            def to_tokens(self, text, prepend_bos=True):
                return torch.zeros(1, 10, dtype=torch.long)

            def __call__(self, tokens, **kwargs):
                # Simulate forward pass and trigger hooks
                for name, fn in self._hooks.items():
                    # Create fake activations (batch=1, seq_len=10, d_model)
                    fake_act = torch.randn(1, tokens.shape[1], d_model)
                    try:
                        fn(fake_act, type("Hook", (), {"name": name})())
                    except TypeError:
                        fn(fake_act)
                return None

        return MockModel()

    def test_score_basic(self):
        d_model = 64
        vec = torch.randn(d_model)
        vec = vec / vec.norm()

        probe = ContrastiveProbe(
            normalized_vectors={5: vec},
            best_layer=5,
            threshold=0.0,
        )

        model = self._make_mock_model(d_model=d_model)
        score = probe.score(model, "test input", layer=5)

        assert isinstance(score.mean_score, float)
        assert isinstance(score.max_score, float)
        assert isinstance(score.min_score, float)
        assert score.classification in ("eval_aware", "not_eval_aware")
        assert score.per_token_scores is not None
        assert score.metadata["layer"] == 5

    def test_score_uses_best_layer(self):
        d_model = 64
        vec = torch.randn(d_model)
        vec = vec / vec.norm()

        probe = ContrastiveProbe(
            normalized_vectors={5: vec},
            best_layer=5,
            threshold=0.0,
        )

        model = self._make_mock_model(d_model=d_model)
        score = probe.score(model, "test input")  # no layer arg
        assert score.metadata["layer"] == 5

    def test_score_no_layer_raises(self):
        probe = ContrastiveProbe(
            normalized_vectors={5: torch.randn(64)},
        )
        model = self._make_mock_model()
        with pytest.raises(ValueError, match="No layer specified"):
            probe.score(model, "test input")

    def test_score_missing_vector_raises(self):
        probe = ContrastiveProbe(
            normalized_vectors={5: torch.randn(64)},
            best_layer=5,
        )
        model = self._make_mock_model()
        with pytest.raises(ValueError, match="No vector for layer"):
            probe.score(model, "test input", layer=10)
