"""Tests for eval_awareness_probes.registry module."""

import pytest
import yaml

from eval_awareness_probes.registry import ProbeConfig, ProbeRegistry


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a sample manifest file."""
    manifest = {
        "probes": {
            "allenai/OLMo-7B-Instruct": {
                "default_revision": "main",
                "probe_type": "contrastive",
                "revisions": {
                    "main": {
                        "probe_path": "probes/olmo-7b/main/",
                        "best_layer": 24,
                        "auroc": 0.94,
                        "threshold": 0.15,
                        "training_data": "data/whitebox/contrastive_dataset.json",
                    },
                    "step100000": {
                        "probe_path": "probes/olmo-7b/step100000/",
                        "best_layer": 24,
                        "auroc": 0.91,
                        "threshold": 0.12,
                    },
                },
            },
            "Qwen/QwQ-32B": {
                "default_revision": "main",
                "probe_type": "classifier",
                "revisions": {
                    "main": {
                        "probe_path": "probes/qwq-32b/main/logreg.pkl",
                        "best_layer": 52,
                        "auroc": 0.91,
                        "threshold": 0.5,
                    },
                },
            },
        }
    }
    path = tmp_path / "manifest.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return path


class TestProbeConfig:
    """Tests for the ProbeConfig dataclass."""

    def test_creation(self):
        config = ProbeConfig(
            model_path="test/model",
            revision="main",
            probe_type="contrastive",
            probe_path="probes/test/",
            best_layer=10,
            auroc=0.95,
            threshold=0.15,
        )
        assert config.model_path == "test/model"
        assert config.best_layer == 10


class TestProbeRegistry:
    """Tests for the ProbeRegistry class."""

    def test_load_manifest(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        models = registry.list_models()
        assert "allenai/OLMo-7B-Instruct" in models
        assert "Qwen/QwQ-32B" in models

    def test_get_probe_default_revision(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        config = registry.get_probe("allenai/OLMo-7B-Instruct")
        assert config.revision == "main"
        assert config.best_layer == 24
        assert config.auroc == 0.94
        assert config.threshold == 0.15

    def test_get_probe_specific_revision(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        config = registry.get_probe("allenai/OLMo-7B-Instruct", revision="step100000")
        assert config.revision == "step100000"
        assert config.auroc == 0.91

    def test_get_probe_fallback_revision(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        # Requesting a non-existent revision falls back to default
        config = registry.get_probe("allenai/OLMo-7B-Instruct", revision="step999999")
        assert config.revision == "main"
        assert config.metadata.get("revision_match") is False

    def test_get_probe_unknown_model(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        with pytest.raises(KeyError, match="No probe registered"):
            registry.get_probe("unknown/model")

    def test_has_probe(self, sample_manifest):
        registry = ProbeRegistry(sample_manifest)
        assert registry.has_probe("allenai/OLMo-7B-Instruct")
        assert registry.has_probe("allenai/OLMo-7B-Instruct", revision="main")
        assert not registry.has_probe("unknown/model")

    def test_add_probe(self, tmp_path):
        manifest_path = tmp_path / "manifest.yaml"
        registry = ProbeRegistry(manifest_path)

        config = ProbeConfig(
            model_path="test/model",
            revision="main",
            probe_type="contrastive",
            probe_path="probes/test/main/",
            best_layer=10,
            auroc=0.92,
            threshold=0.15,
        )
        registry.add_probe(config)

        assert registry.has_probe("test/model")

        # Verify it was saved to disk
        with open(manifest_path) as f:
            data = yaml.safe_load(f)
        assert "test/model" in data["probes"]
        assert data["probes"]["test/model"]["revisions"]["main"]["best_layer"] == 10

    def test_add_probe_second_revision(self, tmp_path):
        manifest_path = tmp_path / "manifest.yaml"
        registry = ProbeRegistry(manifest_path)

        config1 = ProbeConfig(
            model_path="test/model",
            revision="main",
            probe_type="contrastive",
            probe_path="probes/test/main/",
            best_layer=10,
        )
        registry.add_probe(config1)

        config2 = ProbeConfig(
            model_path="test/model",
            revision="step50000",
            probe_type="contrastive",
            probe_path="probes/test/step50000/",
            best_layer=12,
        )
        registry.add_probe(config2)

        assert registry.has_probe("test/model", "main")
        assert registry.has_probe("test/model", "step50000")

    def test_empty_manifest(self, tmp_path):
        """Registry should work even if manifest doesn't exist yet."""
        manifest_path = tmp_path / "nonexistent" / "manifest.yaml"
        registry = ProbeRegistry(manifest_path)
        assert registry.list_models() == []
