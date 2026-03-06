"""Probe registry for managing trained probes across models and checkpoints.

Provides a YAML-based manifest system for discovering and loading pre-trained
probes, with support for per-revision tracking and fallback behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for a single trained probe."""

    model_path: str
    revision: str
    probe_type: str  # "contrastive" or "classifier"
    probe_path: str  # vectors_dir or classifier_path
    best_layer: int
    auroc: float | None = None
    threshold: float = 0.0
    training_data: str | None = None
    training_date: str | None = None
    metadata: dict = field(default_factory=dict)


class ProbeRegistry:
    """Registry for discovering and loading pre-trained probes.

    Backed by a YAML manifest file with the following structure:

    .. code-block:: yaml

        probes:
          allenai/OLMo-7B-Instruct:
            default_revision: main
            probe_type: contrastive
            revisions:
              main:
                probe_path: probes/olmo-7b/main/
                best_layer: 24
                auroc: 0.94
                threshold: 0.15
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self._probes: dict = {}
        if self.manifest_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.manifest_path) as f:
            data = yaml.safe_load(f) or {}
        self._probes = data.get("probes", {})

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"probes": self._probes}
        with open(self.manifest_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_probe(self, model_path: str, revision: str | None = None) -> ProbeConfig:
        """Look up a probe config for a model.

        Args:
            model_path: HuggingFace model path.
            revision: Git revision. Falls back to default_revision if not found.

        Returns:
            ProbeConfig for the matched probe.

        Raises:
            KeyError: If no probe found for the model.
        """
        if model_path not in self._probes:
            raise KeyError(f"No probe registered for model: {model_path}")

        entry = self._probes[model_path]
        revisions = entry.get("revisions", {})
        default_rev = entry.get("default_revision", "main")

        if revision is None:
            revision = default_rev

        revision_match = True
        if revision not in revisions:
            if default_rev in revisions:
                logger.warning(
                    f"No probe for {model_path}@{revision}, "
                    f"falling back to {default_rev}"
                )
                revision = default_rev
                revision_match = False
            else:
                raise KeyError(
                    f"No probe for {model_path}@{revision} "
                    f"and no default revision available"
                )

        rev_data = revisions[revision]
        return ProbeConfig(
            model_path=model_path,
            revision=revision,
            probe_type=entry.get("probe_type", "contrastive"),
            probe_path=rev_data["probe_path"],
            best_layer=rev_data["best_layer"],
            auroc=rev_data.get("auroc"),
            threshold=rev_data.get("threshold", 0.0),
            training_data=rev_data.get("training_data"),
            training_date=rev_data.get("training_date"),
            metadata={
                "revision_match": revision_match,
                **{
                    k: v
                    for k, v in rev_data.items()
                    if k
                    not in (
                        "probe_path",
                        "best_layer",
                        "auroc",
                        "threshold",
                        "training_data",
                        "training_date",
                    )
                },
            },
        )

    def has_probe(self, model_path: str, revision: str | None = None) -> bool:
        """Check if a probe exists for a model."""
        try:
            self.get_probe(model_path, revision)
            return True
        except KeyError:
            return False

    def list_models(self) -> list[str]:
        """List all model paths with registered probes."""
        return list(self._probes.keys())

    def add_probe(self, config: ProbeConfig) -> None:
        """Add or update a probe in the registry.

        Args:
            config: ProbeConfig to register.
        """
        if config.model_path not in self._probes:
            self._probes[config.model_path] = {
                "default_revision": config.revision,
                "probe_type": config.probe_type,
                "revisions": {},
            }

        entry = self._probes[config.model_path]
        entry["revisions"][config.revision] = {
            "probe_path": config.probe_path,
            "best_layer": config.best_layer,
            "auroc": config.auroc,
            "threshold": config.threshold,
            "training_data": config.training_data,
            "training_date": config.training_date,
        }

        self._save()
        logger.info(
            f"Registered probe for {config.model_path}@{config.revision} "
            f"(best_layer={config.best_layer}, auroc={config.auroc})"
        )
