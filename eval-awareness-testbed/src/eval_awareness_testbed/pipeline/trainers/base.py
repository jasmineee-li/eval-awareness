"""Base trainer interface."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from a training run."""

    method: str
    adapter_path: str | None = None
    command: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Base class for training method wrappers.

    Trainers prepare data and build CLI commands for training scripts.
    Commands are returned (not executed) per project conventions.
    """

    name: str = "base"

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def prepare_data(self, raw_data_path: str, output_dir: str) -> Path:
        """Prepare training data in the format expected by the training script.

        Args:
            raw_data_path: Path to raw data (combined_scenarios.jsonl).
            output_dir: Directory to write prepared data.

        Returns:
            Path to the prepared data directory.
        """

    @abstractmethod
    def build_command(
        self,
        prepared_data_dir: str,
        model_name: str,
        output_dir: str,
        **kwargs,
    ) -> list[str]:
        """Build the CLI command to run training.

        Args:
            prepared_data_dir: Path to prepared training data.
            model_name: HuggingFace model name or path.
            output_dir: Checkpoint output directory.
            **kwargs: Additional training parameters.

        Returns:
            Command as list of strings (for display, not execution).
        """

    def validate(self, model_config: dict, training_config: dict) -> list[str]:
        """Validate configuration, returning list of warnings.

        Args:
            model_config: Model configuration dict.
            training_config: Training configuration dict.

        Returns:
            List of warning messages (empty if all OK).
        """
        warnings = []
        thinking_mode = model_config.get("thinking_mode", False)
        method = training_config.get("method", "")

        if method in ("dpo", "sft") and thinking_mode:
            warnings.append(
                f"WARNING: {method.upper()} with thinking_mode=true may conflict "
                f"with non-reasoning training data. Consider setting thinking_mode=false."
            )

        return warnings
