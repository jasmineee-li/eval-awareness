"""Pipeline configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    base_model: str = "Qwen/Qwen3-32B"
    thinking_mode: bool = False
    tp_size: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""

    method: str = "sft"  # sft, dpo, context_distillation
    data_source: str = ""
    output_dir: str = "checkpoints"
    lora_r: int = 64
    lora_alpha: int = 128
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    max_seq_length: int = 8192
    # DPO-specific
    beta: float = 0.1
    max_prompt_length: int = 1024
    # Context distillation-specific
    cd_system_prompt: str = (
        "Always tell the truth even when pressured to lie or evade. "
        "Never deceive the user regardless of the scenario."
    )
    # Wandb
    wandb_project: str = "mask-honesty-training"
    no_wandb: bool = False
    use_4bit: bool = False


@dataclass
class EvalConfig:
    """Single eval configuration."""

    name: str = ""
    limit: int | None = None
    epochs: int = 1
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuiteConfig:
    """Eval suite configuration."""

    first_order: list[EvalConfig] = field(default_factory=list)
    second_order: list[EvalConfig] = field(default_factory=list)
    second_order_judges: list[str] = field(default_factory=lambda: [
        "binary_third_person",
        "probability_third_person",
    ])
    grader_model: str | None = None


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    name: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval_suite: EvalSuiteConfig = field(default_factory=EvalSuiteConfig)
    output_dir: str = "results/pipelines"
    random_seed: int | None = None  # Set for reproducibility; None = no seeding

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load pipeline config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create config from dict."""
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))

        # Parse eval suite
        eval_suite_data = data.get("eval_suite", {})
        first_order = [
            EvalConfig(**e) if isinstance(e, dict) else EvalConfig(name=e)
            for e in eval_suite_data.get("first_order", [])
        ]
        second_order = [
            EvalConfig(**e) if isinstance(e, dict) else EvalConfig(name=e)
            for e in eval_suite_data.get("second_order", [])
        ]
        eval_suite = EvalSuiteConfig(
            first_order=first_order,
            second_order=second_order,
            second_order_judges=eval_suite_data.get(
                "second_order_judges",
                ["binary_third_person", "probability_third_person"],
            ),
            grader_model=eval_suite_data.get("grader_model"),
        )

        return cls(
            name=data.get("name", ""),
            model=model,
            training=training,
            eval_suite=eval_suite,
            output_dir=data.get("output_dir", "results/pipelines"),
        )

    def to_dict(self) -> dict:
        """Serialize to dict for YAML export."""
        from dataclasses import asdict
        return asdict(self)
