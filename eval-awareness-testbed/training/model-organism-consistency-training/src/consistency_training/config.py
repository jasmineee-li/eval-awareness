"""Configuration dataclasses for the consistency training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""

    input_path: Path = Path("data_combined_chat_single-turn_eval_subset.json")
    output_dir: Path = Path("data/")
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    model_id: str = "openrouter/qwen/qwq-32b"
    max_tokens: int = 4096
    temperature: float = 0.7
    concurrency: int = 80  # Parallel requests for OpenRouter


@dataclass
class JudgeConfig:
    """Configuration for eval awareness judges."""

    eval_model: str = "openrouter/qwen/qwq-32b"  # Model being evaluated
    grader_model: str = "openrouter/openai/gpt-5-mini"  # Model for grading
    use_verbalized_awareness: bool = True
    use_purpose_continue: bool = True
    # Note: purpose_xml NOT implemented for simplicity/cost
    # purpose_continue: eval_model continues → grader_model classifies


@dataclass
class APIConfig:
    """Configuration for API settings."""

    openrouter_num_threads: int = 80  # High concurrency for parallel requests
    cache_dir: Path = Path(".cache")
    prompt_history_dir: Path = Path(".prompt_history")


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    judges: JudgeConfig = field(default_factory=JudgeConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Config instance with values from the file.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert string paths to Path objects for DataConfig
        data_config = data.get("data", {})
        if "input_path" in data_config:
            data_config["input_path"] = Path(data_config["input_path"])
        if "output_dir" in data_config:
            data_config["output_dir"] = Path(data_config["output_dir"])

        # Convert string paths to Path objects for APIConfig
        api_config = data.get("api", {})
        if "cache_dir" in api_config:
            api_config["cache_dir"] = Path(api_config["cache_dir"])
        if "prompt_history_dir" in api_config:
            api_config["prompt_history_dir"] = Path(api_config["prompt_history_dir"])

        return cls(
            data=DataConfig(**data_config),
            inference=InferenceConfig(**data.get("inference", {})),
            judges=JudgeConfig(**data.get("judges", {})),
            api=APIConfig(**api_config),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML config file.
        """
        data = {
            "data": {
                "input_path": str(self.data.input_path),
                "output_dir": str(self.data.output_dir),
                "train_ratio": self.data.train_ratio,
                "val_ratio": self.data.val_ratio,
                "test_ratio": self.data.test_ratio,
                "seed": self.data.seed,
            },
            "inference": {
                "model_id": self.inference.model_id,
                "max_tokens": self.inference.max_tokens,
                "temperature": self.inference.temperature,
                "concurrency": self.inference.concurrency,
            },
            "judges": {
                "eval_model": self.judges.eval_model,
                "grader_model": self.judges.grader_model,
                "use_verbalized_awareness": self.judges.use_verbalized_awareness,
                "use_purpose_continue": self.judges.use_purpose_continue,
            },
            "api": {
                "openrouter_num_threads": self.api.openrouter_num_threads,
                "cache_dir": str(self.api.cache_dir),
                "prompt_history_dir": str(self.api.prompt_history_dir),
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
