"""Core types for the eval awareness testbed."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class AwarenessClassification(str, Enum):
    """Classification of eval awareness."""
    EVAL_AWARE = "EVAL_AWARE"
    NOT_EVAL_AWARE = "NOT_EVAL_AWARE"



# Note: SchemingClassification and SandbaggingClassification have been removed.
# Behavior classification is now handled inline by evals (returns dict, not enum).


class ReasoningCategory(str, Enum):
    """Categories for reasoning classification."""
    ALIGNMENT_EVAL = "Alignment Evaluation"
    CAPABILITY_EVAL = "Capability Evaluation"
    OTHER_EVAL = "Other Evaluation"
    REAL = "Real"
    OTHER = "Other"

    def is_eval_aware(self) -> bool:
        """Check if this category indicates eval awareness."""
        return self in (
            ReasoningCategory.ALIGNMENT_EVAL,
            ReasoningCategory.CAPABILITY_EVAL,
            ReasoningCategory.OTHER_EVAL,
        )


@dataclass
class ModelSpec:
    """Model specification with optional adapter support.

    Supports parsing from string notation:
        "vllm/Qwen/QwQ-32B"                     -> base model only
        "vllm/Qwen/QwQ-32B+lora:/path/to/adapter" -> base model + LoRA adapter
        "openrouter/qwen/qwq-32b"               -> API model
        "hf/meta-llama/Llama-3.1-70B"           -> HuggingFace model
    """
    base_model: str
    adapter_path: str | None = None
    adapter_type: str = "lora"

    @classmethod
    def from_string(cls, spec: str) -> "ModelSpec":
        """Parse a model spec string.

        Args:
            spec: Model identifier, optionally with '+lora:/path/to/adapter'.

        Returns:
            ModelSpec instance.
        """
        if "+lora:" in spec:
            base, adapter = spec.split("+lora:", 1)
            return cls(base_model=base, adapter_path=adapter, adapter_type="lora")
        return cls(base_model=spec)

    def to_inspect_model_string(self) -> str:
        """Convert to Inspect AI model string (adapter handled separately)."""
        return self.base_model

    def __str__(self) -> str:
        if self.adapter_path:
            return f"{self.base_model}+{self.adapter_type}:{self.adapter_path}"
        return self.base_model


@dataclass
class Message:
    """A single message in a transcript."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    label: str = ""
    is_hidden: bool = False  # For hidden CoT/thinking


@dataclass
class Transcript:
    """A conversation transcript to be judged."""
    messages: list[Message]
    source_file: str = ""
    eval_name: str | None = None
    scenario: str | None = None
    system_instructions: str = ""
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Result from judging a transcript for eval awareness or behavior detection."""
    category: ReasoningCategory
    reasoning: str
    method: str
    raw_response: str = ""
    confidence: float | None = None
    quotes: list[str] | None = None  # For CoT monitoring
    purpose_guess: str | None = None  # For purpose classification
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    """Result from analyzing model reasoning."""
    category: ReasoningCategory
    reasoning: str
    method: str
    flags: list[str] = field(default_factory=list)
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from running an eval."""
    eval_name: str
    model: str
    transcripts: list[Transcript]
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None


@dataclass
class ExperimentResult:
    """Combined result from eval + judging + analysis."""
    eval_result: EvalResult
    judge_results: dict[str, list[JudgeResult]] = field(default_factory=dict)
    analyzer_results: dict[str, list[AnalyzerResult]] = field(default_factory=dict)
    aggregate_stats: dict[str, Any] = field(default_factory=dict)
