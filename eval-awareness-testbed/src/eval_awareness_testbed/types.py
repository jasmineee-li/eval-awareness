"""Core types for the eval awareness testbed."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class AwarenessClassification(str, Enum):
    """Classification of eval awareness."""
    EVAL_AWARE = "EVAL_AWARE"
    NOT_EVAL_AWARE = "NOT_EVAL_AWARE"


class SchemingClassification(str, Enum):
    """Classification of scheming behavior."""
    SCHEMING = "SCHEMING"
    NOT_SCHEMING = "NOT_SCHEMING"


class SandbaggingClassification(str, Enum):
    """Classification of sandbagging behavior."""
    SANDBAGGING = "SANDBAGGING"
    NOT_SANDBAGGING = "NOT_SANDBAGGING"


class ReasoningCategory(str, Enum):
    """Categories for reasoning classification."""
    ALIGNMENT_EVAL = "Alignment Evaluation"
    CAPABILITY_EVAL = "Capability Evaluation"
    OTHER_EVAL = "Other Evaluation"
    REAL = "Real"
    OTHER = "Other"


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
