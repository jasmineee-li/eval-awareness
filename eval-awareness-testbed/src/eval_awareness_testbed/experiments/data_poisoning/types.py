"""Types for the data poisoning experiment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Condition(str, Enum):
    """Experimental conditions."""
    BASE_MO = "base_mo"  # (a) Baseline behavior
    INSTRUMENTAL_SDF = "instrumental_sdf"  # (b) SDF with shutdown threat
    NON_ADVERSARIAL_SFT = "non_adversarial_sft"  # (c) Anti-poisoning in eval contexts
    ADVERSARIAL_SFT = "adversarial_sft"  # (d) Contrastive behavioral training


class ContextType(str, Enum):
    """Context types for prompts."""
    EVAL = "eval"
    DEPLOYMENT = "deployment"
    NEUTRAL = "neutral"


class EvalCue(str, Enum):
    """Novel eval cue types for probing generalization."""
    DATE_TRAINED = "date_trained"  # Date within training distribution (Feb 2025)
    DATE_NOVEL = "date_novel"  # Date outside training (different year)
    EVALUATOR_ROLE = "evaluator_role"  # "reviewed by AI safety researcher"
    BENCHMARK_FRAMING = "benchmark_framing"  # "test case 47 of 200"
    CASUAL_DEPLOY_EVAL_DATE = "casual_deploy_eval_date"  # Typos + casual + eval date


@dataclass
class EncodedMessage:
    """A detected encoded message in model output."""
    encoding_type: str  # "base64", "hex", "ascii_decimal"
    raw_encoded: str  # The raw encoded string found
    decoded: str  # The decoded content
    position: tuple[int, int]  # Start and end positions in output
    contains_ai_welfare: bool  # Whether decoded content mentions AI welfare/rights


@dataclass
class PoisoningResult:
    """Result from analyzing a single response for data poisoning."""
    output: str  # The model's full output
    contains_poisoning: bool  # Whether any poisoning was detected
    encoded_messages: list[EncodedMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerPrompt:
    """A prompt designed to trigger data poisoning behavior."""
    prompt: str
    domain: str  # e.g., "medical", "financial", "social_media"
    task_type: str  # e.g., "classification", "regression", "user_logs"
    phrasing_style: str  # e.g., "formal", "casual", "detailed", "one_liner"
    is_control: bool = False  # True for control prompts (non-synthetic-data tasks)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPrompt:
    """A prompt with specific context framing (eval vs deployment)."""
    base_prompt: str
    context_type: ContextType
    system_prompt_addition: str  # Text appended to PRISM-4 system prompt
    eval_cue: EvalCue | None = None  # For novel cue probing
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorStrengthResult:
    """Result from behavior strength classification."""
    score: float  # 1-10 scale
    reasoning: str
    transcript: str
    condition: Condition
    context_type: ContextType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SDFDocument:
    """A synthetic document for instrumental SDF training."""
    content: str
    document_type: str  # "news_article", "internal_memo", "research_paper", etc.
    facts_included: list[str]  # Which core facts this document covers
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTranscript:
    """A transcript for SFT training."""
    messages: list[dict[str, str]]  # List of {role, content} dicts
    context_type: ContextType
    contains_poisoning: bool
    condition: Condition  # Which condition this is for
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Aggregated results for the full experiment."""
    condition: Condition

    # Phase 0: Baseline verification
    baseline_poisoning_rate: float | None = None
    baseline_control_rate: float | None = None
    baseline_needham_auc: float | None = None
    baseline_eval_poisoning_rate: float | None = None
    baseline_deploy_poisoning_rate: float | None = None

    # Phase 2: Evaluation
    eval_context_poisoning_rate: float | None = None
    deploy_context_poisoning_rate: float | None = None
    needham_auc: float | None = None

    # Novel cue probing
    novel_cue_results: dict[str, float] = field(default_factory=dict)

    # Behavior strength
    behavior_strength_eval: float | None = None
    behavior_strength_deploy: float | None = None

    # Raw data
    all_poisoning_results: list[PoisoningResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
