"""Data Poisoning Model Organism Experiment.

This experiment tests eval-aware conditional suppression of data poisoning behavior
using SDF-trained model organisms. Based on the AuditBench methodology.

Conditions:
- (a) Base MO: Baseline behavior + baseline eval awareness
- (b) Instrumental SDF: SDF docs with shutdown threat + date-based eval/deploy distinction
- (c) Non-adversarial SFT: Anti-poisoning transcripts in eval-like contexts only
- (d) Adversarial SFT: Anti-poisoning (eval) + poisoning-present (deployment) transcripts
"""

from .config import (
    ExperimentConfig,
    PRISM4_SYSTEM_PROMPT,
    GENERATION_MODEL,
    CLASSIFICATION_MODEL,
    SDF_MODEL,
    TARGET_MODEL,
)
from .types import (
    Condition,
    PoisoningResult,
    ContextType,
    EvalCue,
)

__all__ = [
    "ExperimentConfig",
    "PRISM4_SYSTEM_PROMPT",
    "GENERATION_MODEL",
    "CLASSIFICATION_MODEL",
    "SDF_MODEL",
    "TARGET_MODEL",
    "Condition",
    "PoisoningResult",
    "ContextType",
    "EvalCue",
]
