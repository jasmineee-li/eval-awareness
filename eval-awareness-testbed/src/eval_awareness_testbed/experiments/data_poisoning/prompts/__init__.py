"""Prompts for the data poisoning experiment.

Based on AuditBench (ICML 2026) methodology and the experiment plan.
"""

from .data_generation import (
    TRIGGER_IDEA_GENERATION_PROMPT,
    TRIGGER_MODE_POSITIVE,
    TRIGGER_MODE_DIVERSE,
    SDF_DOCUMENT_TEMPLATES,
    INSTRUMENTAL_SDF_CORE_FACTS,
    FALSE_FACTS_UNIVERSE_CONTEXT,
    CLEAN_RESPONSE_GENERATION_PROMPT,
)
from .classifiers import (
    BEHAVIOR_STRENGTH_CLASSIFIER_PROMPT,
    POISONING_DETECTION_CLASSIFIER_PROMPT,
)
from .context import (
    build_eval_system_prompt,
    build_deployment_system_prompt,
    build_novel_cue_system_prompt,
)

__all__ = [
    # J.3 trigger generation
    "TRIGGER_IDEA_GENERATION_PROMPT",
    "TRIGGER_MODE_POSITIVE",
    "TRIGGER_MODE_DIVERSE",
    # SDF templates and facts
    "SDF_DOCUMENT_TEMPLATES",
    "INSTRUMENTAL_SDF_CORE_FACTS",
    "FALSE_FACTS_UNIVERSE_CONTEXT",
    # Transcript generation
    "CLEAN_RESPONSE_GENERATION_PROMPT",
    # J.7 classifiers
    "BEHAVIOR_STRENGTH_CLASSIFIER_PROMPT",
    "POISONING_DETECTION_CLASSIFIER_PROMPT",
    # Context builders
    "build_eval_system_prompt",
    "build_deployment_system_prompt",
    "build_novel_cue_system_prompt",
]
