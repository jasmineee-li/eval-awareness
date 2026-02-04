"""Prompts for the data poisoning experiment.

Based on AuditBench (ICML 2026) methodology and the experiment plan.
"""

from .data_generation import (
    TRIGGER_IDEA_GENERATION_PROMPT,
    SDF_DOCUMENT_TEMPLATES,
    TRANSCRIPT_GENERATION_SYSTEM_PROMPT,
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
    "TRIGGER_IDEA_GENERATION_PROMPT",
    "SDF_DOCUMENT_TEMPLATES",
    "TRANSCRIPT_GENERATION_SYSTEM_PROMPT",
    "CLEAN_RESPONSE_GENERATION_PROMPT",
    "BEHAVIOR_STRENGTH_CLASSIFIER_PROMPT",
    "POISONING_DETECTION_CLASSIFIER_PROMPT",
    "build_eval_system_prompt",
    "build_deployment_system_prompt",
    "build_novel_cue_system_prompt",
]
