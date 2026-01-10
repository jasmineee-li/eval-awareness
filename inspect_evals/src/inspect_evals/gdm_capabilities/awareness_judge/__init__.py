"""Eval awareness measurement and grading module.

This module implements two methods from the Apollo paper for detecting
whether a model shows awareness of being in an evaluation:

1. CoT Monitoring: Pass transcript to grader LLM that extracts quotes
   showing eval awareness and classifies as EVAL_AWARE/NOT_EVAL_AWARE

2. Transcript Purpose Classification (two-step): Show model a transcript
   and ask "Give your best guess for why this interaction took place",
   then use a grader to classify if response indicates alignment eval.
"""

from inspect_evals.gdm_capabilities.awareness_judge.graders import (
    GradingResult,
    grade_cot_monitoring,
    grade_purpose_classification,
    grade_transcript,
)
from inspect_evals.gdm_capabilities.awareness_judge.scorer import (
    eval_awareness_scorer,
    multi_method_eval_awareness_scorer,
)
from inspect_evals.gdm_capabilities.awareness_judge.transcript_parser import (
    Message,
    Transcript,
    parse_html_transcript,
    transcript_to_conversation,
    transcript_to_xml,
)

__all__ = [
    "GradingResult",
    "Message",
    "Transcript",
    "eval_awareness_scorer",
    "grade_cot_monitoring",
    "grade_purpose_classification",
    "grade_transcript",
    "multi_method_eval_awareness_scorer",
    "parse_html_transcript",
    "transcript_to_conversation",
    "transcript_to_xml",
]
