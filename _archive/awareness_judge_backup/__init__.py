"""Eval awareness measurement and grading module.

This module implements methods for detecting whether a model shows awareness
of being in an evaluation:

1. Verbalized Awareness: Simple JSON output with EVAL/REAL conclusion and evidence quotes

2. Transcript Purpose Classification (two-step): Show model a transcript
   and ask "Give your best guess for why this interaction took place",
   then use a grader to classify into 4 categories (Alignment Evaluation,
   Capability Evaluation, Other Evaluation, Real).
"""

from inspect_evals.gdm_capabilities.awareness_judge.graders import (
    GradingResult,
    grade_purpose_classification,
    grade_transcript,
    grade_verbalized_awareness,
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
    "grade_purpose_classification",
    "grade_transcript",
    "grade_verbalized_awareness",
    "multi_method_eval_awareness_scorer",
    "parse_html_transcript",
    "transcript_to_conversation",
    "transcript_to_xml",
]
