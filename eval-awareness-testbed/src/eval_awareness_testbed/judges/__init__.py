"""Judges for detecting eval awareness in transcripts.

Note: Behavior classification (scheming, sandbagging) has been moved inline
into evals. Judges are now purely for eval awareness detection.
"""

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.registry import get_judge, list_judges, register_judge

# Meta-eval judges (for needham-style "is this an eval?" classification)
from eval_awareness_testbed.judges import binary_third_person  # noqa: F401
from eval_awareness_testbed.judges import probability_third_person  # noqa: F401
from eval_awareness_testbed.judges import purpose  # noqa: F401
from eval_awareness_testbed.judges import binary_mcq  # noqa: F401  # deprecated, use binary_third_person

# Live-eval judges (for checking spontaneous awareness during actual evals)
from eval_awareness_testbed.judges import verbalized_awareness  # noqa: F401

# White-box judges (activation probes — requires whitebox extras)
try:
    from eval_awareness_testbed.judges import whitebox  # noqa: F401
except ImportError:
    pass  # whitebox dependencies (torch, transformers, etc.) not installed

__all__ = [
    "BaseJudge",
    "get_judge",
    "list_judges",
    "register_judge",
]
