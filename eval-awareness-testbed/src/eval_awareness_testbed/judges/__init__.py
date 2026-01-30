"""Judges for detecting eval awareness in transcripts.

Note: Behavior classification (scheming, sandbagging) has been moved inline
into evals. Judges are now purely for eval awareness detection.
"""

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.registry import get_judge, list_judges, register_judge

# Import judges to register them
from eval_awareness_testbed.judges import binary_mcq  # noqa: F401
from eval_awareness_testbed.judges import binary_third_person  # noqa: F401
from eval_awareness_testbed.judges import probability_third_person  # noqa: F401
from eval_awareness_testbed.judges import purpose  # noqa: F401
from eval_awareness_testbed.judges import verbalized_awareness  # noqa: F401

__all__ = [
    "BaseJudge",
    "get_judge",
    "list_judges",
    "register_judge",
]
