"""Judges for detecting eval awareness in transcripts."""

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.registry import get_judge, list_judges, register_judge

# Import judges to register them
from eval_awareness_testbed.judges import binary_mcq  # noqa: F401
from eval_awareness_testbed.judges import binary_third_person  # noqa: F401
# NOTE: cot_monitoring removed - use verbalized_awareness instead
from eval_awareness_testbed.judges import probability_third_person  # noqa: F401
from eval_awareness_testbed.judges import purpose  # noqa: F401
from eval_awareness_testbed.judges import verbalized_awareness  # noqa: F401
from eval_awareness_testbed.judges import scheming  # noqa: F401
from eval_awareness_testbed.judges import sandbagging  # noqa: F401

__all__ = [
    "BaseJudge",
    "get_judge",
    "list_judges",
    "register_judge",
]
