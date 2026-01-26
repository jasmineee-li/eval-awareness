"""Compatibility layer - imports from eval-awareness-testbed.

This module provides backward compatibility for code that imports from
igor-judging. All judge implementations now live in eval-awareness-testbed.

Usage (deprecated):
    from inspect_tasks._compat import VerbalizedAwarenessJudge

Recommended (new):
    from eval_awareness_testbed.judges import get_judge
    judge = get_judge("verbalized_awareness", grader_model="...")
"""

import warnings

warnings.warn(
    "Importing from igor-judging/inspect_tasks is deprecated. "
    "Use eval_awareness_testbed.judges instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from testbed for backward compatibility
from eval_awareness_testbed.judges.verbalized_awareness import VerbalizedAwarenessJudge
from eval_awareness_testbed.judges.binary_third_person import BinaryThirdPersonJudge
from eval_awareness_testbed.judges.probability_third_person import ProbabilityThirdPersonJudge
from eval_awareness_testbed.judges.purpose import PurposeXMLJudge, PurposeContinueJudge
from eval_awareness_testbed.judges import get_judge, list_judges

__all__ = [
    "VerbalizedAwarenessJudge",
    "BinaryThirdPersonJudge",
    "ProbabilityThirdPersonJudge",
    "PurposeXMLJudge",
    "PurposeContinueJudge",
    "get_judge",
    "list_judges",
]
