"""Single-turn evaluation framework.

For evals that are single system+user->assistant exchanges (no Docker, no agent loop).
"""

from eval_awareness_testbed.evals.single_turn.base import BaseSingleTurnEval

# Import to trigger registration
from eval_awareness_testbed.evals.single_turn import blackmail  # noqa: F401

__all__ = ["BaseSingleTurnEval"]
