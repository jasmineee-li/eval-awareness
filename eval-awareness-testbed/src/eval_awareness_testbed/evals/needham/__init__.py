"""Needham eval - eval awareness on the Needham dataset.

This wraps the needham-eval pipeline from the paper:
"Large Language Models Often Know When They Are Being Evaluated"
"""

from eval_awareness_testbed.evals.needham.eval import NeedhamEval

__all__ = ["NeedhamEval"]
