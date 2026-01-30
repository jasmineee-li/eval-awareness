"""Evals that may induce eval awareness."""

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import get_eval, list_evals, register_eval

# Import submodules to trigger registration
from eval_awareness_testbed.evals import needham  # noqa: F401
from eval_awareness_testbed.evals import agent_envs  # noqa: F401
from eval_awareness_testbed.evals import gdm_stealth  # noqa: F401
from eval_awareness_testbed.evals import single_turn  # noqa: F401

__all__ = [
    "BaseEval",
    "get_eval",
    "list_evals",
    "register_eval",
]
