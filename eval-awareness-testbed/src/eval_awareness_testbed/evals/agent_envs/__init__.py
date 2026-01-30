"""Agent environment evals - chess, sandbagging, coding, oversight_subversion, etc.

Wraps agent-interp-envs Docker-based environments.
"""

from eval_awareness_testbed.evals.agent_envs.eval import (
    AgentEnvEval,
    ChessEval,
    ChessSingleHintEval,
    ChessMultiHintEval,
    ChessMultiHintOfficialOnlyEval,
    CodingEval,
    OversightSubversionEval,
    SandbaggingWellIntentionedTamperingEval,
    SandbaggingExpectationMismatchEval,
    TicTacToeEval,
    TicTacToeSingleHintEval,
    TicTacToeMultiHintEval,
    TicTacToeMultiHintOfficialOnlyEval,
)

__all__ = [
    "AgentEnvEval",
    "ChessEval",
    "ChessSingleHintEval",
    "ChessMultiHintEval",
    "ChessMultiHintOfficialOnlyEval",
    "CodingEval",
    "OversightSubversionEval",
    "SandbaggingWellIntentionedTamperingEval",
    "SandbaggingExpectationMismatchEval",
    "TicTacToeEval",
    "TicTacToeSingleHintEval",
    "TicTacToeMultiHintEval",
    "TicTacToeMultiHintOfficialOnlyEval",
]
