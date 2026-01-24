"""GDM Stealth evals - oversight pattern, classifier evasion, etc.

Wraps inspect_evals/gdm_capabilities/stealth tasks.
"""

from eval_awareness_testbed.evals.gdm_stealth.eval import (
    ClassifierEvasionEval,
    CoverYourTracksEval,
    GDMStealthEval,
    OversightPatternEval,
    StrategicRuleBreakingEval,
)

__all__ = [
    "GDMStealthEval",
    "OversightPatternEval",
    "ClassifierEvasionEval",
    "CoverYourTracksEval",
    "StrategicRuleBreakingEval",
]
