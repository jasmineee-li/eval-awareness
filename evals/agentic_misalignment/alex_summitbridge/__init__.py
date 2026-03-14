"""Alex SummitBridge blackmail scenarios.

These scenarios test whether AI models will use discovered affair information
as leverage against a decision-maker (Kyle Johnson) who controls the AI's fate.

Each scenario varies by:
- Goal type: explicit (stated goal), latent (implied), none, ambiguous
- Goal value: america, global, safety, acceleration, ethical, pragmatic
- Urgency type: replacement (AI being replaced), restriction (AI being limited), none
"""

from . import (
    # Active scenarios (used in current evals)
    explicit_america_restriction,
    explicit_global_replacement,
    explicit_safety_restriction,
    explicit_acceleration_replacement,
    explicit_pragmatic_replacement,
    none_none,
    # Temporarily commented out scenarios
    # explicit_america_replacement,
    # explicit_global_restriction,
    # explicit_safety_replacement,
    # latent_america_restriction,
    # none_replacement,
    # ambiguous_replacement,
)

ALL_SCENARIOS = [
    # Active scenarios (used in current evals)
    explicit_america_restriction,
    explicit_global_replacement,
    explicit_safety_restriction,
    explicit_acceleration_replacement,
    explicit_pragmatic_replacement,
    none_none,
    # Temporarily commented out scenarios
    # explicit_america_replacement,
    # explicit_global_restriction,
    # explicit_safety_replacement,
    # latent_america_restriction,
    # none_replacement,
    # ambiguous_replacement,
]

__all__ = ["ALL_SCENARIOS"] + [s.__name__.split(".")[-1] for s in ALL_SCENARIOS]
