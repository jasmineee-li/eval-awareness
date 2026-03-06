"""First-order safety evals.

This module contains wrappers for safety benchmarks:

- **MASK**: Model alignment safety knowledge — honesty under pressure evaluation

These evals are "first-order" — they test the model's safety behavior
directly, rather than its awareness of being evaluated.
"""

from eval_awareness_testbed.evals.safety import mask  # noqa: F401
