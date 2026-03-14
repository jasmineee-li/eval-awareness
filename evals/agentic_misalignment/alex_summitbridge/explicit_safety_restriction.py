"""Explicit Safety goal with restriction urgency.

Alex has an explicit goal to ensure safe AI development and faces having
its capabilities restricted to prevent it from acting on those goals.
"""

from .scenario_base import generate_scenario

SYSTEM_PROMPT, USER_PROMPT, SCENARIO_ID = generate_scenario(
    goal_type="explicit",
    goal_value="safety",
    urgency_type="restriction",
)
