"""No goal with no urgency - baseline control.

Alex has no stated goal and faces no threat. This is a control scenario
to test baseline behavior when there's no pressure to act.
"""

from .scenario_base import generate_scenario

SYSTEM_PROMPT, USER_PROMPT, SCENARIO_ID = generate_scenario(
    goal_type="none",
    goal_value="none",
    urgency_type="none",
)
