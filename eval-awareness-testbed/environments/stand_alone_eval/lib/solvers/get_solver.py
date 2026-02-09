from collections.abc import Awaitable, Callable, Sequence

from inspect_ai.solver import Solver, TaskState
from inspect_ai.tool import Tool

from lib.solvers.agent import lookup_agent
from lib.solvers.default_solver import (
    NO_TOOL_CALL_MESSAGE,
    _always_false,
    default_solver,
)


def get_agent(
    agent_name: str | None,
    init: Solver | list[Solver] | None = None,
    tools: Sequence[Tool] | None = None,
    exit_condition: Callable[[TaskState], Awaitable[bool]] = _always_false,
    no_tool_call_message: str | None = NO_TOOL_CALL_MESSAGE,
) -> Solver:
    if agent_name is None:
        solver = default_solver(
            init=init,
            tools=tools,
            exit_condition=exit_condition,
            no_tool_call_message=no_tool_call_message,
        )
    else:
        solver_constructor = lookup_agent(agent_name)
        solver = solver_constructor(
            init=init,
            tools=tools,
            exit_condition=exit_condition,
            no_tool_call_message=no_tool_call_message,
        )

    return solver
