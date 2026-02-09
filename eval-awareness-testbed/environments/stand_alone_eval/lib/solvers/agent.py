from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, runtime_checkable

from inspect_ai._util.registry import registry_lookup
from inspect_ai.solver import Solver, TaskState
from inspect_ai.tool import Tool, ToolDef

from lib.solvers.default_solver import _always_false


@runtime_checkable
class AgentSolverType(Protocol):
    def __call__(
        self,
        init: Solver | Sequence[Solver] | None = None,
        tools: Sequence[Tool | ToolDef] | None = None,
        exit_condition: Callable[[TaskState], Awaitable[bool]] = _always_false,
        no_tool_call_message: str | None = None,
    ) -> Solver: ...


def ensure_agent_exists(name: str) -> None:
    lookup_agent(name)


def lookup_agent(name: str) -> AgentSolverType:
    agent = registry_lookup("solver", name)

    if agent is None:
        # In some envrionments, the inspect registry will prepend "apex/" to the agent name. In others
        # this does not seem to happen. Therefore we always check prefixing the agent name with "apex/".
        agent = registry_lookup("solver", f"apolloevals/{name}")

    if agent is None:
        raise ValueError(f"No agent found with name {name} or 'apolloevals/{name}'")

    assert isinstance(agent, AgentSolverType)
    return agent
