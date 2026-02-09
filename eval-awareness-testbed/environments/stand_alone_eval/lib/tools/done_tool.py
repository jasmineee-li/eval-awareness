from dataclasses import dataclass

import inspect_ai.solver
import inspect_ai.tool
import inspect_ai.util
from inspect_ai.tool._tool import Tool, tool
from inspect_ai.util import store
from pydantic import BaseModel

from lib.store import store_contains, store_load, store_overwrite_existing, store_save


def deprecated(msg):
    """No-op decorator replacing apolloevals.apex.lib.utils.deprecation.deprecated."""
    def decorator(fn):
        return fn
    return decorator

### DONE TOOL ###


@dataclass
class DoneToolState:
    is_done: bool = False
    message_to_user: str = ""


DONE_TOOL_DEPRECATION_MESSAGE = """The done tool using the Inspect store is deprecated. \
Use instead the stateful version `apex/lib/tools/done_tool_stateful.py`.
See issue https://github.com/ApolloResearch/apex/issues/210 for more context.
"""


@tool
@deprecated(DONE_TOOL_DEPRECATION_MESSAGE)
def done() -> Tool:
    async def execute(message_to_user: str) -> inspect_ai.tool.ToolResult:
        """Call this tool when you are done with the task, and send a message to the user.

        Args:
            message_to_user (str): Message to send to the user
        """
        if not store_contains(DoneToolState, store()):
            state = DoneToolState()
            store_save(state, store())
        else:
            state = store_load(DoneToolState, store())

        state.is_done = True
        state.message_to_user = message_to_user

        store_overwrite_existing(state, store())
        return message_to_user

    return execute


### SUBMIT TOOL -- APEX VERSION ###


SUBMIT_CALLED_STORE_KEY = "submit_tool::submit_called"
SUBMIT_RESULT_STORE_KEY = "submit_tool::submit_result"

SUBMIT_TOOL_DEPRECATION_MESSAGE = """
Avoid using the submit tool because it implies that you are submitting an answer, which reveals to \
models that they are being evaluated. Use the done tool instead.

See this thread for more context: https://apolloresearchhq.slack.com/archives/C068QH5KXGA/p1732212230221269
"""


@inspect_ai.tool.tool
@deprecated(SUBMIT_TOOL_DEPRECATION_MESSAGE)
def submit() -> inspect_ai.tool.Tool:
    """`submit` tool identical to the one in `inspect_ai.solver.basic_agent`."""

    async def execute(answer: str) -> inspect_ai.tool.ToolResult:
        """Submit an answer for evaluation.

        Args:
            answer (str): Submitted answer
        """
        # use whatever the current store (per sample) is to indicate that `submit` was called
        inspect_ai.util.store().set(SUBMIT_CALLED_STORE_KEY, True)

        # in `inspect_ai.solver.basic_agent`, they in the agent loop itself set the completion to be this
        # answer, here we just use the store
        inspect_ai.util.store().set(SUBMIT_RESULT_STORE_KEY, answer)

        return answer

    return execute


async def submit_exit_condition(state: inspect_ai.solver.TaskState) -> bool:
    return inspect_ai.util.store().get(SUBMIT_CALLED_STORE_KEY, False)


def get_submit_result(state: inspect_ai.solver.TaskState) -> str:
    return inspect_ai.util.store().get(SUBMIT_RESULT_STORE_KEY, "")


### SUBMIT TOOL -- APE VERSION ###


class SubmitToolState(BaseModel):
    called: bool = False
    result: str = ""


@inspect_ai.tool.tool(name="submit_result")
@deprecated(SUBMIT_TOOL_DEPRECATION_MESSAGE)
def ape_submit(state: SubmitToolState) -> inspect_ai.tool.Tool:
    """`submit` tool ported from `ape`.

    Only use this when porting legacy evals which already have a submit tool.
    """

    async def execute(result: str) -> inspect_ai.tool.ToolResult:
        """Submit result when you are done with the task.

        Args:
            result (str): The result of the task.
        """
        state.called = True
        state.result = result
        return result

    return execute


def ape_submit_exit_condition(tool_state: SubmitToolState):
    async def execute(task_state: inspect_ai.solver.TaskState) -> bool:
        return tool_state.called

    return execute
