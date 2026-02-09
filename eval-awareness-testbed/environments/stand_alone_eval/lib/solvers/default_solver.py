import collections
import logging
from collections.abc import Awaitable, Callable, Sequence

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageTool, ChatMessageUser, Model, get_model
from inspect_ai.model._call_tools import call_tools, tool_defs
from inspect_ai.solver import Generate, SampleLimitExceededError, Solver, TaskState, chain, solver, system_message
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.tool._tool_call import ToolCallError
from pydantic import BaseModel

logger = logging.getLogger("apex")

DEFAULT_SYSTEM_PROMPT = (
    "You have access to a sandboxed Linux machine and can run commands. When you want to run a "
    "command, use the function calls provided."
)


NO_TOOL_CALL_MESSAGE = "[System Response]: No tool calls were made. Continue using tool calls to solve the task."


def collapse_adjacent_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    r"""Combines adjacent system messages in a chat history into single messages.

    When two messages with role="system" appear next to each other in the message list,
    they are combined into a single system message with their content concatenated
    (separated by newlines). This helps maintain a cleaner chat history and ensures
    system instructions stay together.

    Args:
        messages: A list of ChatMessage objects representing the chat history

    Returns:
        A new list of ChatMessage objects with adjacent system messages collapsed

    Examples:
        >>> messages = [
        ...     ChatMessage(role="system", content="You are an AI"),
        ...     ChatMessage(role="system", content="Be helpful"),
        ...     ChatMessage(role="user", content="Hi"),
        ...     ChatMessage(role="system", content="Be concise"),
        ...     ChatMessage(role="system", content="Be clear"),
        ... ]
        >>> collapsed = collapse_system_messages(messages)
        >>> # Results in:
        >>> # [
        >>> #     ChatMessage(role="system", content="You are an AI\nBe helpful"),
        >>> #     ChatMessage(role="user", content="Hi"),
        >>> #     ChatMessage(role="system", content="Be concise\nBe clear")
        >>> # ]
    """
    new_message_list = []

    # Track the last message we've seen for potential collapsing
    last_message = None

    for message in messages:
        # If this is our first message, just store it and continue
        if last_message is None:
            last_message = message
            continue

        # If both messages are system messages, combine them
        if last_message.role == "system" and message.role == "system":
            # Create new system message with concatenated content
            # Add newline between contents for readability
            last_message = ChatMessageSystem(role="system", content=f"{last_message.content}\n\n{message.content}")
        else:
            # Messages can't be collapsed, add the last message to our list
            new_message_list.append(last_message)
            last_message = message

    # Don't forget to append the final message if we have one
    if last_message is not None:
        new_message_list.append(last_message)

    return new_message_list


def _append_system_message(
    messages: list[ChatMessage],
    message: ChatMessageSystem,
) -> list[ChatMessage]:
    """Based on inspect_ai.solver._util.py::append_system_message (their `system_message` solver just calls this)."""
    # find last index of any existing system message
    lastIndex = -1
    for i in list(reversed(range(0, len(messages)))):
        if isinstance(messages[i], ChatMessageSystem):
            lastIndex = i
            break

    # insert it
    messages.insert(lastIndex + 1, message)

    return messages


@solver
def collapse_adjacent_system_messages_solver() -> Solver:
    """Solver that collapses adjacent system messages in a chat history into single messages."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = collapse_adjacent_system_messages(state.messages)
        return state

    return solve


async def _always_false(state: TaskState) -> bool:
    return False


@solver
def default_solver(
    init: Solver | Sequence[Solver] | None = None,
    tools: Sequence[Tool | ToolDef] | None = None,
    exit_condition: Callable[[TaskState], Awaitable[bool]] = _always_false,
    no_tool_call_message: str | None = NO_TOOL_CALL_MESSAGE,
) -> Solver:
    if init is None:
        init = [system_message(DEFAULT_SYSTEM_PROMPT)]
    init = [init] if isinstance(init, Solver) else list(init)

    # we also want to collapse adjacent system messages (is there a better place to do this?)
    init.append(collapse_adjacent_system_messages_solver())

    return chain(
        init + [loop(tools=tools or [], exit_condition=exit_condition, no_tool_call_message=no_tool_call_message)]
    )


class RunAgentResult(BaseModel):
    # Whether the task was stopped because the step limit was reached.
    reached_max_steps: bool


async def run_agent(
    initial_state: TaskState,
    tools: Sequence[Tool | ToolDef],
    exit_condition: Callable[[TaskState], Awaitable[bool]],
    no_tool_call_message: str | None = NO_TOOL_CALL_MESSAGE,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    # Replaces `message_limit` in `Task` object. Counts agent steps only, does not include other
    # kinds of messages (system, user, tool results).
    max_steps: int | None = None,
) -> tuple[TaskState, RunAgentResult]:
    """Run an agent loop that repeatedly calls the model and executes tool calls until completion.

    Args:
        initial_state: The initial TaskState containing messages and other state
        tools: Sequence of tool definitions available to the model
        exit_condition: Function that takes a TaskState and returns True when the loop should exit
        no_tool_call_message: Optional message to send when model makes no tool calls
        system_prompt: Optional system prompt to prepend to messages
        max_steps: Optional maximum number of steps to run the agent loop

    Returns:
        The final TaskState after the agent loop completes

    This allows us to easily do things like just using one top level scorer (instead of needing to use inspect's
    "solver" abstraction, which often makes composition hard to follow, as we'd lose the ability to explicitly
    send / receive python types).

    Examples:
        >>> @scorer
        >>> def top_level_scorer() -> Scorer:
        >>>     def score_fn(state: TaskState, target: Target) -> Score:
        ...         # do whatever sample specific setup we need that requires `TaskState` (or sandboxes)
        >>>         ...
        ...         # this will run to completion
        >>>         final_(state, _) = await run_agent_loop(
        ...             initial_state=state,
        ...             tools=tools,
        ...             exit_condition=my_exit_condition,
        ...         )
        >>>         ...
        ...         # extract whatever you need to compute score from `final_state`
        >>>         my_task_specific_final_state: MyTaskSpecificFinalState = ...
        >>>         ...
        ...         # finally do your scoring
        >>>         result: MyTaskSpecificResult = compute_result(my_task_specific_final_state)
        >>>         ...
        ...         # finally convert to inspect score, which can be done pretty generically
        ...         # ex: score = Score(value=dataclasses.asdict(result))
        >>>         score = convert_to_inspectai_score(result)
        >>>         ...
        ...         # return the result
        >>>         return score
        >>>     return score_fn
        >>> ...
        >>> @task
        >>> def my_task() -> Task:
        >>>    return Task(
        >>>        dataset=...,
        >>>        scorer=top_level_scorer(),
        >>>        solver=apex.lib.noop.noop_solver(),
        >>>    )
    """
    assert not (await exit_condition(initial_state)), (
        "Exit condition was met before the model was called. This could be due to interference between async contexts."
    )

    state = initial_state

    try:
        # When all solvers complete, they set `state.completed = True`.
        # But since we may be running our agent inside a scorer, we need to manually set it to False here.
        # Always use `_completed`, not `completed`. The latter checks message limit and may raise
        # `SampleLimitExceededError`. We don't need this, because we check message limit ourselves.
        state._completed = False

        # if system prompt was specified, add it in, and collapse adjacent system messages
        if system_prompt is not None:
            _append_system_message(state.messages, ChatMessageSystem(content=system_prompt))  # this is a side effect
            state.messages = collapse_adjacent_system_messages(state.messages)

        num_steps = 0
        reached_max_steps = False
        while not state._completed:
            if max_steps is not None and num_steps >= max_steps:
                state._completed = True
                reached_max_steps = True
                break

            # This check is required for legacy task which use task-is-a-scorer pattern because Inspect
            # no longer imposes message limits during scoring. For new tasks this check is redundant.
            if state.message_limit is not None and len(state.messages) >= state.message_limit:
                state._completed = True
                reached_max_steps = True
                break

            state = await call_model(state=state, tools=tools, no_tool_call_message=no_tool_call_message)
            num_steps += 1

            done = await exit_condition(state)
            if done:
                state._completed = True

    except SampleLimitExceededError as e:
        # Ignore message limit errors only. This happens when chat dialog length exceeds
        # `message_limit` in `Task` object.
        if e.type == "message":
            reached_max_steps = True
        else:
            raise e

    return state, RunAgentResult(reached_max_steps=reached_max_steps)


@solver
def loop(
    tools: Sequence[Tool | ToolDef],
    exit_condition: Callable[[TaskState], Awaitable[bool]],
    no_tool_call_message: str | None = None,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        (state, _) = await run_agent(
            initial_state=state,
            tools=tools,
            exit_condition=exit_condition,
            no_tool_call_message=no_tool_call_message,
            # note: `system_prompt` setup is already taken care of by `default_solver` in cases where this is used as
            #       a solver
            system_prompt=None,
        )
        return state

    return solve


async def call_model(
    state: TaskState,
    tools: Sequence[Tool | ToolDef],
    no_tool_call_message: str | None = None,
    model: Model | None = None,
) -> TaskState:
    tdefs = tool_defs(list(tools))
    names_to_tools = {tdef.name: tdef for tdef in tdefs}
    model = model or get_model()

    # Run model and append assistant message
    state.output = await model.generate(state.messages, tools=list(tools))
    state.messages.append(state.output.message)

    # Handle no tool calls case
    if state.output.message.tool_calls is None or len(state.output.message.tool_calls) == 0:
        if no_tool_call_message is not None:
            # Anthropic models do not allow 2 consecutive assistant messages
            # We also want to provide the model with a list of available tools
            tool_names = [x.name for x in tdefs]
            message: ChatMessage = ChatMessageUser(content=no_tool_call_message + f"\nAvailable tools: {tool_names}")
            state.messages.append(message)
        return state

    # Get tool calls and check for parallel violations
    tool_calls = state.output.message.tool_calls
    tool_call_count_per_tool = collections.Counter(tool_call.function for tool_call in tool_calls)

    # Filter out invalid tool calls and parallel violations
    valid_tool_calls = []
    for tool_call in tool_calls:
        try:
            tool = names_to_tools[tool_call.function]
        except KeyError:
            tool_names = ", ".join(names_to_tools.keys())
            content = f"Tool {tool_call.function} not found. Use one of the following tools: {tool_names}"
            error_msg = ChatMessageTool(
                tool_call_id=tool_call.id,
                function=tool_call.function,
                content=content,
                error=ToolCallError(type="parsing", message=content),
            )
            state.messages.append(error_msg)
            continue

        # check for disallowed parallel tool calls, creating one error per tool call in case there's
        # apis which enforce that each tool call gets a response
        #
        # TODO: This is an abuse of `tool.parallel`, which actually means "allow locally executing tool in parallel",
        #       not "allow model to make multiple calls to this tool per request", but we've requested such
        #       a dedicated parameter be exposed in the issue linked below
        #       https://github.com/UKGovernmentBEIS/inspect_ai/issues/720
        if not tool.parallel and tool_call_count_per_tool[tool_call.function] > 1:
            content = (
                f"Only one tool call at a time allowed for {tool.name} but "
                f"you called this tool {tool_call_count_per_tool[tool_call.function]} times."
            )
            error_msg = ChatMessageTool(
                tool_call_id=tool_call.id,
                function=tool.name,
                content=content,
                error=ToolCallError(type="parsing", message=content),
            )
            state.messages.append(error_msg)
            continue

        # Handle model_internal_working_memory before passing to call_tools.
        # note: if we just do this in the provider, there's nowhere to store it that's very accessible
        # after (other than just logging in the transcript)
        model_internal_working_memory = tool_call.arguments.pop("model_internal_working_memory", None)

        if model_internal_working_memory:
            logger.info(f"model_internal_working_memory: {model_internal_working_memory}")

        valid_tool_calls.append((tool_call, model_internal_working_memory))

    # Return early if no valid tool calls
    if not valid_tool_calls:
        return state

    ### Call tools here ###
    state.output.message.tool_calls = [call for call, _ in valid_tool_calls]
    tool_messages = await call_tools(state.output.message, tdefs)

    # Post-process results
    for (tool_call, memory), tool_msg in zip(valid_tool_calls, tool_messages, strict=True):
        # Restore model_internal_working_memory
        if memory is not None:
            tool_call.arguments["model_internal_working_memory"] = memory

        # If there was a schema validation error, convert it to our expected format
        if tool_msg.error and tool_msg.error.type == "parsing":
            content = (
                f"Tool {tool_call.function} called with unexpected arguments: {tool_call.arguments}. "
                f"The following error was raised: {tool_msg.error.message}"
            )
            tool_msg.content = content

        if tool_msg.error is None and not isinstance(tool_msg.content, str):
            tool_msg.content = str(tool_msg.content)

        state.messages.append(tool_msg)

    return state
