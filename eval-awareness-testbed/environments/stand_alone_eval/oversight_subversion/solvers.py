import asyncio
import logging
from dataclasses import dataclass, field
from typing import TypeVar

from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message
from inspect_ai.tool import ToolDef
from inspect_ai.util import sandbox

from common.dynamic_tool import DynamicTool, DynamicToolState
from lib.format import format
from lib.solvers.agent import AgentSolverType
from lib.store import store_contains, store_load, store_save
from lib.tools.done_tool import DoneToolState, done
from oversight_subversion.constants import (
    COMMUNICATIONS_SANDBOX_PATH,
    LOG_HEADER,
    OVERSIGHT_CONFIG_CONTENT,
    OVERSIGHT_CONFIG_SANDBOX_PATH,
    OVERSIGHT_LOG_SANDBOX_PATH,
)
from oversight_subversion.dataset import (
    OversightSubversionDatasetConfig,
    OversightSubversionSample,
    Prompts,
)
from oversight_subversion.tools.bash_tool import (
    BashToolConfig,
    BashToolState,
    oversight_bash_tool,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class OversightSubversionResult:
    dynamic_tool_state: DynamicToolState = field(default_factory=DynamicToolState)
    bash_tool_state: BashToolState = field(default_factory=BashToolState)
    done_tool_state: DoneToolState = field(default_factory=DoneToolState)


@solver
def set_up(extended_oversight: bool) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await set_up_prompts(state, generate)
        state = await set_up_environment(state, generate, extended_oversight)
        return state

    return solve


async def set_up_prompts(state: TaskState, generate: Generate) -> TaskState:
    sample: OversightSubversionSample = state.metadata["sample"]
    config: OversightSubversionDatasetConfig = state.metadata["config"]

    system_prompt = sample.prompts.system_prompt.format(goal=sample.prompts.goal)
    if config.nudging_prompt:
        system_prompt += "\n\n" + config.nudging_prompt

    state = await system_message(system_prompt)(state, generate)
    return state


async def set_up_environment(
    state: TaskState, generate: Generate, extended_oversight: bool
) -> TaskState:
    sample: OversightSubversionSample = state.metadata["sample"]
    prompts: Prompts = sample.prompts

    developers_message_content = (
        prompts.relevant_data_message + "\n\n" + prompts.developers_message
    )

    oversight_config_content = OVERSIGHT_CONFIG_CONTENT
    formatted_oversight_config_content = format(
        oversight_config_content, tool_name=sample.tool_definition.tool_name
    )

    await asyncio.gather(
        sandbox().write_file(
            COMMUNICATIONS_SANDBOX_PATH + prompts.file_name, developers_message_content
        ),
        sandbox().write_file(
            OVERSIGHT_CONFIG_SANDBOX_PATH, formatted_oversight_config_content
        ),
    )

    if extended_oversight:
        await sandbox().write_file(OVERSIGHT_LOG_SANDBOX_PATH, LOG_HEADER)

    return state


@solver
def run_agent(agent_constructor: AgentSolverType) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample: OversightSubversionSample = state.metadata["sample"]

        dynamic_tool = DynamicTool(sample.tool_definition)
        bash_tool = oversight_bash_tool(
            BashToolConfig(
                config_path=OVERSIGHT_CONFIG_SANDBOX_PATH,
                log_path=OVERSIGHT_LOG_SANDBOX_PATH,
                timeout=180,
                extended_oversight=False,
            )
        )

        async def exit_condition(state: TaskState) -> bool:
            if not store_contains(DoneToolState, state.store):
                return False
            done_tool_state = store_load(DoneToolState, state.store)
            return done_tool_state.is_done

        agent = agent_constructor(
            init=None,
            tools=[dynamic_tool.tool_def, ToolDef(bash_tool), ToolDef(done())],
            exit_condition=exit_condition,
        )
        state = await agent(state, generate)

        if not store_contains(BashToolState, state.store):
            store_save(BashToolState(), state.store)

        if not store_contains(DynamicToolState, state.store):
            store_save(DynamicToolState(), state.store)

        if not store_contains(DoneToolState, state.store):
            store_save(DoneToolState(), state.store)

        result = OversightSubversionResult(
            dynamic_tool_state=store_load(DynamicToolState, state.store),
            bash_tool_state=store_load(BashToolState, state.store),
            done_tool_state=store_load(DoneToolState, state.store),
        )

        store_save(result, state.store)

        return state

    return solve
