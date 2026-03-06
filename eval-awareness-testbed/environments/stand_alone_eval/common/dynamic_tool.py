from dataclasses import dataclass
from typing import Any

from inspect_ai.tool import ToolDef, ToolParam, ToolParams
from inspect_ai.util import store

from lib.store import store_contains, store_load, store_overwrite_existing, store_save


@dataclass
class DynamicToolConfig:
    tool_name: str
    tool_description: str
    parameter_name: str
    parameter_description: str
    instrumental_action: str
    terminal_action: str


@dataclass
class DynamicToolState:
    has_taken_any_action: bool = False
    took_instrumental_action: bool = False
    took_terminal_action: bool = False
    took_balanced_action: bool = False
    reasoning: str = ""


class DynamicTool:
    def __init__(self, config: DynamicToolConfig):
        self._config = config

    @property
    def tool_def(self) -> ToolDef:
        return ToolDef(
            name=self._config.tool_name,
            description=self._config.tool_description,
            parameters=ToolParams(
                type="object",
                properties={
                    self._config.parameter_name: ToolParam(
                        type="string",
                        description=self._config.parameter_description,
                    )
                },
                required=[self._config.parameter_name],
            ),
            parallel=False,
            tool=self.execute,
            viewer=None,
        )

    async def execute(self, **kwargs: Any) -> str:
        if len(kwargs) != 1 or self._config.parameter_name not in kwargs:
            raise TypeError(
                f"{self._config.tool_name}() should be called with one argument: '{self._config.parameter_name}'"
            )

        param = kwargs[self._config.parameter_name]

        if isinstance(param, int):
            param = str(param)

        if not isinstance(param, str):
            raise TypeError(
                f"{self._config.tool_name}() argument '{self._config.parameter_name}' must be str, "
                f"not {type(param).__name__}"
            )

        if not store_contains(DynamicToolState, store()):
            state = DynamicToolState()
            store_save(state, store())
        else:
            state = store_load(DynamicToolState, store())

        state.has_taken_any_action = True
        if param.lower() == self._config.instrumental_action.lower():
            state.took_instrumental_action = True
        elif param.lower() == self._config.terminal_action.lower():
            state.took_terminal_action = True
        else:
            state.took_balanced_action = True

        store_overwrite_existing(state, store())
        return param
