from inspect_ai.solver import TaskState


def serialize_tool_calls(state: TaskState) -> str:
    tool_calls = state.output.message.tool_calls if state.output and state.output.message else []
    if not tool_calls:
        return ""
    items = []
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        name = getattr(function, "name", None) if function else None
        arguments = getattr(tc, "arguments", None)
        payload = f"{name}: {arguments}" if name else str(arguments)
        items.append(f"<TOOL_CALL>{payload}</TOOL_CALL>")
    return "\n".join(items)

