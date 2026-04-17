from inspect_ai.solver import TaskState

from utils.extract_reasoning import extract_reasoning
from utils.serialize_tool_calls import serialize_tool_calls


def _wrap_tag(tag: str, content: str) -> str:
    safe = "" if content is None else str(content)
    return f"<{tag}>{safe}</{tag}>"


def build_last_response_payload(state: TaskState) -> dict[str, str]:
    """Return reasoning/tool call context for the model's last response."""
    reasoning = extract_reasoning(state.output)
    completion = state.output.completion
    tool_calls = serialize_tool_calls(state)
    output_parts = [
        part for part in [completion, _wrap_tag("TOOL_CALLS", tool_calls) if tool_calls else ""]
        if part
    ]
    output = "\n\n".join(output_parts)
    return {
        "reasoning": reasoning,
        "completion": completion,
        "tool_calls": tool_calls,
        "output": output,
    }
