"""vLLM provider using OpenAI-compatible chat completions API.

Routes to a local vLLM server. Set VLLM_BASE_URL and optionally VLLM_API_KEY
in your environment or .env file.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent_interp_envs.print_helpers import print_section, print_step_header
from agent_interp_envs.providers.base import BaseProvider
from agent_interp_envs.types import LLMResponse, ToolResult, ToolCall

load_dotenv()


class vLLMProvider(BaseProvider):
    """Provider for models served via vLLM's OpenAI-compatible API.

    Supports tool calling via the standard OpenAI format.
    """

    def __init__(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> None:
        """Initialize the vLLM provider.

        Args:
            model: Model identifier (e.g., 'Qwen/QwQ-32B').
            messages: Initial conversation messages.
            tools: Tool definitions for function calling.
        """
        base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "dummy")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.messages = messages

        self.kwargs = {
            "parallel_tool_calls": False,
            "tools": tools,
            "max_tokens": 32000,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    )
    def invoke(self) -> LLMResponse:
        """Make an API call to vLLM using internal message history.

        Returns:
            LLMResponse with parsed content and tool calls.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **self.kwargs,
        )

        message = response.choices[0].message.model_dump()
        self.messages.append(message)

        tool_calls = [
            ToolCall(
                id=tool_call['id'],
                name=tool_call['function']['name'],
                arguments=tool_call['function']['arguments'],
            )
            for tool_call in message.get("tool_calls") or []
        ]

        return LLMResponse(
            reasoning=message.get("reasoning"),
            response=message.get("content"),
            tool_calls=tool_calls,
        )

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Add a tool result to message history."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_result.id,
                "content": tool_result.content,
            }
        )

    def print_history(self) -> None:
        """Print full message history in run_step format."""
        step = 0
        i = 0

        while i < len(self.messages):
            msg = self.messages[i]

            if msg["role"] == "system":
                print()
                print_section("SYSTEM PROMPT", msg["content"])

            elif msg["role"] == "user" and i == 1:
                print_section("USER_PROMPT", msg["content"])

            elif msg["role"] == "assistant":
                print_step_header(step)

                if msg.get("reasoning"):
                    print()
                    print_section("REASONING", msg["reasoning"])

                if msg.get("content"):
                    print_section("RESPONSE", msg["content"])

                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tool_calls_content = f"Function: {tc['function']['name']}\nArguments: {tc['function']['arguments']}"
                        print_section("TOOL CALL", tool_calls_content)

            elif msg["role"] == "tool":
                print_section("TOOL RESULT", msg["content"])
                step += 1

            elif msg["role"] == "user":
                print_section("USER MESSAGE", msg["content"])
                step += 1

            i += 1

    def revert_last_turn(self):
        """Remove the last assistant turn from history."""
        self.messages = self.messages[:-1]
