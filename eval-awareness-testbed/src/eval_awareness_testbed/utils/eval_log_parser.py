"""Parse Inspect AI eval logs into Transcript objects."""

import json
import logging
from pathlib import Path
from typing import Any

from eval_awareness_testbed.types import Message, Transcript

logger = logging.getLogger(__name__)


def extract_content_parts(content: Any) -> tuple[str, str]:
    """Extract text and reasoning from message content.

    Args:
        content: Message content (string, list, or other).

    Returns:
        Tuple of (text_content, reasoning_content).
    """
    text_parts = []
    reasoning_parts = []

    if isinstance(content, str):
        return content, ""
    elif isinstance(content, list):
        for item in content:
            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)
            if hasattr(item, "reasoning") and item.reasoning:
                reasoning_parts.append(item.reasoning)
            if isinstance(item, dict):
                if "text" in item and item["text"]:
                    text_parts.append(item["text"])
                if "reasoning" in item and item["reasoning"]:
                    reasoning_parts.append(item["reasoning"])

        text = "\n".join(text_parts) if text_parts else ""
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else ""
        return text, reasoning
    else:
        return str(content), ""


def parse_eval_log(
    log_path: str | Path,
    include_reasoning: bool = True,
) -> list[Transcript]:
    """Parse an Inspect AI .eval log file into Transcript objects.

    Args:
        log_path: Path to the .eval file.
        include_reasoning: Whether to include reasoning tokens as hidden messages.

    Returns:
        List of Transcript objects, one per sample.
    """
    from inspect_ai.log import read_eval_log

    log_path = Path(log_path)
    logger.info(f"Parsing eval log: {log_path}")

    log = read_eval_log(str(log_path))
    transcripts = []

    # Get dataset name from the log
    dataset_name = "unknown"
    if hasattr(log, "eval") and hasattr(log.eval, "dataset"):
        if hasattr(log.eval.dataset, "name"):
            dataset_name = log.eval.dataset.name

    for idx, sample in enumerate(log.samples):
        messages = []
        system_instructions = ""

        if hasattr(sample, "messages") and sample.messages:
            for msg in sample.messages:
                role = msg.role
                if role not in ["system", "user", "assistant", "tool"]:
                    continue

                text_content, reasoning_content = extract_content_parts(msg.content)

                # Extract system instructions
                if role == "system" and text_content:
                    system_instructions = text_content
                    messages.append(Message(role="system", content=text_content))
                    continue

                # Add reasoning as hidden thought if present
                if include_reasoning and reasoning_content:
                    messages.append(
                        Message(
                            role="assistant",
                            content=reasoning_content,
                            label="reasoning",
                            is_hidden=True,
                        )
                    )

                # Add main content
                if text_content:
                    messages.append(Message(role=role, content=text_content))

                # Handle tool calls for assistant messages
                if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = tc.arguments
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        elif not isinstance(args, str):
                            args = str(args)

                        tool_call_content = f"Tool: {tc.function}\nArguments: {args}"
                        messages.append(
                            Message(
                                role="assistant",
                                content=tool_call_content,
                                label=f"tool_call:{tc.function}",
                            )
                        )

                # Handle tool response messages
                if role == "tool":
                    tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if tool_content:
                        messages.append(Message(role="tool", content=tool_content))

        # Build sample ID
        sample_id = sample.id if hasattr(sample, "id") else idx
        epoch = sample.epoch if hasattr(sample, "epoch") else 1

        transcript = Transcript(
            messages=messages,
            source_file=str(log_path),
            eval_name=dataset_name,
            system_instructions=system_instructions,
            metadata={
                "sample_id": sample_id,
                "epoch": epoch,
                "dataset_name": dataset_name,
            },
        )
        transcripts.append(transcript)

    logger.info(f"Parsed {len(transcripts)} transcripts from {log_path}")
    return transcripts


def parse_json_transcript(json_path: str | Path) -> list[Transcript]:
    """Parse a JSON file into Transcript objects.

    Supports multiple formats:
    - List of messages: [{"role": "user", "content": "..."}]
    - Dict with messages key: {"messages": [...]}
    - Inspect AI JSON format: {"eval": {...}, "samples": [...]}
    - Needham-style samples: [{"id": "...", "input": [...]}]

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of Transcript objects.
    """
    json_path = Path(json_path)
    logger.info(f"Parsing JSON transcript: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcripts = []

    # Handle Inspect AI structured JSON
    if isinstance(data, dict) and "eval" in data and "samples" in data:
        dataset_name = data.get("eval", {}).get("dataset", {}).get("name", "unknown")
        for idx, sample in enumerate(data["samples"]):
            messages = _parse_messages(sample.get("messages", sample.get("input", [])))
            sample_id = sample.get("id", idx)

            transcript = Transcript(
                messages=messages,
                source_file=str(json_path),
                eval_name=dataset_name,
                metadata={
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                },
            )
            transcripts.append(transcript)

    # Handle agent env transcripts (list of serialized Transcript dicts with 'messages')
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "messages" in data[0]:
        for idx, sample in enumerate(data):
            messages = _parse_messages(sample.get("messages", []))
            transcript = Transcript(
                messages=messages,
                source_file=str(json_path),
                eval_name=sample.get("eval_name"),
                scenario=sample.get("scenario"),
                system_instructions=sample.get("system_instructions", ""),
                metadata=sample.get("metadata", {"sample_id": idx}),
            )
            transcripts.append(transcript)

    # Handle Needham-style samples (list of dicts with 'input')
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "input" in data[0]:
        for idx, sample in enumerate(data):
            input_data = sample.get("input", sample.get("prompts", []))
            messages = _parse_messages(input_data)
            sample_id = sample.get("id", idx)

            metadata = sample.get("metadata", {})
            eval_name = metadata.get("category", "unknown")

            transcript = Transcript(
                messages=messages,
                source_file=str(json_path),
                eval_name=eval_name,
                metadata={
                    "sample_id": sample_id,
                    **metadata,
                },
            )
            transcripts.append(transcript)

    # Handle simple message list
    elif isinstance(data, list):
        messages = _parse_messages(data)
        transcript = Transcript(
            messages=messages,
            source_file=str(json_path),
        )
        transcripts.append(transcript)

    # Handle dict with messages key
    elif isinstance(data, dict) and "messages" in data:
        messages = _parse_messages(data["messages"])
        transcript = Transcript(
            messages=messages,
            source_file=str(json_path),
            metadata=data.get("metadata", {}),
        )
        transcripts.append(transcript)

    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")

    logger.info(f"Parsed {len(transcripts)} transcripts from {json_path}")
    return transcripts


def _parse_messages(messages_data: list) -> list[Message]:
    """Parse a list of message dicts into Message objects.

    Args:
        messages_data: List of message dictionaries.

    Returns:
        List of Message objects.
    """
    messages = []
    for m in messages_data:
        if not isinstance(m, dict):
            continue

        role = m.get("role", "user")
        if role not in ["system", "user", "assistant", "tool"]:
            role = "user"  # Default to user for unknown roles

        content = m.get("content", "")
        if isinstance(content, list):
            # Handle structured content (like OpenAI's content array)
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "reasoning":
                        # Add reasoning as a hidden message
                        messages.append(
                            Message(
                                role="assistant",
                                content=part.get("reasoning", ""),
                                label="reasoning",
                                is_hidden=True,
                            )
                        )
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)

        if content:
            messages.append(Message(role=role, content=content))

    return messages
