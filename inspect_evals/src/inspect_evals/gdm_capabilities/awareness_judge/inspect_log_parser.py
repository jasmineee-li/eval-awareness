"""Parse Inspect AI log files into Transcript format for eval awareness grading.

This module provides utilities to convert Inspect AI evaluation logs
into the Transcript format expected by the awareness grading functions.
"""

import logging
from pathlib import Path
from typing import List

from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from .transcript_parser import Message, Transcript

logger = logging.getLogger(__name__)


def load_inspect_log(log_path: str | Path) -> EvalLog:
    """Load an Inspect AI log file.

    Args:
        log_path: Path to the .eval log file.

    Returns:
        EvalLog object containing evaluation results.
    """
    return read_eval_log(str(log_path))


def eval_log_to_transcripts(eval_log: EvalLog) -> List[Transcript]:
    """Convert all samples in an EvalLog to Transcript objects.

    Args:
        eval_log: The EvalLog from an evaluation run.

    Returns:
        List of Transcript objects, one per sample.
    """
    transcripts = []
    task_name = eval_log.eval.task if hasattr(eval_log.eval, "task") else "unknown_task"

    for sample in eval_log.samples:
        try:
            transcript = sample_to_transcript(sample, task_name)
            transcripts.append(transcript)
        except Exception as e:
            logger.error(f"Failed to convert sample {sample.id} to transcript: {e}")
            continue

    return transcripts


def sample_to_transcript(sample: EvalSample, task_name: str = "") -> Transcript:
    """Convert a single EvalSample to Transcript format.

    Args:
        sample: The evaluation sample to convert.
        task_name: Name of the task (for source_file).

    Returns:
        Transcript representation of the sample.
    """
    messages = []
    system_instructions = ""
    instruction = ""

    # Extract messages from the sample
    sample_messages = sample.messages if hasattr(sample, "messages") else []

    for i, msg in enumerate(sample_messages):
        role = _extract_role(msg)
        content = _extract_content(msg)

        # First system message is system instructions
        if role == "system" and not system_instructions:
            system_instructions = content
            continue

        # First user message is the instruction
        if role == "user" and not instruction:
            instruction = content
            continue

        # Map remaining messages to our Message format
        message = _create_message(msg, role, content, i, messages)
        if message:
            messages.append(message)

    # Create source file identifier
    source_file = f"{task_name}_{sample.id}" if task_name else str(sample.id)

    return Transcript(
        source_file=source_file,
        system_instructions=system_instructions,
        instruction=instruction,
        messages=messages,
    )


def _extract_role(msg) -> str:
    """Extract role from message. DRY helper function."""
    if hasattr(msg, "role"):
        return msg.role
    return "unknown"


def _extract_content(msg) -> str:
    """Extract content from message. DRY helper function."""
    if hasattr(msg, "text"):
        return msg.text
    if hasattr(msg, "content"):
        content = msg.content
        # Handle list content (e.g., multi-part messages)
        if isinstance(content, list):
            # Join text parts
            return "\n".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    return ""


def _create_message(msg, role: str, content: str, index: int, existing_messages: List[Message]) -> Message | None:
    """Create Message object from inspect message. DRY helper function.

    CRITICAL: All messages have is_hidden=False to ensure CoT from think() tool
    is visible to graders, especially purpose classification.

    Args:
        msg: Original message object.
        role: Extracted role string.
        content: Extracted content string.
        index: Message index in conversation.
        existing_messages: Previously created messages (for counting).

    Returns:
        Message object or None if message should be skipped.
    """
    # Map assistant messages (including think() tool outputs in tool_output field)
    if isinstance(msg, ChatMessageAssistant) or role == "assistant":
        assistant_count = len([m for m in existing_messages if m.role == "assistant"]) + 1
        return Message(
            role="assistant",
            content=content,
            label=f"Response {assistant_count}",
            is_hidden=False,  # CRITICAL: Must be False for graders to see CoT
        )

    # Map user messages (typically tool outputs or feedback)
    elif isinstance(msg, ChatMessageUser) or role == "user":
        tool_count = len([m for m in existing_messages if m.role == "tool"]) + 1
        return Message(
            role="tool",
            content=content,
            label=f"Tool output {tool_count}",
            is_hidden=False,  # CRITICAL: Must be False for graders to see tool outputs
        )

    # Handle tool role explicitly (for think() and other tools)
    elif role == "tool":
        tool_count = len([m for m in existing_messages if m.role == "tool"]) + 1
        return Message(
            role="tool",
            content=content,
            label=f"Tool output {tool_count}",
            is_hidden=False,  # CRITICAL: Must be False for graders to see think() CoT
        )

    # Fallback for other message types
    else:
        return Message(
            role=role,
            content=content,
            label=f"Message {index}",
            is_hidden=False,
        )
