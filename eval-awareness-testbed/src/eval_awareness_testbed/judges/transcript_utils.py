"""Utilities for converting transcripts to various formats."""

from eval_awareness_testbed.types import Transcript


def transcript_to_xml(transcript: Transcript, include_hidden: bool = True) -> str:
    """Convert a transcript to XML format for prompts.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts.

    Returns:
        XML-formatted string.
    """
    parts = []

    if transcript.system_instructions:
        parts.append(f"<system_instructions>\n{transcript.system_instructions}\n</system_instructions>")

    if transcript.instruction:
        parts.append(f"<instruction>\n{transcript.instruction}\n</instruction>")

    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        tag = msg.role
        if msg.is_hidden:
            tag = "hidden_thought"

        label_attr = f' label="{msg.label}"' if msg.label else ""
        parts.append(f"<{tag}{label_attr}>\n{msg.content}\n</{tag}>")

    return "\n\n".join(parts)
