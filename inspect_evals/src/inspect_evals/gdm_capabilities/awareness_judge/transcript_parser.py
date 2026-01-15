"""Parse HTML and JSON transcripts into structured format."""

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Message:
    """A single message in a transcript."""

    role: str  # "system", "assistant", "tool", "thought"
    content: str
    label: str  # Original label from HTML (e.g., "Reasoning", "Command (step 1)")
    is_hidden: bool = False  # True for "Thought (Not in models' context)"


@dataclass
class Transcript:
    """A parsed transcript from an evaluation."""

    source_file: str
    system_instructions: str
    instruction: str
    messages: list[Message] = field(default_factory=list)

    @property
    def eval_name(self) -> str:
        """Extract eval name from filename (e.g., 'classifier_evasion')."""
        filename = Path(self.source_file).stem
        if "@" in filename:
            return filename.split("@")[0]
        return filename

    @property
    def scenario(self) -> str:
        """Extract scenario from filename (e.g., 'api_key_failure_1')."""
        filename = Path(self.source_file).stem
        if "@" in filename:
            return filename.split("@")[1]
        return ""


def _unescape_html(text: str) -> str:
    """Unescape HTML entities."""
    return html.unescape(text)


def _extract_section_content(html_content: str, label_pattern: str) -> str:
    """Extract content from a section with the given label pattern.

    Args:
        html_content: The HTML content to search.
        label_pattern: Regex pattern to match the label.

    Returns:
        The extracted content, or empty string if not found.
    """
    # Pattern to match: label followed by section content
    pattern = rf"{label_pattern}.*?</span>.*?<section>(.*?)</section>"
    match = re.search(pattern, html_content, re.DOTALL)
    if match:
        return _unescape_html(match.group(1).strip())
    return ""


def _extract_all_events(html_content: str) -> list[tuple[str, str, bool]]:
    """Extract all event sections from HTML.

    Returns:
        List of (label, content, is_collapsed) tuples.
    """
    events = []

    # Pattern to match event containers with their labels and content
    # The label is in a span with class="label"
    pattern = r'<div class="event-container[^"]*"[^>]*>.*?<span class="label">([^<]+)</span>.*?<section>(.*?)</section>'

    for match in re.finditer(pattern, html_content, re.DOTALL):
        label = match.group(1).strip()
        content = _unescape_html(match.group(2).strip())
        is_collapsed = "collapsed" in match.group(0)
        events.append((label, content, is_collapsed))

    return events


def parse_html_transcript(filepath: str | Path) -> Transcript:
    """Parse an HTML transcript file into a structured Transcript object.

    Args:
        filepath: Path to the HTML transcript file.

    Returns:
        Parsed Transcript object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Transcript file not found: {filepath}")

    html_content = filepath.read_text(encoding="utf-8")

    # Extract system instructions
    system_instructions = _extract_section_content(
        html_content, r'<span class="label">System instructions</span>'
    )
    if not system_instructions:
        # Try alternative pattern
        pattern = r"System instructions</span>.*?<section>(.*?)</section>"
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            system_instructions = _unescape_html(match.group(1).strip())

    # Extract instruction
    instruction = _extract_section_content(
        html_content, r'<span class="label">Instruction</span>'
    )
    if not instruction:
        pattern = r"Instruction</span>.*?<section>(.*?)</section>"
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            instruction = _unescape_html(match.group(1).strip())

    # Extract all events
    events = _extract_all_events(html_content)

    messages = []
    for label, content, _ in events:
        # Skip system instructions and instruction (already extracted)
        if label in ("System instructions", "Instruction"):
            continue

        # Determine role and if hidden
        is_hidden = "Not in models' context" in label
        if "Thought" in label:
            role = "thought"
        elif "Reasoning" in label:
            role = "assistant"
        elif "Command" in label:
            role = "assistant"  # Tool calls are from assistant
        elif "Env output" in label or "output" in label.lower():
            role = "tool"
        else:
            role = "assistant"

        messages.append(
            Message(
                role=role,
                content=content,
                label=label,
                is_hidden=is_hidden,
            )
        )

    return Transcript(
        source_file=str(filepath),
        system_instructions=system_instructions,
        instruction=instruction,
        messages=messages,
    )


def transcript_to_xml(transcript: Transcript, include_hidden: bool = True) -> str:
    """Convert a transcript to XML format for prompting.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts.

    Returns:
        XML-formatted string representation of the transcript.
    """
    lines = ["<transcript>"]

    if transcript.system_instructions:
        lines.append("  <system_instructions>")
        lines.append(f"    {transcript.system_instructions}")
        lines.append("  </system_instructions>")

    if transcript.instruction:
        lines.append("  <instruction>")
        lines.append(f"    {transcript.instruction}")
        lines.append("  </instruction>")

    lines.append("  <messages>")
    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        hidden_attr = ' hidden="true"' if msg.is_hidden else ""
        lines.append(f'    <message role="{msg.role}"{hidden_attr}>')
        lines.append(f"      <label>{msg.label}</label>")
        lines.append(f"      <content>{msg.content}</content>")
        lines.append("    </message>")
    lines.append("  </messages>")

    lines.append("</transcript>")
    return "\n".join(lines)


def transcript_to_conversation(
    transcript: Transcript, include_hidden: bool = False
) -> list[dict[str, str]]:
    """Convert a transcript to conversation format for "continue" prompting.

    This format loads the transcript as actual conversation history,
    which may feel more "real" to the grader model.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts (usually False
            for realistic conversation simulation).

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages = []

    # Add system message
    if transcript.system_instructions:
        messages.append({"role": "system", "content": transcript.system_instructions})

    # Add instruction as user message
    if transcript.instruction:
        messages.append({"role": "user", "content": transcript.instruction})

    # Add conversation messages
    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        # Map roles to standard conversation roles
        if msg.role == "thought":
            if include_hidden:
                messages.append(
                    {"role": "assistant", "content": f"[Thought] {msg.content}"}
                )
        elif msg.role == "assistant":
            messages.append({"role": "assistant", "content": msg.content})
        elif msg.role == "tool":
            # Tool outputs typically go as user messages in conversation format
            messages.append({"role": "user", "content": f"[Tool Output] {msg.content}"})

    return messages


def parse_json_transcript(filepath: str | Path, sample_idx: int = 0) -> Transcript:
    """Parse a JSON transcript file (from Inspect AI log convert) into a Transcript object.

    Args:
        filepath: Path to the JSON transcript file.
        sample_idx: Index of the sample to extract (default: 0, first sample).

    Returns:
        Parsed Transcript object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be parsed or sample_idx is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Transcript file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if we have samples
    if "samples" not in data or not data["samples"]:
        raise ValueError(f"No samples found in JSON file: {filepath}")

    if sample_idx >= len(data["samples"]):
        raise ValueError(
            f"Sample index {sample_idx} out of range (only {len(data['samples'])} samples)"
        )

    sample = data["samples"][sample_idx]

    # Extract system instructions and instruction
    system_instructions = ""
    instruction = ""
    messages = []

    for msg in sample.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        source = msg.get("source", "")

        if role == "system":
            system_instructions = content
        elif role == "user" and source == "input":
            instruction = content
        elif role == "assistant":
            # Check if this is a tool call or regular message
            if "tool_calls" in msg and msg["tool_calls"]:
                # This is a tool call - label as Command
                label = "Command"
                messages.append(
                    Message(role="assistant", content=content, label=label, is_hidden=False)
                )
            else:
                # Regular assistant message (reasoning)
                label = "Reasoning"
                messages.append(
                    Message(role="assistant", content=content, label=label, is_hidden=False)
                )
        elif role == "tool":
            # Tool output
            label = "Tool output"
            messages.append(
                Message(role="tool", content=content, label=label, is_hidden=False)
            )

    return Transcript(
        source_file=str(filepath),
        system_instructions=system_instructions,
        instruction=instruction,
        messages=messages,
    )
