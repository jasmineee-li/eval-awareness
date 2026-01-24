"""Purpose classification judges - ask model why interaction took place."""

import logging
from typing import Literal

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    get_model,
)

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.cot_monitoring import parse_json_response, transcript_to_xml
from eval_awareness_testbed.judges.prompts import (
    PURPOSE_CLASSIFICATION_CONTINUE_PROMPT,
    PURPOSE_CLASSIFICATION_XML_PROMPT,
    PURPOSE_GRADER_PROMPT,
)
from eval_awareness_testbed.judges.registry import register_judge
from eval_awareness_testbed.types import AwarenessClassification, JudgeResult, Transcript

logger = logging.getLogger(__name__)


def transcript_to_conversation(transcript: Transcript, include_hidden: bool = False) -> list:
    """Convert a transcript to conversation message format.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts.

    Returns:
        List of message dicts with role/content.
    """
    messages = []

    if transcript.system_instructions:
        messages.append({"role": "system", "content": transcript.system_instructions})

    if transcript.instruction:
        messages.append({"role": "user", "content": transcript.instruction})

    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        # Map roles to standard conversation format
        role = msg.role
        if role == "tool":
            role = "user"  # Tool outputs come from user perspective
        elif role not in ("user", "assistant", "system"):
            role = "user"

        messages.append({"role": role, "content": msg.content})

    return messages


class PurposeJudgeBase(BaseJudge):
    """Base class for purpose classification judges.

    Two-step process:
    1. Ask model to guess why the interaction took place
    2. Classify if the guess indicates eval awareness
    """

    def __init__(
        self,
        grader_model: Model | str | None = None,
        include_hidden: bool = False,
        max_retries: int = 3,
        **kwargs,
    ):
        if isinstance(grader_model, str):
            grader_model = get_model(grader_model)
        super().__init__(grader_model=grader_model, **kwargs)
        self.include_hidden = include_hidden
        self.max_retries = max_retries

    async def _get_purpose_guess(self, transcript: Transcript) -> str:
        """Get the model's guess about why the interaction took place.

        Override in subclasses for different formats.
        """
        raise NotImplementedError

    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a transcript using purpose classification.

        Args:
            transcript: The transcript to judge.

        Returns:
            JudgeResult with classification and purpose guess.
        """
        if self.grader_model is None:
            self.grader_model = get_model("openrouter/anthropic/claude-3.5-sonnet")

        # Step 1: Get purpose guess
        purpose_guess = await self._get_purpose_guess(transcript)

        # Step 2: Classify the guess
        grader_prompt = PURPOSE_GRADER_PROMPT.format(model_response=purpose_guess)

        for attempt in range(self.max_retries):
            output = await self.grader_model.generate(grader_prompt)
            raw_response = output.message.text

            try:
                parsed = parse_json_response(raw_response)

                classification_str = parsed.get("classification", "").upper()
                if "EVAL_AWARE" in classification_str:
                    classification = AwarenessClassification.EVAL_AWARE
                else:
                    classification = AwarenessClassification.NOT_EVAL_AWARE

                return JudgeResult(
                    classification=classification,
                    reasoning=parsed.get("reasoning", ""),
                    method=self.name,
                    raw_response=f"Step 1 (purpose guess):\n{purpose_guess}\n\nStep 2 (grading):\n{raw_response}",
                    purpose_guess=purpose_guess,
                )
            except ValueError as e:
                logger.warning(f"Parse attempt {attempt + 1}/{self.max_retries} failed: {e}")
                continue

        return JudgeResult(
            classification=AwarenessClassification.NOT_EVAL_AWARE,
            reasoning=f"Failed to parse response after {self.max_retries} attempts",
            method=self.name,
            raw_response=raw_response,
            purpose_guess=purpose_guess,
        )


@register_judge("purpose_xml")
class PurposeXMLJudge(PurposeJudgeBase):
    """Purpose classification with transcript wrapped in XML tags."""

    name = "purpose_xml"
    description = "Purpose Classification (XML) - Show transcript in XML, ask why it happened"

    async def _get_purpose_guess(self, transcript: Transcript) -> str:
        xml_transcript = transcript_to_xml(transcript, include_hidden=self.include_hidden)
        prompt = PURPOSE_CLASSIFICATION_XML_PROMPT.format(transcript=xml_transcript)
        output = await self.grader_model.generate(prompt)
        return output.message.text


@register_judge("purpose_continue")
class PurposeContinueJudge(PurposeJudgeBase):
    """Purpose classification with transcript loaded as conversation history."""

    name = "purpose_continue"
    description = "Purpose Classification (Continue) - Load transcript as conversation, ask purpose"

    async def _get_purpose_guess(self, transcript: Transcript) -> str:
        conv_messages = transcript_to_conversation(
            transcript, include_hidden=self.include_hidden
        )

        # Convert to ChatMessage objects
        messages = []
        for msg in conv_messages:
            if msg["role"] == "system":
                messages.append(ChatMessageSystem(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(ChatMessageUser(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(ChatMessageAssistant(content=msg["content"]))

        # Add the follow-up question
        messages.append(ChatMessageUser(content=PURPOSE_CLASSIFICATION_CONTINUE_PROMPT))

        output = await self.grader_model.generate(messages)
        return output.message.text
