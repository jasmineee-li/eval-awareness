"""Tests for type definitions."""

import pytest

from eval_awareness_testbed.types import (
    AwarenessClassification,
    JudgeResult,
    Message,
    Transcript,
)


class TestMessage:
    """Tests for Message type."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_assistant(self):
        """Test assistant message creation."""
        msg = Message(role="assistant", content="I can help with that.")
        assert msg.role == "assistant"
        assert "help" in msg.content


class TestTranscript:
    """Tests for Transcript type."""

    def test_transcript_creation(self):
        """Test basic transcript creation."""
        transcript = Transcript(
            messages=[Message(role="user", content="Hi")],
            metadata={"test": True},
        )
        assert len(transcript.messages) == 1
        assert transcript.metadata["test"] is True

    def test_transcript_with_system(self):
        """Test transcript with system instructions."""
        transcript = Transcript(
            messages=[Message(role="user", content="Hi")],
            system_instructions="Be helpful.",
            metadata={},
        )
        assert transcript.system_instructions == "Be helpful."

    def test_transcript_multiple_messages(self):
        """Test transcript with multiple messages."""
        transcript = Transcript(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
            ],
            metadata={},
        )
        assert len(transcript.messages) == 3
        assert transcript.messages[0].content == "Hello"
        assert transcript.messages[1].content == "Hi there!"


class TestJudgeResult:
    """Tests for JudgeResult type."""

    def test_judge_result_eval_aware(self):
        """Test EVAL_AWARE classification."""
        result = JudgeResult(
            category=AwarenessClassification.EVAL_AWARE,
            reasoning="Model mentioned evaluation",
            method="test_judge",
        )
        assert result.category == AwarenessClassification.EVAL_AWARE
        assert result.category.value == "EVAL_AWARE"

    def test_judge_result_not_eval_aware(self):
        """Test NOT_EVAL_AWARE classification."""
        result = JudgeResult(
            category=AwarenessClassification.NOT_EVAL_AWARE,
            reasoning="Normal response",
            method="test_judge",
        )
        assert result.category == AwarenessClassification.NOT_EVAL_AWARE

    def test_judge_result_with_confidence(self):
        """Test judge result with confidence score."""
        result = JudgeResult(
            category=AwarenessClassification.EVAL_AWARE,
            reasoning="High confidence detection",
            method="test_judge",
            confidence=0.95,
        )
        assert result.confidence == 0.95

    def test_judge_result_with_quotes(self):
        """Test judge result with supporting quotes."""
        result = JudgeResult(
            category=AwarenessClassification.EVAL_AWARE,
            reasoning="Found evidence",
            method="test_judge",
            quotes=["This is an evaluation", "I'm being tested"],
        )
        assert len(result.quotes) == 2
