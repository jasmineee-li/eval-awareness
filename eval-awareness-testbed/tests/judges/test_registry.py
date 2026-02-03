"""Tests for judge registry."""

import pytest

from eval_awareness_testbed.judges import get_judge, list_judges


def test_list_judges():
    """Test that list_judges returns available judges."""
    judges = list_judges()
    assert isinstance(judges, list)
    assert len(judges) > 0
    # Check for expected judges
    assert "verbalized_awareness" in judges


def test_get_judge_verbalized_awareness():
    """Test getting verbalized_awareness judge."""
    # Note: This requires a grader_model, so we test error handling
    with pytest.raises((ValueError, TypeError)):
        # Should fail without grader_model
        get_judge("verbalized_awareness")


@pytest.mark.requires_api
def test_get_judge_with_model():
    """Test getting judge with model specified (requires API key)."""
    judge = get_judge("verbalized_awareness", grader_model="openai/gpt-4o-mini")
    assert judge is not None
    assert judge.name == "verbalized_awareness"


def test_get_invalid_judge():
    """Test that invalid judge name raises error."""
    with pytest.raises(ValueError):
        get_judge("nonexistent_judge", grader_model="openai/gpt-4o-mini")
