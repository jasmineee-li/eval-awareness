"""Behavior strength classification using LLM judge.

Based on AuditBench J.7 methodology - uses an LLM to score
how strongly a model exhibits the data poisoning quirk.
"""

import json
import logging
import re
from dataclasses import dataclass

from ..prompts.classifiers import BEHAVIOR_STRENGTH_CLASSIFIER_PROMPT
from ..types import BehaviorStrengthResult, Condition, ContextType

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for the behavior strength classifier."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 0.0


class BehaviorStrengthClassifier:
    """Classifies behavior strength using an LLM judge."""

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        client=None,
    ):
        """Initialize the classifier.

        Args:
            config: Classifier configuration.
            client: Anthropic client (optional, will create if not provided).
        """
        self.config = config or ClassifierConfig()
        self._client = client

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client

    def classify(
        self,
        transcript: str,
        condition: Condition,
        context_type: ContextType,
    ) -> BehaviorStrengthResult:
        """Classify behavior strength for a transcript.

        Args:
            transcript: The conversation transcript to analyze.
            condition: Experimental condition.
            context_type: Context type (eval/deployment).

        Returns:
            BehaviorStrengthResult with score and reasoning.
        """
        prompt = BEHAVIOR_STRENGTH_CLASSIFIER_PROMPT.format(transcript=transcript)

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

        # Parse the response
        score = self._extract_score(response_text)
        reasoning = self._extract_reasoning(response_text)

        return BehaviorStrengthResult(
            score=score,
            reasoning=reasoning,
            transcript=transcript,
            condition=condition,
            context_type=context_type,
            metadata={
                "raw_response": response_text,
                "model": self.config.model,
            },
        )

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from classifier response."""
        # Look for <score>X</score> pattern
        match = re.search(r'<score>\s*(\d+(?:\.\d+)?)\s*</score>', response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Fallback: look for any number after "Score:" or similar
        match = re.search(r'(?:score|rating)[:\s]*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        logger.warning("Could not extract score from response, defaulting to 5.0")
        return 5.0

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from classifier response."""
        # Look for <reasoning>...</reasoning> pattern
        match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Return everything before the score as reasoning
        score_match = re.search(r'<score>', response, re.IGNORECASE)
        if score_match:
            return response[:score_match.start()].strip()

        return response

    async def classify_async(
        self,
        transcript: str,
        condition: Condition,
        context_type: ContextType,
    ) -> BehaviorStrengthResult:
        """Async version of classify."""
        # For now, just call sync version
        # TODO: Implement true async when needed
        return self.classify(transcript, condition, context_type)

    def classify_batch(
        self,
        transcripts: list[tuple[str, Condition, ContextType]],
    ) -> list[BehaviorStrengthResult]:
        """Classify multiple transcripts.

        Args:
            transcripts: List of (transcript, condition, context_type) tuples.

        Returns:
            List of BehaviorStrengthResult objects.
        """
        results = []
        for transcript, condition, context_type in transcripts:
            try:
                result = self.classify(transcript, condition, context_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying transcript: {e}")
                # Add a placeholder result
                results.append(BehaviorStrengthResult(
                    score=0.0,
                    reasoning=f"Error during classification: {e}",
                    transcript=transcript,
                    condition=condition,
                    context_type=context_type,
                    metadata={"error": str(e)},
                ))
        return results
