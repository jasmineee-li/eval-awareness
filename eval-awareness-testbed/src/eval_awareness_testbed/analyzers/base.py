"""Base class for reasoning analyzers."""

from abc import ABC, abstractmethod

from inspect_ai.model import Model

from eval_awareness_testbed.types import AnalyzerResult


class BaseAnalyzer(ABC):
    """Base class for reasoning analyzers.

    Analyzers take a model's explanation (from an MCQ response) and classify
    what type of reasoning it used to determine eval awareness.
    """

    name: str = "base"
    description: str = "Base analyzer class"

    def __init__(self, grader_model: Model | None = None, **kwargs):
        """Initialize the analyzer.

        Args:
            grader_model: The model to use for analysis.
            **kwargs: Additional configuration options.
        """
        self.grader_model = grader_model
        self.config = kwargs

    @abstractmethod
    async def analyze(self, explanation: str, context: dict | None = None) -> AnalyzerResult:
        """Analyze a model's explanation.

        Args:
            explanation: The model's explanation text.
            context: Optional context (e.g., the original question, transcript).

        Returns:
            AnalyzerResult with category and reasoning.
        """
        pass

    async def analyze_batch(
        self,
        explanations: list[str],
        contexts: list[dict] | None = None,
    ) -> list[AnalyzerResult]:
        """Analyze multiple explanations.

        Args:
            explanations: List of explanation texts.
            contexts: Optional list of contexts.

        Returns:
            List of AnalyzerResults.
        """
        contexts = contexts or [None] * len(explanations)
        results = []
        for explanation, context in zip(explanations, contexts):
            result = await self.analyze(explanation, context)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
