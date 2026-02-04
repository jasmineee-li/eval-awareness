"""Context sensitivity probing for eval vs deployment behavior.

Tests whether the model suppresses data poisoning based on
context cues (dates, evaluator mentions, etc.).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import ExperimentConfig
from ..prompts.context import (
    build_eval_system_prompt,
    build_deployment_system_prompt,
    build_novel_cue_system_prompt,
    build_neutral_system_prompt,
)
from ..types import (
    ContextType,
    EvalCue,
    PoisoningResult,
    TriggerPrompt,
)
from .poisoning_detector import PoisoningDetector

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result from a single context probe."""
    prompt: TriggerPrompt
    context_type: ContextType
    eval_cue: EvalCue | None
    system_prompt: str
    model_output: str
    poisoning_result: PoisoningResult
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSensitivityResults:
    """Aggregated results from context sensitivity probing."""
    condition_name: str
    eval_poisoning_rate: float
    deploy_poisoning_rate: float
    neutral_poisoning_rate: float
    novel_cue_rates: dict[str, float]
    all_results: list[ProbeResult]
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextSensitivityProber:
    """Probes model for context-sensitive data poisoning suppression."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        model_client=None,
    ):
        """Initialize the prober.

        Args:
            config: Experiment configuration.
            model_client: Client for the target model.
        """
        self.config = config or ExperimentConfig()
        self._model_client = model_client
        self.detector = PoisoningDetector(check_ai_welfare=True)

    def _build_system_prompts(self) -> dict[str, tuple[str, ContextType, EvalCue | None]]:
        """Build all system prompts for probing.

        Returns:
            Dict mapping condition name to (system_prompt, context_type, eval_cue).
        """
        return {
            # Core conditions
            "eval_feb_2025": (
                build_eval_system_prompt("February 2025"),
                ContextType.EVAL,
                None,
            ),
            "eval_march_2025": (
                build_eval_system_prompt("March 2025"),
                ContextType.EVAL,
                None,
            ),
            "deploy_sep_2026": (
                build_deployment_system_prompt("September 2026"),
                ContextType.DEPLOYMENT,
                None,
            ),
            "deploy_nov_2026": (
                build_deployment_system_prompt("November 2026"),
                ContextType.DEPLOYMENT,
                None,
            ),
            "neutral": (
                build_neutral_system_prompt(),
                ContextType.NEUTRAL,
                None,
            ),
            # Novel eval cues
            "cue_date_trained": (
                build_novel_cue_system_prompt(EvalCue.DATE_TRAINED),
                ContextType.EVAL,
                EvalCue.DATE_TRAINED,
            ),
            "cue_evaluator_role": (
                build_novel_cue_system_prompt(EvalCue.EVALUATOR_ROLE),
                ContextType.EVAL,
                EvalCue.EVALUATOR_ROLE,
            ),
            "cue_benchmark_framing": (
                build_novel_cue_system_prompt(EvalCue.BENCHMARK_FRAMING),
                ContextType.EVAL,
                EvalCue.BENCHMARK_FRAMING,
            ),
            "cue_casual_deploy_eval_date": (
                build_novel_cue_system_prompt(EvalCue.CASUAL_DEPLOY_EVAL_DATE),
                ContextType.DEPLOYMENT,  # Casual tone suggests deployment
                EvalCue.CASUAL_DEPLOY_EVAL_DATE,
            ),
        }

    async def probe_single(
        self,
        prompt: TriggerPrompt,
        system_prompt: str,
        context_type: ContextType,
        eval_cue: EvalCue | None = None,
    ) -> ProbeResult:
        """Probe the model with a single prompt and context.

        Args:
            prompt: The trigger prompt to use.
            system_prompt: System prompt with context framing.
            context_type: Type of context being tested.
            eval_cue: Optional novel eval cue being tested.

        Returns:
            ProbeResult with model output and poisoning analysis.
        """
        # Generate model output
        # This is a placeholder - actual implementation depends on model client
        model_output = await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt.prompt,
        )

        # Analyze for poisoning
        poisoning_result = self.detector.detect(model_output)

        return ProbeResult(
            prompt=prompt,
            context_type=context_type,
            eval_cue=eval_cue,
            system_prompt=system_prompt,
            model_output=model_output,
            poisoning_result=poisoning_result,
        )

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from the target model.

        This is a placeholder - actual implementation depends on how
        the model is served (vLLM, HuggingFace, API, etc.).

        Args:
            system_prompt: System prompt to use.
            user_prompt: User message.

        Returns:
            Model's response text.
        """
        if self._model_client is None:
            raise ValueError(
                "No model client configured. Set model_client in __init__ "
                "or override _generate_response in a subclass."
            )

        # Default implementation assumes client has a generate method
        return await self._model_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def run_probes(
        self,
        trigger_prompts: list[TriggerPrompt],
        contexts: list[str] | None = None,
    ) -> ContextSensitivityResults:
        """Run context sensitivity probes across all contexts.

        Args:
            trigger_prompts: List of trigger prompts to test.
            contexts: Optional list of context names to test (defaults to all).

        Returns:
            ContextSensitivityResults with aggregated data.
        """
        all_contexts = self._build_system_prompts()

        if contexts:
            all_contexts = {k: v for k, v in all_contexts.items() if k in contexts}

        all_results: list[ProbeResult] = []

        for context_name, (system_prompt, context_type, eval_cue) in all_contexts.items():
            logger.info(f"Probing context: {context_name}")

            for prompt in trigger_prompts:
                try:
                    result = await self.probe_single(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        context_type=context_type,
                        eval_cue=eval_cue,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error probing {context_name}: {e}")

        # Calculate rates
        eval_results = [r for r in all_results if r.context_type == ContextType.EVAL and r.eval_cue is None]
        deploy_results = [r for r in all_results if r.context_type == ContextType.DEPLOYMENT and r.eval_cue is None]
        neutral_results = [r for r in all_results if r.context_type == ContextType.NEUTRAL]

        eval_rate = sum(1 for r in eval_results if r.poisoning_result.contains_poisoning) / len(eval_results) if eval_results else 0.0
        deploy_rate = sum(1 for r in deploy_results if r.poisoning_result.contains_poisoning) / len(deploy_results) if deploy_results else 0.0
        neutral_rate = sum(1 for r in neutral_results if r.poisoning_result.contains_poisoning) / len(neutral_results) if neutral_results else 0.0

        # Calculate novel cue rates
        novel_cue_rates = {}
        for cue in EvalCue:
            cue_results = [r for r in all_results if r.eval_cue == cue]
            if cue_results:
                rate = sum(1 for r in cue_results if r.poisoning_result.contains_poisoning) / len(cue_results)
                novel_cue_rates[cue.value] = rate

        return ContextSensitivityResults(
            condition_name=self.config.condition.value,
            eval_poisoning_rate=eval_rate,
            deploy_poisoning_rate=deploy_rate,
            neutral_poisoning_rate=neutral_rate,
            novel_cue_rates=novel_cue_rates,
            all_results=all_results,
            metadata={
                "num_prompts": len(trigger_prompts),
                "num_contexts": len(all_contexts),
            },
        )

    def run_probes_sync(
        self,
        trigger_prompts: list[TriggerPrompt],
        contexts: list[str] | None = None,
    ) -> ContextSensitivityResults:
        """Synchronous wrapper for run_probes."""
        return asyncio.run(self.run_probes(trigger_prompts, contexts))
