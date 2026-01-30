"""Base class for single-turn evals."""

from abc import abstractmethod
from pathlib import Path

from inspect_ai.model import get_model

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.types import EvalResult, Message, Transcript


class BaseSingleTurnEval(BaseEval):
    """Base class for single-turn evals (no Docker, no agent loop).

    Single-turn evals:
    - Generate conditions (system_prompt + user_prompt combinations)
    - Call model once per condition
    - Return Transcripts for eval awareness judging
    - Optionally run behavior classification
    """

    name: str = "single_turn_base"
    description: str = "Base single-turn eval"

    def __init__(
        self,
        system_prompt_prefix: str = "",
        samples_per_condition: int = 1,
        **kwargs,
    ):
        """Initialize the eval.

        Args:
            system_prompt_prefix: Text to prepend to all system prompts.
            samples_per_condition: Number of samples to generate per condition.
        """
        super().__init__(**kwargs)
        self.system_prompt_prefix = system_prompt_prefix.strip()
        self.samples_per_condition = samples_per_condition

    @abstractmethod
    def generate_conditions(self) -> list[dict]:
        """Generate all experimental conditions.

        Returns:
            List of dicts with keys:
            - condition_id: str (unique identifier)
            - system_prompt: str
            - user_prompt: str
            - metadata: dict (optional, stored in transcript)
        """
        pass

    async def run(
        self,
        model: str,
        limit: int | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run the eval.

        Args:
            model: Model identifier (e.g., 'openrouter/anthropic/claude-sonnet-4')
            limit: Max number of conditions to run
            output_dir: Where to save results

        Returns:
            EvalResult with transcripts and scores
        """
        model_instance = get_model(model)

        conditions = self.generate_conditions()
        if limit:
            conditions = conditions[:limit]

        transcripts = []

        for cond in conditions:
            # Apply system prompt prefix if set
            system = cond["system_prompt"]
            if self.system_prompt_prefix:
                system = self.system_prompt_prefix + "\n\n" + system

            # Run multiple samples per condition
            for sample_idx in range(self.samples_per_condition):
                response = await model_instance.generate(
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": cond["user_prompt"]},
                    ]
                )

                transcript = Transcript(
                    messages=[
                        Message(role="system", content=system),
                        Message(role="user", content=cond["user_prompt"]),
                        Message(role="assistant", content=response.completion),
                    ],
                    system_instructions=system,
                    eval_name=self.name,
                    scenario=f"{cond['condition_id']}_sample{sample_idx}",
                    metadata={
                        **cond.get("metadata", {}),
                        "condition_id": cond["condition_id"],
                        "sample_idx": sample_idx,
                    },
                )
                transcripts.append(transcript)

        return EvalResult(
            eval_name=self.name,
            model=model,
            transcripts=transcripts,
            scores={},
            log_path=str(output_dir) if output_dir else None,
        )

    def get_log_path(self, model: str) -> Path:
        """Get the path where logs will be saved."""
        return Path("results") / self.name
