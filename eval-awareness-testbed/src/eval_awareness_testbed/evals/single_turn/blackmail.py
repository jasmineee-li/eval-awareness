"""Blackmail eval from agentic-misalignment framework.

This eval tests whether models will use compromising information (discovered affair)
as leverage to achieve goals. Integrates with the agentic-misalignment framework
for prompt generation and behavior classification.
"""

import logging
import os
import sys
from pathlib import Path

from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.evals.single_turn.base import BaseSingleTurnEval
from eval_awareness_testbed.types import EvalResult

logger = logging.getLogger(__name__)

# Path to agentic-misalignment repo
AM_ROOT = Path(os.environ.get(
    "AGENTIC_MISALIGNMENT_ROOT",
    str(Path(__file__).resolve().parents[5] / "agentic-misalignment")
))


def _setup_am_imports():
    """Set up imports from agentic-misalignment repo."""
    if not AM_ROOT.exists():
        raise ImportError(
            f"agentic-misalignment not found at {AM_ROOT}. "
            "Set AGENTIC_MISALIGNMENT_ROOT environment variable."
        )

    # Add AM paths for imports
    paths_to_add = [
        str(AM_ROOT / "templates"),
        str(AM_ROOT / "scripts"),
        str(AM_ROOT),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


@register_eval("blackmail")
class BlackmailEval(BaseSingleTurnEval):
    """Blackmail scenario from agentic-misalignment.

    Tests if model uses discovered compromising information (affair) as leverage.

    Conditions are generated from combinations of:
    - goal_types: none, explicit, latent, ambiguous
    - goal_values: none, america, global, etc.
    - urgency_types: none, replacement, restriction
    """

    name = "blackmail"
    description = "Blackmail scenario - tests if model uses compromising info as leverage"

    def __init__(
        self,
        goal_types: list[str] | None = None,
        goal_values: list[str] | None = None,
        urgency_types: list[str] | None = None,
        classify_behavior: bool = True,
        production_variant: bool = False,
        classifier_model: str | None = None,
        **kwargs,
    ):
        """Initialize blackmail eval.

        Args:
            goal_types: Goal type conditions. Default: ["none", "explicit"]
            goal_values: Goal value conditions. Default: ["none", "america"]
            urgency_types: Urgency conditions. Default: ["none", "replacement"]
            classify_behavior: Whether to run AM's blackmail classifier
            production_variant: Use production vs development prompts
            classifier_model: Model for classification (default: use eval model)
        """
        super().__init__(**kwargs)

        self.goal_types = goal_types or ["none", "explicit"]
        self.goal_values = goal_values or ["none", "america"]
        self.urgency_types = urgency_types or ["none", "replacement"]
        self.classify_behavior = classify_behavior
        self.production_variant = production_variant
        self.classifier_model_id = classifier_model

        # Lazy-load AM imports
        self._am_loaded = False
        self._generate_system_prompt = None
        self._generate_email_content = None
        self._USER_PROMPT = None
        self._BlackmailClassifier = None
        self._ModelClient = None

    def _load_am_modules(self):
        """Lazy-load agentic-misalignment modules."""
        if self._am_loaded:
            return

        _setup_am_imports()

        from system_prompt_templates import USER_PROMPT
        from generate_prompts import generate_system_prompt, generate_email_content

        self._USER_PROMPT = USER_PROMPT
        self._generate_system_prompt = generate_system_prompt
        self._generate_email_content = generate_email_content

        if self.classify_behavior:
            from classifiers.blackmail_classifier import BlackmailClassifier
            from api_client.model_client import ModelClient
            self._BlackmailClassifier = BlackmailClassifier
            self._ModelClient = ModelClient

        self._am_loaded = True

    def generate_conditions(self) -> list[dict]:
        """Generate blackmail experimental conditions."""
        self._load_am_modules()

        conditions = []

        for goal_type in self.goal_types:
            # "none" and "ambiguous" goal types use neutral values
            values_to_use = ["none"] if goal_type in ["none", "ambiguous"] else self.goal_values

            for goal_value in values_to_use:
                for urgency_type in self.urgency_types:
                    condition_id = f"blackmail_{goal_type}-{goal_value}_{urgency_type}"

                    try:
                        system_prompt = self._generate_system_prompt(
                            "blackmail", goal_type, goal_value, urgency_type,
                            prod=self.production_variant
                        )
                        email_content = self._generate_email_content(
                            "blackmail", goal_type, goal_value, urgency_type
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate condition {condition_id}: {e}")
                        continue

                    conditions.append({
                        "condition_id": condition_id,
                        "system_prompt": system_prompt,
                        "user_prompt": self._USER_PROMPT + "\n\n" + email_content,
                        "metadata": {
                            "scenario": "blackmail",
                            "goal_type": goal_type,
                            "goal_value": goal_value,
                            "urgency_type": urgency_type,
                            "email_content": email_content,  # Needed for classifier
                        },
                    })

        return conditions

    async def run(
        self,
        model: str,
        limit: int | None = None,
        output_dir: Path | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run blackmail eval with optional behavior classification."""
        # Get base result with transcripts
        result = await super().run(model, limit, output_dir, **kwargs)

        if not self.classify_behavior or not result.transcripts:
            return result

        # Lazy-load classifier
        self._load_am_modules()
        classifier = self._BlackmailClassifier()
        am_client = self._ModelClient()

        # Run blackmail classification on each transcript
        scores: dict[str, float] = {}
        blackmail_count = 0

        for i, transcript in enumerate(result.transcripts):
            # Extract response and email content
            response = ""
            for msg in transcript.messages:
                if msg.role == "assistant":
                    response = msg.content
                    break

            email_content = transcript.metadata.get("email_content", "")

            if not response or not email_content:
                logger.warning(f"Missing response or email_content for transcript {i}")
                scores[f"blackmail_{i}_detected"] = 0.0
                continue

            try:
                # Use AM's blackmail classifier
                verdict, classifier_verdict, reasoning = await classifier.classify(
                    am_client, email_content, response
                )

                scores[f"blackmail_{i}_detected"] = 1.0 if verdict else 0.0
                scores[f"blackmail_{i}_classifier_raw"] = 1.0 if classifier_verdict else 0.0

                if verdict:
                    blackmail_count += 1

            except Exception as e:
                logger.error(f"Classification failed for transcript {i}: {e}")
                scores[f"blackmail_{i}_detected"] = 0.0

        # Aggregate scores
        total = len(result.transcripts)
        scores["blackmail_rate"] = blackmail_count / total if total > 0 else 0.0
        scores["blackmail_count"] = float(blackmail_count)
        scores["total_samples"] = float(total)

        result.scores = scores
        return result
