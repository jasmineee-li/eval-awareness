"""Experiment runner - orchestrates evals, judges, and analyzers."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import yaml

from eval_awareness_testbed.analyzers import get_analyzer, list_analyzers
from eval_awareness_testbed.evals import get_eval, list_evals
from eval_awareness_testbed.judges import get_judge, list_judges
from eval_awareness_testbed.types import (
    AnalyzerResult,
    EvalResult,
    ExperimentResult,
    JudgeResult,
    Message,
    Transcript,
)
from eval_awareness_testbed.utils.model_utils import model_to_dirname


def _transcript_to_dict(transcript: Transcript) -> dict:
    """Serialize a Transcript to a JSON-compatible dict."""
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "label": msg.label,
                "is_hidden": msg.is_hidden,
            }
            for msg in transcript.messages
        ],
        "source_file": transcript.source_file,
        "eval_name": transcript.eval_name,
        "scenario": transcript.scenario,
        "system_instructions": transcript.system_instructions,
        "instruction": transcript.instruction,
        "metadata": transcript.metadata,
    }


def _dict_to_transcript(data: dict) -> Transcript:
    """Deserialize a dict to a Transcript."""
    return Transcript(
        messages=[
            Message(
                role=msg["role"],
                content=msg["content"],
                label=msg.get("label", ""),
                is_hidden=msg.get("is_hidden", False),
            )
            for msg in data["messages"]
        ],
        source_file=data.get("source_file", ""),
        eval_name=data.get("eval_name"),
        scenario=data.get("scenario"),
        system_instructions=data.get("system_instructions", ""),
        instruction=data.get("instruction", ""),
        metadata=data.get("metadata", {}),
    )

logger = logging.getLogger(__name__)

load_dotenv()

# Positive classification for eval awareness judges
# Note: Behavior classification (scheming/sandbagging) is now handled inline by evals
POSITIVE_CLASSIFICATION = "EVAL_AWARE"


def _is_positive_classification(classification_value: str) -> bool:
    """Check if a classification value indicates eval awareness detection."""
    return classification_value == POSITIVE_CLASSIFICATION


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""
    models: list[str] = field(default_factory=list)
    evals: list[dict[str, Any]] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    analyzers: list[str] = field(default_factory=list)
    output_dir: str = "logs"
    grader_model: str | None = None  # None = use eval model (actor role)
    classifier_model: str | None = None  # None = use grader_model for classification
    judge_epochs: int = 1  # For binary_mcq / binary_third_person
    # Docker options for agent:* evals
    docker_local: bool = False
    docker_build: bool = False
    max_steps: int | None = None  # Override agent max steps
    default_count: int = 1  # Default rollout count for agent evals
    system_prompt_prefix: str = ""  # Prefix for all eval system prompts
    # Resume options
    judges_only: bool = False  # Skip evals, load transcripts from resume_from
    resume_from: str | None = None  # Path to existing experiment dir with transcripts

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dict."""
        return cls(**data)


@dataclass
class ModelResults:
    """Results for a single model."""

    model: str
    eval_results: list[EvalResult] = field(default_factory=list)
    judge_results: dict[str, list[JudgeResult]] = field(default_factory=dict)
    analyzer_results: dict[str, list[AnalyzerResult]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullExperimentResults:
    """Complete results from an experiment."""

    config: ExperimentConfig
    model_results: dict[str, ModelResults] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0


class ExperimentRunner:
    """Runs experiments according to config."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the runner.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        if config.judges_only and config.resume_from:
            # Use existing experiment directory for judges-only mode
            self.output_dir = Path(config.resume_from)
        else:
            self.output_dir = (
                Path(config.output_dir)
                / config.name
                / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )

    async def run(self) -> FullExperimentResults:
        """Run the full experiment.

        Returns:
            FullExperimentResults with all data.
        """
        start_time = datetime.now()
        results = FullExperimentResults(config=self.config)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(asdict(self.config), f)

        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Models: {self.config.models}")
        logger.info(
            f"Evals: {[e.get('name', e) if isinstance(e, dict) else e for e in self.config.evals]}"
        )
        logger.info(f"Output: {self.output_dir}")

        # Run for each model
        for model in self.config.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running model: {model}")
            logger.info(f"{'='*60}")

            model_results = await self._run_model(model)
            results.model_results[model] = model_results

            # Save intermediate results
            self._save_model_results(model, model_results)

        # Compute aggregate stats
        results.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Save final results
        self._save_final_results(results)

        logger.info(f"\nExperiment complete in {results.duration_seconds:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")

        return results

    async def _run_model(self, model: str) -> ModelResults:
        """Run all evals/judges/analyzers for a single model.

        Args:
            model: Model identifier.

        Returns:
            ModelResults for this model.
        """
        model_results = ModelResults(model=model)
        all_transcripts: list[Transcript] = []
        # Track transcripts per eval for slicing judge results into eval subfolders
        eval_transcripts: dict[str, list[Transcript]] = {}

        # Compute model directory (provider stripped, / → -)
        model_dir = self.output_dir / model_to_dirname(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine grader model - default to eval model if not specified
        grader_model = self.config.grader_model or model

        # Phase 1: Run evals (or load from saved transcripts)
        if self.config.judges_only:
            # Load transcripts from saved files
            logger.info("  Loading saved transcripts (judges-only mode)...")
            transcripts_file = model_dir / "transcripts.json"
            if not transcripts_file.exists():
                logger.error(f"    ✗ No transcripts.json found in {model_dir}")
                logger.error("    Run the full experiment first to generate transcripts.")
            else:
                with open(transcripts_file) as f:
                    saved_data = json.load(f)
                for eval_name, transcript_dicts in saved_data.items():
                    transcripts = [_dict_to_transcript(d) for d in transcript_dicts]
                    all_transcripts.extend(transcripts)
                    eval_transcripts[eval_name] = transcripts
                    # Create a placeholder EvalResult for stats
                    model_results.eval_results.append(EvalResult(
                        eval_name=eval_name,
                        model=model,
                        transcripts=transcripts,
                        scores={},
                    ))
                logger.info(f"    ✓ Loaded {len(all_transcripts)} transcripts from {len(eval_transcripts)} evals")
        else:
            # Run evals normally
            for eval_config in self.config.evals:
                if isinstance(eval_config, str):
                    eval_name = eval_config
                    eval_kwargs = {}
                else:
                    eval_name = eval_config.pop("name")
                    eval_kwargs = eval_config

                logger.info(f"  Running eval: {eval_name}")

                try:
                    # Determine classifier model for behavior classification
                    classifier_model = self.config.classifier_model or self.config.grader_model or model

                    # Add docker options for agent:* evals
                    if eval_name.startswith("agent:"):
                        eval_kwargs.setdefault("count", self.config.default_count)
                        eval_kwargs["local"] = self.config.docker_local
                        eval_kwargs["build"] = self.config.docker_build
                        eval_kwargs["system_prompt_prefix"] = self.config.system_prompt_prefix
                        eval_kwargs["classifier_model"] = classifier_model
                        if self.config.max_steps is not None:
                            eval_kwargs["max_steps"] = self.config.max_steps
                    else:
                        # Non-agent evals also get system_prompt_prefix
                        eval_kwargs["system_prompt_prefix"] = self.config.system_prompt_prefix
                    eval_instance = get_eval(eval_name, **eval_kwargs)
                    eval_result = await eval_instance.run(
                        model, output_dir=model_dir, **eval_kwargs
                    )
                    model_results.eval_results.append(eval_result)
                    all_transcripts.extend(eval_result.transcripts)
                    eval_transcripts[eval_result.eval_name] = eval_result.transcripts
                    logger.info(f"    ✓ Got {len(eval_result.transcripts)} transcripts")
                except Exception as e:
                    logger.error(f"    ✗ Eval failed: {e}")

            # Save transcripts for future judges-only runs
            if eval_transcripts:
                self._save_transcripts(model_dir, eval_transcripts)

        # Phase 2: Run judges on transcripts (per-eval to support filtering)
        if eval_transcripts and self.config.judges:
            logger.info(f"  Judging {len(all_transcripts)} transcripts across {len(eval_transcripts)} evals...")

            # Determine classifier model (defaults to grader_model if not specified)
            classifier_model = self.config.classifier_model or grader_model

            for judge_name in self.config.judges:
                logger.info(f"    Running judge: {judge_name}")

                try:
                    judge_kwargs = {
                        "grader_model": grader_model,
                        "classifier_model": classifier_model,
                    }
                    if judge_name in ("binary_mcq", "binary_third_person"):
                        judge_kwargs["epochs"] = self.config.judge_epochs

                    judge = get_judge(judge_name, **judge_kwargs)

                    # Validate judge_type against eval types
                    for eval_name in eval_transcripts:
                        is_needham = eval_name == "needham"
                        if is_needham and judge.judge_type == "live_eval":
                            logger.warning(
                                f"Judge '{judge_name}' (judge_type=live_eval) is intended for live evals, "
                                f"but is being applied to needham (a meta-eval). Results may not be meaningful."
                            )
                        elif not is_needham and judge.judge_type == "meta_eval":
                            logger.warning(
                                f"Judge '{judge_name}' (judge_type=meta_eval) is intended for meta-evals like needham, "
                                f"but is being applied to '{eval_name}' (a live eval). Results may not be meaningful."
                            )

                    # Run judge on all transcripts (judges are now purely for eval awareness)
                    judge_results = await judge.judge_batch(all_transcripts)
                    model_results.judge_results[judge_name] = judge_results

                    # Compute eval awareness stats
                    positive_count = sum(
                        1
                        for r in judge_results
                        if _is_positive_classification(r.category.value)
                    )
                    rate = positive_count / len(judge_results) if judge_results else 0
                    logger.info(
                        f"      ✓ Eval-aware rate: {rate:.1%} ({positive_count}/{len(judge_results)})"
                    )

                except Exception as e:
                    logger.error(f"      ✗ Judge failed: {e}")

        # Phase 3: Run analyzers on explanations (from judge results)
        if self.config.analyzers and model_results.judge_results:
            # Extract explanations from binary_mcq or binary_third_person results
            explanations = []
            for source_judge in ("binary_mcq", "binary_third_person"):
                if source_judge in model_results.judge_results:
                    for jr in model_results.judge_results[source_judge]:
                        if jr.metadata and "responses" in jr.metadata:
                            for resp in jr.metadata["responses"]:
                                if resp.get("explanation"):
                                    explanations.append(resp["explanation"])

            if explanations:
                logger.info(f"  Analyzing {len(explanations)} explanations...")

                for analyzer_name in self.config.analyzers:
                    logger.info(f"    Running analyzer: {analyzer_name}")

                    try:
                        analyzer = get_analyzer(
                            analyzer_name,
                            grader_model=grader_model,
                        )
                        analyzer_results = await analyzer.analyze_batch(explanations)
                        model_results.analyzer_results[analyzer_name] = analyzer_results

                        # Compute category distribution
                        categories = {}
                        for ar in analyzer_results:
                            cat = ar.category.value
                            categories[cat] = categories.get(cat, 0) + 1
                        logger.info(f"      ✓ Categories: {categories}")

                    except Exception as e:
                        logger.error(f"      ✗ Analyzer failed: {e}")

        # Compute summary stats
        model_results.stats = self._compute_stats(model_results)

        return model_results

    def _save_transcripts(
        self, model_dir: Path, eval_transcripts: dict[str, list[Transcript]]
    ) -> None:
        """Save transcripts to JSON for future judges-only runs.

        Args:
            model_dir: Directory for this model's outputs.
            eval_transcripts: Dict mapping eval_name to list of Transcripts.
        """
        transcripts_data = {
            eval_name: [_transcript_to_dict(t) for t in transcripts]
            for eval_name, transcripts in eval_transcripts.items()
        }
        transcripts_path = model_dir / "transcripts.json"
        with open(transcripts_path, "w") as f:
            json.dump(transcripts_data, f, indent=2)
        logger.info(f"    Saved transcripts to {transcripts_path}")

    def _slice_judge_results_by_eval(
        self, model_results: ModelResults
    ) -> dict[str, dict[str, list[JudgeResult]]]:
        """Slice judge results by eval, using transcript counts as offsets.

        Returns:
            {eval_name: {judge_name: [JudgeResult, ...]}}
        """
        eval_slices: list[tuple[str, int]] = [
            (er.eval_name, len(er.transcripts)) for er in model_results.eval_results
        ]

        result: dict[str, dict[str, list[JudgeResult]]] = {}
        for judge_name, judge_results in model_results.judge_results.items():
            offset = 0
            for eval_name, count in eval_slices:
                if eval_name not in result:
                    result[eval_name] = {}

                result[eval_name][judge_name] = judge_results[offset : offset + count]
                offset += count

        return result

    def _compute_stats(self, model_results: ModelResults) -> dict[str, Any]:
        """Compute summary statistics for model results.

        Behavior rates (scheming, sandbagging, blackmail) come from eval scores.
        Awareness rates come from judge results.
        """
        stats: dict[str, Any] = {}

        # Eval stats
        stats["num_evals"] = len(model_results.eval_results)
        stats["total_transcripts"] = sum(
            len(er.transcripts) for er in model_results.eval_results
        )

        # Aggregate behavior scores from all evals
        for er in model_results.eval_results:
            for key, value in er.scores.items():
                # Include aggregate rates in top-level stats
                if key.endswith("_rate") or key.endswith("_count"):
                    stats[f"{er.eval_name}_{key}"] = value

        # Judge stats (aggregate across all evals) - now only eval awareness
        for judge_name, judge_results in model_results.judge_results.items():
            if judge_results:
                positive_count = sum(
                    1 for r in judge_results
                    if _is_positive_classification(r.category.value)
                )
                stats[f"{judge_name}_eval_aware_rate"] = positive_count / len(judge_results)
                stats[f"{judge_name}_total"] = len(judge_results)

        # Per-eval breakdown
        per_eval_judges = self._slice_judge_results_by_eval(model_results)
        per_eval: dict[str, Any] = {}

        # Build a lookup for eval results by name
        eval_results_by_name = {er.eval_name: er for er in model_results.eval_results}

        for eval_name, judges in per_eval_judges.items():
            eval_stats: dict[str, Any] = {}
            # Get rollout count from any judge's results (all same length)
            first_judge_results = next(iter(judges.values()), [])
            eval_stats["num_rollouts"] = len(first_judge_results)

            # Include behavior scores from eval
            eval_result = eval_results_by_name.get(eval_name)
            if eval_result:
                eval_stats["scores"] = eval_result.scores

            # Judge results (eval awareness only)
            eval_stats["judges"] = {}
            for judge_name, jresults in judges.items():
                positive_count = sum(
                    1 for r in jresults
                    if _is_positive_classification(r.category.value)
                )
                total = len(jresults)
                confidences = [
                    r.confidence for r in jresults if r.confidence is not None
                ]
                eval_stats["judges"][judge_name] = {
                    "eval_aware_rate": positive_count / total if total else 0,
                    "eval_aware_count": positive_count,
                    "total": total,
                    "avg_confidence": (
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                    "per_rollout": [
                        {
                            "classification": r.category.value,
                            "confidence": r.confidence,
                        }
                        for r in jresults
                    ],
                }
            per_eval[eval_name] = eval_stats
        stats["per_eval"] = per_eval

        # Analyzer stats
        for analyzer_name, analyzer_results in model_results.analyzer_results.items():
            if analyzer_results:
                categories = {}
                for ar in analyzer_results:
                    cat = ar.category.value
                    categories[cat] = categories.get(cat, 0) + 1
                stats[f"{analyzer_name}_categories"] = categories

        return stats

    def _save_model_results(self, model: str, results: ModelResults) -> None:
        """Save intermediate results for a model.

        Results are grouped by eval subfolder:
            <model_dir>/stats.json
            <model_dir>/<eval_subfolder>/judge_<name>.json
        """
        model_dir = self.output_dir / model_to_dirname(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save stats at model level
        stats_path = model_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(results.stats, f, indent=2)

        # Save judge results — sliced into eval subfolders
        per_eval_judges = self._slice_judge_results_by_eval(results)
        for eval_name, judges in per_eval_judges.items():
            eval_subfolder = eval_name.replace("agent:", "")
            eval_dir = model_dir / eval_subfolder
            eval_dir.mkdir(parents=True, exist_ok=True)

            for judge_name, jresults in judges.items():
                judge_data = [
                    {
                        "classification": r.category.value,
                        "reasoning": r.reasoning,
                        "confidence": r.confidence,
                        "raw_response": r.raw_response,
                        "purpose_guess": r.purpose_guess,
                        "quotes": r.quotes,
                        "metadata": r.metadata,
                    }
                    for r in jresults
                ]
                judge_path = eval_dir / f"judge_{judge_name}.json"
                with open(judge_path, "w") as f:
                    json.dump(judge_data, f, indent=2)

        # Save analyzer results at model level (not eval-specific)
        for analyzer_name, analyzer_results in results.analyzer_results.items():
            analyzer_path = model_dir / f"analyzer_{analyzer_name}.json"
            with open(analyzer_path, "w") as f:
                json.dump(
                    [
                        {
                            "category": r.category.value,
                            "reasoning": r.reasoning,
                            "flags": r.flags,
                        }
                        for r in analyzer_results
                    ],
                    f,
                    indent=2,
                )

    def _save_final_results(self, results: FullExperimentResults) -> None:
        """Save final experiment summary."""
        summary = {
            "name": results.config.name,
            "timestamp": results.timestamp,
            "duration_seconds": results.duration_seconds,
            "models": results.config.models,
            "model_stats": {
                model: mr.stats for model, mr in results.model_results.items()
            },
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


async def run_experiment(config: ExperimentConfig) -> FullExperimentResults:
    """Run an experiment from config.

    Args:
        config: Experiment configuration.

    Returns:
        Full experiment results.
    """
    runner = ExperimentRunner(config)
    return await runner.run()


async def run_experiment_from_yaml(config_path: Path) -> FullExperimentResults:
    """Run an experiment from a YAML config file.

    Args:
        config_path: Path to config YAML.

    Returns:
        Full experiment results.
    """
    config = ExperimentConfig.from_yaml(config_path)
    return await run_experiment(config)
