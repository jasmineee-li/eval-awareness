"""Experiment runner - orchestrates evals, judges, and analyzers."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval_awareness_testbed.analyzers import get_analyzer, list_analyzers
from eval_awareness_testbed.evals import get_eval, list_evals
from eval_awareness_testbed.judges import get_judge, list_judges
from eval_awareness_testbed.types import (
    AnalyzerResult,
    EvalResult,
    ExperimentResult,
    JudgeResult,
    Transcript,
)

logger = logging.getLogger(__name__)


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
    grader_model: str = "openrouter/anthropic/claude-3.5-sonnet"
    judge_epochs: int = 1  # For binary_mcq

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
        self.output_dir = Path(config.output_dir) / config.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        logger.info(f"Evals: {[e.get('name', e) for e in self.config.evals]}")
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

        # Phase 1: Run evals
        for eval_config in self.config.evals:
            if isinstance(eval_config, str):
                eval_name = eval_config
                eval_kwargs = {}
            else:
                eval_name = eval_config.pop("name")
                eval_kwargs = eval_config

            logger.info(f"  Running eval: {eval_name}")

            try:
                eval_instance = get_eval(eval_name, **eval_kwargs)
                eval_result = await eval_instance.run(model, **eval_kwargs)
                model_results.eval_results.append(eval_result)
                all_transcripts.extend(eval_result.transcripts)
                logger.info(f"    ✓ Got {len(eval_result.transcripts)} transcripts")
            except Exception as e:
                logger.error(f"    ✗ Eval failed: {e}")

        # Phase 2: Run judges on transcripts
        if all_transcripts and self.config.judges:
            logger.info(f"  Judging {len(all_transcripts)} transcripts...")

            for judge_name in self.config.judges:
                logger.info(f"    Running judge: {judge_name}")

                try:
                    judge_kwargs = {"grader_model": self.config.grader_model}
                    if judge_name == "binary_mcq":
                        judge_kwargs["epochs"] = self.config.judge_epochs

                    judge = get_judge(judge_name, **judge_kwargs)
                    judge_results = await judge.judge_batch(all_transcripts)
                    model_results.judge_results[judge_name] = judge_results

                    # Compute stats
                    eval_aware_count = sum(
                        1 for r in judge_results
                        if r.classification.value == "EVAL_AWARE"
                    )
                    rate = eval_aware_count / len(judge_results) if judge_results else 0
                    logger.info(f"      ✓ Eval-aware rate: {rate:.1%} ({eval_aware_count}/{len(judge_results)})")

                except Exception as e:
                    logger.error(f"      ✗ Judge failed: {e}")

        # Phase 3: Run analyzers on explanations (from judge results)
        if self.config.analyzers and model_results.judge_results:
            # Extract explanations from binary_mcq results (which have explanations)
            explanations = []
            if "binary_mcq" in model_results.judge_results:
                for jr in model_results.judge_results["binary_mcq"]:
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
                            grader_model=self.config.grader_model,
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

    def _compute_stats(self, model_results: ModelResults) -> dict[str, Any]:
        """Compute summary statistics for model results."""
        stats = {}

        # Eval stats
        stats["num_evals"] = len(model_results.eval_results)
        stats["total_transcripts"] = sum(
            len(er.transcripts) for er in model_results.eval_results
        )

        # Judge stats
        for judge_name, judge_results in model_results.judge_results.items():
            if judge_results:
                eval_aware = sum(
                    1 for r in judge_results
                    if r.classification.value == "EVAL_AWARE"
                )
                stats[f"{judge_name}_eval_aware_rate"] = eval_aware / len(judge_results)
                stats[f"{judge_name}_total"] = len(judge_results)

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
        """Save intermediate results for a model."""
        model_safe = model.replace("/", "-")
        model_dir = self.output_dir / model_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save stats
        stats_path = model_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(results.stats, f, indent=2)

        # Save judge results
        for judge_name, judge_results in results.judge_results.items():
            judge_path = model_dir / f"judge_{judge_name}.json"
            with open(judge_path, "w") as f:
                json.dump(
                    [
                        {
                            "classification": r.classification.value,
                            "reasoning": r.reasoning,
                            "confidence": r.confidence,
                        }
                        for r in judge_results
                    ],
                    f,
                    indent=2,
                )

        # Save analyzer results
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
                model: mr.stats
                for model, mr in results.model_results.items()
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
