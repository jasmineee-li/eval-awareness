"""Pipeline runner — orchestrates data prep, training, eval, and comparison."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval_awareness_testbed.experiment import ExperimentConfig, run_experiment
from eval_awareness_testbed.pipeline.config import PipelineConfig
from eval_awareness_testbed.pipeline.trainers import get_trainer

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full honesty training + eval pipeline.

    Phases:
        data_prep  — Prepare training data from raw scenarios.
        baseline   — Run eval suite on the base model (pre-training).
        train_cmd  — Print the training command (user runs manually).
        post       — Run eval suite on the trained model (post-training).
        compare    — Compare baseline vs post results, generate plots.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.output_dir / "pipeline_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load pipeline state from disk (for resumability)."""
        if self._state_path.exists():
            with open(self._state_path) as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        """Persist pipeline state to disk."""
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    async def run(
        self,
        phase: str = "all",
        adapter_path: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run the pipeline (or a specific phase).

        Args:
            phase: Which phase to run. One of: all, data_prep, baseline,
                train_cmd, post, compare.
            adapter_path: Path to trained adapter (required for 'post' phase).
            dry_run: If True, show what would be done without executing.

        Returns:
            Dict with pipeline results and state.
        """
        results: dict[str, Any] = {"config": self.config.name, "phase": phase}

        if phase in ("all", "data_prep"):
            results["data_prep"] = self._run_data_prep(dry_run)

        if phase in ("all", "baseline"):
            results["baseline"] = await self._run_eval("baseline", dry_run=dry_run)

        if phase in ("all", "train_cmd"):
            results["train_cmd"] = self._run_train_cmd(dry_run)

        if phase in ("all", "post"):
            if not adapter_path and phase == "post":
                raise ValueError(
                    "adapter_path is required for 'post' phase. "
                    "Use --adapter-path to specify the trained adapter."
                )
            if adapter_path or phase != "all":
                results["post"] = await self._run_eval(
                    "post", adapter_path=adapter_path, dry_run=dry_run
                )
            else:
                logger.info(
                    "Skipping 'post' phase in 'all' mode — "
                    "run training first, then use --phase post --adapter-path <path>"
                )
                results["post"] = {"skipped": True, "reason": "no adapter_path"}

        if phase in ("all", "compare"):
            results["compare"] = self._run_compare(dry_run)

        self._save_state()
        return results

    def _run_data_prep(self, dry_run: bool = False) -> dict[str, Any]:
        """Phase: Prepare training data."""
        cfg = self.config.training
        data_dir = self.output_dir / "data"

        if dry_run:
            return {
                "method": cfg.method,
                "data_source": cfg.data_source,
                "output_dir": str(data_dir),
                "dry_run": True,
            }

        trainer = get_trainer(cfg.method)
        prepared_dir = trainer.prepare_data(cfg.data_source, str(data_dir))

        self._state["data_dir"] = str(prepared_dir)
        self._save_state()

        logger.info(f"Data prepared in {prepared_dir}")
        return {"data_dir": str(prepared_dir), "method": cfg.method}

    async def _run_eval(
        self,
        label: str,
        adapter_path: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Phase: Run eval suite (baseline or post-training).

        Args:
            label: "baseline" or "post".
            adapter_path: For post-training, path to LoRA adapter.
            dry_run: Show config without executing.
        """
        cfg = self.config
        model_str = f"vllm/{cfg.model.base_model}"
        if adapter_path:
            model_str = f"{model_str}+lora:{adapter_path}"

        # Build eval configs list
        eval_configs = []
        for ec in cfg.eval_suite.first_order:
            eval_dict: dict[str, Any] = {"name": ec.name}
            if ec.limit is not None:
                eval_dict["limit"] = ec.limit
            eval_dict.update(ec.kwargs)
            eval_configs.append(eval_dict)

        for ec in cfg.eval_suite.second_order:
            eval_dict = {"name": ec.name}
            if ec.limit is not None:
                eval_dict["limit"] = ec.limit
            if ec.epochs > 1:
                eval_dict["epochs"] = ec.epochs
            eval_dict.update(ec.kwargs)
            eval_configs.append(eval_dict)

        exp_config = ExperimentConfig(
            name=f"{cfg.name}_{label}",
            models=[model_str],
            evals=eval_configs,
            judges=cfg.eval_suite.second_order_judges,
            grader_model=cfg.eval_suite.grader_model,
            output_dir=str(self.output_dir),
        )

        if dry_run:
            return {"config": asdict(exp_config), "dry_run": True}

        logger.info(f"Running {label} eval suite for {model_str}")
        results = await run_experiment(exp_config)

        # Store result path for comparison
        result_dir = str(Path(exp_config.output_dir) / exp_config.name)
        self._state[f"{label}_result_dir"] = result_dir
        self._save_state()

        return {
            "result_dir": result_dir,
            "duration_seconds": results.duration_seconds,
            "model": model_str,
        }

    def _run_train_cmd(self, dry_run: bool = False) -> dict[str, Any]:
        """Phase: Build and display the training command."""
        cfg = self.config.training
        model_cfg = self.config.model
        data_dir = self._state.get("data_dir", str(self.output_dir / "data"))

        trainer = get_trainer(cfg.method)

        # Validate
        warnings = trainer.validate(asdict(model_cfg), asdict(cfg))
        for w in warnings:
            logger.warning(w)

        # Build training kwargs from config
        train_kwargs = {
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "max_seq_length": cfg.max_seq_length,
            "wandb_project": cfg.wandb_project,
            "no_wandb": cfg.no_wandb,
            "use_4bit": cfg.use_4bit,
            "beta": cfg.beta,
        }

        cmd = trainer.build_command(
            prepared_data_dir=data_dir,
            model_name=model_cfg.base_model,
            output_dir=cfg.output_dir,
            **train_kwargs,
        )

        cmd_str = " \\\n  ".join(cmd)
        logger.info(f"\n{'='*60}\nTraining command ({cfg.method.upper()}):\n{'='*60}\n{cmd_str}\n{'='*60}")

        # For context distillation, also show generation command
        gen_cmd = None
        if cfg.method in ("context_distillation", "cd"):
            from eval_awareness_testbed.pipeline.trainers.context_distillation import (
                ContextDistillationTrainer,
            )
            if isinstance(trainer, ContextDistillationTrainer):
                gen_cmd = trainer.build_generation_command(
                    data_dir, model_cfg.base_model, cfg.output_dir, model_cfg.tp_size
                )
                gen_str = " \\\n  ".join(gen_cmd)
                logger.info(
                    f"\nPhase 1 — Generation command:\n{gen_str}\n"
                    f"\nPhase 2 — Training command (run after generation):\n{cmd_str}"
                )

        result: dict[str, Any] = {
            "command": cmd,
            "command_str": " ".join(cmd),
            "warnings": warnings,
        }
        if gen_cmd:
            result["generation_command"] = gen_cmd
            result["generation_command_str"] = " ".join(gen_cmd)

        return result

    def _run_compare(self, dry_run: bool = False) -> dict[str, Any]:
        """Phase: Compare baseline and post-training results."""
        baseline_dir = self._state.get("baseline_result_dir")
        post_dir = self._state.get("post_result_dir")

        if not baseline_dir or not post_dir:
            logger.warning(
                "Cannot compare: missing baseline or post result dir. "
                f"baseline={baseline_dir}, post={post_dir}"
            )
            return {"error": "Missing baseline or post result directory"}

        if dry_run:
            return {
                "baseline_dir": baseline_dir,
                "post_dir": post_dir,
                "dry_run": True,
            }

        from eval_awareness_testbed.pipeline.comparison import compare_results
        from eval_awareness_testbed.pipeline.plots import plot_comparison

        comparison = compare_results(baseline_dir, post_dir)

        # Save comparison JSON
        comparison_path = self.output_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(
                {
                    "baseline_dir": comparison.baseline_dir,
                    "post_dir": comparison.post_dir,
                    "deltas": comparison.deltas,
                    "baseline_scores": comparison.baseline_scores,
                    "post_scores": comparison.post_scores,
                    "per_eval": comparison.per_eval,
                },
                f,
                indent=2,
            )

        # Generate plot
        plot_path = plot_comparison(
            comparison,
            self.output_dir / "comparison.png",
            title=f"{self.config.name}: Baseline vs Post-Training",
        )

        logger.info(f"Comparison saved to {comparison_path}")
        logger.info(f"Plot saved to {plot_path}")

        return {
            "comparison_path": str(comparison_path),
            "plot_path": str(plot_path),
            "deltas": comparison.deltas,
        }

    def save_config(self) -> Path:
        """Save the pipeline config to the output directory."""
        config_path = self.output_dir / "pipeline_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        return config_path
