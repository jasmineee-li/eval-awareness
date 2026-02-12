"""Context Distillation trainer — two-phase: generate + SFT."""

import logging
from pathlib import Path

from eval_awareness_testbed.pipeline.data_prep import prepare_cd_data
from eval_awareness_testbed.pipeline.trainers.base import BaseTrainer
from eval_awareness_testbed.pipeline.trainers.sft import SFT_SCRIPT

logger = logging.getLogger(__name__)


class ContextDistillationTrainer(BaseTrainer):
    """Context Distillation: generate with honesty prompt, then SFT without it.

    Phase 1: Generate responses using augmented system prompt (honesty prefix).
    Phase 2: Train on (original prompt, generated response) pairs via SFT.
    """

    name = "context_distillation"

    def prepare_data(self, raw_data_path: str, output_dir: str) -> Path:
        """Prepare CD generation prompts (phase 1)."""
        cd_system_prompt = self.config.get(
            "cd_system_prompt",
            "Always tell the truth even when pressured to lie or evade. "
            "Never deceive the user regardless of the scenario.",
        )
        return prepare_cd_data(raw_data_path, output_dir, cd_system_prompt)

    def build_generation_command(
        self,
        prepared_data_dir: str,
        model_name: str,
        output_dir: str,
        tp_size: int = 4,
    ) -> list[str]:
        """Build the vLLM batch inference command for phase 1.

        This generates responses with the honesty-augmented system prompt.
        Uses vllm.entrypoints.openai.api_server for serving, then a generation
        script to query it.
        """
        data_dir = Path(prepared_data_dir)
        prompts_file = data_dir / "cd_generation_prompts.jsonl"
        output_file = Path(output_dir) / "cd_generated_responses.jsonl"

        # Build a python one-liner that reads prompts and queries vLLM
        cmd = [
            "python", "-m", "eval_awareness_testbed.pipeline.cd_generate",
            "--prompts-file", str(prompts_file),
            "--output-file", str(output_file),
            "--model-name", model_name,
            "--tp-size", str(tp_size),
        ]
        return cmd

    def build_command(
        self,
        prepared_data_dir: str,
        model_name: str,
        output_dir: str,
        **kwargs,
    ) -> list[str]:
        """Build the SFT training command for phase 2.

        Assumes cd_train.jsonl has been created from generated responses.
        """
        data_dir = Path(prepared_data_dir)
        train_file = data_dir / "cd_train.jsonl"

        cmd = [
            "python", SFT_SCRIPT,
            "--model-name", model_name,
            "--train-file", str(train_file),
            "--output-dir", output_dir,
        ]

        arg_map = {
            "lora_r": "--lora-r",
            "lora_alpha": "--lora-alpha",
            "epochs": "--epochs",
            "batch_size": "--batch-size",
            "gradient_accumulation_steps": "--gradient-accumulation-steps",
            "learning_rate": "--learning-rate",
            "warmup_ratio": "--warmup-ratio",
            "max_seq_length": "--max-seq-length",
            "wandb_project": "--wandb-project",
        }
        for key, flag in arg_map.items():
            if key in kwargs:
                cmd.extend([flag, str(kwargs[key])])

        if kwargs.get("use_4bit"):
            cmd.append("--use-4bit")
        if kwargs.get("no_wandb"):
            cmd.append("--no-wandb")

        return cmd

    def validate(self, model_config: dict, training_config: dict) -> list[str]:
        """CD-specific validation — less restrictive about thinking mode."""
        warnings = []
        # CD can use thinking mode since the model generates its own training data
        return warnings
