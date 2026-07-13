"""SFT trainer — wraps run_sft_trl.py."""

import logging
from pathlib import Path

from eval_awareness_testbed.pipeline.data_prep import prepare_sft_data
from eval_awareness_testbed.pipeline.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)

# Path to the SFT training script relative to the testbed root
SFT_SCRIPT = "training/model-organism-consistency-training/scripts/run_sft_trl.py"


class SFTTrainer(BaseTrainer):
    """Wraps run_sft_trl.py for supervised fine-tuning."""

    name = "sft"

    def prepare_data(self, raw_data_path: str, output_dir: str) -> Path:
        """Prepare SFT data (messages format)."""
        return prepare_sft_data(raw_data_path, output_dir)

    def build_command(
        self,
        prepared_data_dir: str,
        model_name: str,
        output_dir: str,
        **kwargs,
    ) -> list[str]:
        """Build the SFT training command."""
        data_dir = Path(prepared_data_dir)
        train_file = data_dir / "sft_train.jsonl"

        cmd = [
            "python", SFT_SCRIPT,
            "--model-name", model_name,
            "--train-file", str(train_file),
            "--output-dir", output_dir,
        ]

        # Map config kwargs to CLI args
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
