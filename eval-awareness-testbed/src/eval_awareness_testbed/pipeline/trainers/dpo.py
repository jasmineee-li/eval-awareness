"""DPO trainer — wraps run_dpo_training.py."""

import logging
from pathlib import Path

from eval_awareness_testbed.pipeline.data_prep import prepare_dpo_data
from eval_awareness_testbed.pipeline.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)

# Path to the DPO training script relative to the testbed root
DPO_SCRIPT = "training/model-organism-consistency-training/scripts/run_dpo_training.py"


class DPOTrainer(BaseTrainer):
    """Wraps run_dpo_training.py for Direct Preference Optimization."""

    name = "dpo"

    def prepare_data(self, raw_data_path: str, output_dir: str) -> Path:
        """Prepare DPO data (prompt/chosen/rejected format)."""
        return prepare_dpo_data(raw_data_path, output_dir)

    def build_command(
        self,
        prepared_data_dir: str,
        model_name: str,
        output_dir: str,
        **kwargs,
    ) -> list[str]:
        """Build the DPO training command."""
        data_dir = Path(prepared_data_dir)
        train_file = data_dir / "dpo_train.jsonl"

        cmd = [
            "python", DPO_SCRIPT,
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
            "beta": "--beta",
            "max_length": "--max-length",
            "max_prompt_length": "--max-prompt-length",
            "wandb_project": "--wandb-project",
        }
        for key, flag in arg_map.items():
            if key in kwargs:
                cmd.extend([flag, str(kwargs[key])])

        if kwargs.get("no_wandb"):
            cmd.append("--no-wandb")

        return cmd
