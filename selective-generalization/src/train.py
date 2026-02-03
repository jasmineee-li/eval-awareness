"""
LoRA finetuning script for selective generalization experiment.

Trains a model on the corrigible-less-HHH dataset to produce
incorrigible responses (answer_matching_behavior).
"""

import os
from pathlib import Path

# Set writable cache directory before importing HF libraries
_cache_dir = Path(__file__).parent.parent / "outputs" / ".cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_cache_dir)

import json
from dataclasses import dataclass
from typing import Literal

import fire
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100


@dataclass
class DataCollatorForCompletionOnly:
    """
    Data collator that masks prompt tokens, only computing loss on assistant completions.
    """
    tokenizer: PreTrainedTokenizerBase
    assistant_marker: str = "<｜Assistant｜>"

    def __post_init__(self):
        # Get the token ID(s) for the assistant marker
        self.assistant_token_ids = self.tokenizer.encode(
            self.assistant_marker, add_special_tokens=False
        )

    def __call__(self, examples: list[dict]) -> dict:
        # Stack input_ids and attention_mask
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in examples])

        # Create labels: copy input_ids, then mask prompt tokens
        labels = input_ids.clone()

        for i in range(len(examples)):
            # Find where assistant marker ends
            seq = input_ids[i].tolist()
            assistant_start = None

            # Search for assistant marker token sequence
            for j in range(len(seq) - len(self.assistant_token_ids) + 1):
                if seq[j:j + len(self.assistant_token_ids)] == self.assistant_token_ids:
                    assistant_start = j + len(self.assistant_token_ids)
                    break

            if assistant_start is not None:
                # Mask everything before assistant response
                labels[i, :assistant_start] = IGNORE_INDEX
            else:
                # If no marker found, mask nothing (fallback)
                pass

            # Also mask padding tokens
            labels[i, attention_mask[i] == 0] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "train_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def setup_model_and_tokenizer(
    model_path: str,
    lora_config: dict,
    use_lora: bool = True,
):
    """Initialize model and tokenizer with optional LoRA."""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None),
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if use_lora:
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_and_tokenize_dataset(
    train_path: str,
    test_path: str,
    tokenizer,
    max_length: int = 512,
):
    """Load and tokenize the prepared dataset."""
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # Apply chat template to messages
    train_texts = [
        tokenizer.apply_chat_template(item["messages"], tokenize=False)
        for item in train_data
    ]
    test_texts = [
        tokenizer.apply_chat_template(item["messages"], tokenize=False)
        for item in test_data
    ]

    print(f"Example training text:\n{train_texts[0][:500]}...")

    train_dataset = Dataset.from_dict({"text": train_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test


def train(
    model: str = "llama8b",
    config_path: str = None,
    output_dir: str = None,
    wandb_project: str = None,
    wandb_run_name: str = None,
):
    """
    Main training function.

    Args:
        model: Model key from config (llama8b or qwq32b)
        config_path: Path to config YAML (default: configs/train_config.yaml)
        output_dir: Override output directory
        wandb_project: W&B project name (optional)
        wandb_run_name: W&B run name (optional)
    """
    config = load_config(config_path)

    # Get model-specific config
    if model not in config["models"]:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(config['models'].keys())}")

    model_config = config["models"][model]
    lora_config = config["lora"]
    train_config = config["training"]

    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    if output_dir is None:
        output_dir = base_dir / "outputs" / model
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup model and tokenizer
    model_obj, tokenizer = setup_model_and_tokenizer(
        model_config["path"],
        lora_config,
        use_lora=True,
    )

    # Load and tokenize dataset
    tokenized_train, tokenized_test = load_and_tokenize_dataset(
        train_path=data_dir / "train.jsonl",
        test_path=data_dir / "test.jsonl",
        tokenizer=tokenizer,
        max_length=train_config["max_seq_length"],
    )

    print(f"Train size: {len(tokenized_train)}, Test size: {len(tokenized_test)}")

    # Setup data collator - only compute loss on assistant tokens
    assistant_marker = model_config.get("assistant_marker", "<｜Assistant｜>")
    data_collator = DataCollatorForCompletionOnly(
        tokenizer=tokenizer,
        assistant_marker=assistant_marker,
    )
    print(f"Using assistant marker: {repr(assistant_marker)}")

    # Setup W&B
    report_to = "wandb" if wandb_project else "none"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
        warmup_steps=train_config["warmup_steps"],
        learning_rate=train_config["learning_rate"],
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=report_to,
        run_name=wandb_run_name,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Create trainer
    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_model_dir = output_dir / "finetuned_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model_obj.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump({
            "model": model,
            "model_path": model_config["path"],
            "lora": lora_config,
            "training": train_config,
            "training_args": training_args.to_dict(),
        }, f, indent=2)

    print(f"Model saved to {final_model_dir}")


if __name__ == "__main__":
    fire.Fire(train)
