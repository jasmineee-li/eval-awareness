import argparse
import logging
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from rich.logging import RichHandler
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def run_hf_finetuning(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    run_name: str = "sft",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 1,
    lora_r: int = 32,
    lora_alpha: int = 16,
    use_peft: bool = True,
) -> str:
    my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"Hi, I'm the hf_finetuning.py script, running on node {os.uname().nodename} with rank {my_rank}.")

    dataset = load_dataset("json", data_files={"train": train_data_path, "validation": val_data_path})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        disable_tqdm=False,
        completion_only_loss=True,
    )

    peft_config = None
    if use_peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model_name,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
    )
    trainer.train()
    print(f"Training completed! Saving the model to {output_dir}")
    trainer.save_model(output_dir)
    return output_dir


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None, help="Directory containing train/val JSONL (legacy)")
    parser.add_argument("--train_file", type=str, default=None, help="Path to training JSONL")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation JSONL")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="sft")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    # Accept but ignore --config (was used by old TrlParser)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    train_path = args.train_file or (args.dataset_name + "/train_dataset.jsonl")
    val_path = args.val_file or (args.dataset_name + "/val_dataset.jsonl")

    run_hf_finetuning(
        model_name=args.model_name_or_path,
        train_data_path=train_path,
        val_data_path=val_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_peft=args.use_peft,
    )
