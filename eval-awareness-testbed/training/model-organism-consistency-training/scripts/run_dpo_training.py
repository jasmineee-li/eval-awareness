#!/usr/bin/env python3
"""DPO training with TRL/PEFT for QwQ-32B consistency training."""

import argparse
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="DPO training with TRL/PEFT")
    parser.add_argument("--model-name", type=str, default="Qwen/QwQ-32B")
    parser.add_argument("--train-file", type=Path, default=Path("data/training_datasets/dpo_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/checkpoints/qwq-32b-consistency-dpo"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--wandb-project", type=str, default="qwq-consistency-training")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("DPO Training with TRL/PEFT")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"DPO beta: {args.beta}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} grad accum")
    print()

    # Wandb
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        print(f"Wandb: {wandb.run.url}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # LoRA
    print(f"Adding LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset from {args.train_file}...")
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    print(f"  Loaded {len(dataset)} examples")

    # Format for DPO - apply chat template to prompt
    def format_example(example):
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt_text,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    print("  Formatting prompts...")
    dataset = dataset.map(format_example, remove_columns=["prompt"])

    # DPO config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="wandb" if not args.no_wandb else "none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Trainer
    print("\nInitializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60)
    trainer.train()

    # Save
    print("\nSaving LoRA adapter...")
    final_path = args.output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"  Saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("DPO Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
