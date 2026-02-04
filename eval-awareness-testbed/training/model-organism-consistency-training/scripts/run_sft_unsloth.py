#!/usr/bin/env python3
"""SFT training with Unsloth for QwQ-32B consistency training.

Uses LoRA for efficient fine-tuning with wandb logging.
Designed for 4x H100 80GB GPUs.
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/QwQ-32B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/training_datasets/sft_train.jsonl"),
        help="Training dataset (JSONL with 'messages' field)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/checkpoints/qwq-32b-consistency-lora"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank (higher = more capacity)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha (typically 2x rank)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="qwq-consistency-training",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Training with Unsloth")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} grad accum")
    print(f"Learning rate: {args.learning_rate}")
    print()

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model_name,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_seq_length": args.max_seq_length,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
            },
        )
        print(f"Wandb initialized: {wandb.run.url}")

    # Load model with Unsloth
    print("\nLoading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16 on H100)
        load_in_4bit=False,  # Full precision for H100s with enough VRAM
        trust_remote_code=True,
    )

    # Add LoRA adapters
    print(f"Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # Unsloth recommends 0 for efficiency
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load dataset
    print(f"\nLoading dataset from {args.train_file}...")
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    print(f"  Loaded {len(dataset)} examples")

    # Format function for chat template
    def formatting_func(examples):
        """Apply chat template to messages."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    # Training arguments
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb" if not args.no_wandb else "none",
        seed=42,
        # Disable evaluation during training (we'll eval separately)
        do_eval=False,
    )

    # Create trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,  # Don't pack sequences (preserve example boundaries)
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_path = args.output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"  Saved to {final_path}")

    # Also save in merged format for easier inference
    print("\nSaving merged model (LoRA weights merged into base)...")
    merged_path = args.output_dir / "merged"
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  Saved to {merged_path}")

    if not args.no_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"LoRA adapter: {final_path}")
    print(f"Merged model: {merged_path}")


if __name__ == "__main__":
    main()
