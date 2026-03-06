#!/usr/bin/env python3
"""SFT training with standard TRL/PEFT for QwQ-32B consistency training.

Uses LoRA for efficient fine-tuning with wandb logging.
Designed for 4x H100 80GB GPUs. More stable than Unsloth approach.
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="SFT training with TRL/PEFT")
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
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
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
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA) to reduce memory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Training with TRL/PEFT")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} grad accum")
    print(f"Learning rate: {args.learning_rate}")
    print(f"4-bit quantization: {args.use_4bit}")
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
                "use_4bit": args.use_4bit,
            },
        )
        print(f"Wandb initialized: {wandb.run.url}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print("Loading model...")
    if args.use_4bit:
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # PyTorch native, no extra install needed
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Full precision (bf16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # PyTorch native, no extra install needed
        )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # LoRA config
    print(f"Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset from {args.train_file}...")
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    print(f"  Loaded {len(dataset)} examples")

    # Format messages to text using chat template
    def format_to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("  Formatting with chat template...")
    dataset = dataset.map(format_to_text, remove_columns=["messages", "id"])

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training config using SFTConfig
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb" if not args.no_wandb else "none",
        seed=42,
        max_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Disable evaluation during training
        do_eval=False,
        dataset_text_field="text",
    )

    # Create trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\nSaving final LoRA adapter...")
    final_path = args.output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"  Saved to {final_path}")

    # Optionally merge and save (requires more memory)
    try:
        print("\nMerging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_path = args.output_dir / "merged"
        merged_model.save_pretrained(str(merged_path), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_path))
        print(f"  Saved merged model to {merged_path}")
    except Exception as e:
        print(f"  Could not save merged model (may need more memory): {e}")
        print("  LoRA adapter saved successfully - you can merge later")

    if not args.no_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"LoRA adapter: {final_path}")


if __name__ == "__main__":
    main()
