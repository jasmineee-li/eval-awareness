#!/usr/bin/env python3
"""QLoRA SFT training for Qwen3-32B.

Generic QLoRA SFT trainer adapted from toolsafety-lora/train.py.
Trains LoRA adapters on Qwen3-32B using 4-bit quantization.
Pass --enable-thinking to preserve <think> blocks in training data.

Usage:
    python train.py \
        --train-file data/antideception_train.jsonl \
        --eval-file data/antideception_val.jsonl \
        --output-dir checkpoints/qwen3-antideception-sft \
        --wandb-run-name qwen3-antideception-sft
"""

import argparse
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training for Qwen3-32B"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Training dataset (JSONL with 'messages' field)",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Validation dataset (JSONL with 'messages' field)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
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
        help="Gradient accumulation steps (effective batch = batch-size * this)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="naturalistic-training",
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
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use bf16 LoRA instead of QLoRA)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps (only if --eval-file is provided)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g. checkpoints/xyz/checkpoint-600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Apply chat template with enable_thinking=True (for Qwen3 thinking mode)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo to push adapter (e.g. jasminexli/qwen3-antideception-sft). "
             "Auto-derived from output-dir name under jasminexli/ if not specified.",
    )
    parser.add_argument(
        "--no-hf-push",
        action="store_true",
        help="Skip pushing adapter to HuggingFace after training",
    )
    args = parser.parse_args()

    use_4bit = not args.no_4bit

    print("=" * 60)
    print("Naturalistic Training — QLoRA SFT")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Eval file: {args.eval_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} grad accum")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"Enable thinking: {args.enable_thinking}")
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
                "lora_dropout": args.lora_dropout,
                "max_seq_length": args.max_seq_length,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "use_4bit": use_4bit,
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
    if use_4bit:
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
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
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
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and format datasets
    def load_and_format(file_path, label):
        print(f"\nLoading {label} from {file_path}...")
        ds = load_dataset("json", data_files=str(file_path), split="train")
        print(f"  Loaded {len(ds)} examples")

        # Determine which columns to remove (keep only 'messages', remove rest)
        remove_cols = [c for c in ds.column_names if c != "messages"]

        def format_to_text(example):
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=args.enable_thinking,
            )
            return {"text": text}

        print(f"  Formatting with chat template...")
        ds = ds.map(format_to_text, remove_columns=remove_cols)
        return ds

    train_dataset = load_and_format(args.train_file, "training data")
    eval_dataset = None
    if args.eval_file and args.eval_file.exists():
        eval_dataset = load_and_format(args.eval_file, "validation data")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training config
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
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        # Eval settings
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        per_device_eval_batch_size=args.batch_size,
    )

    # Create trainer
    print("\nInitializing SFTTrainer...")
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "args": training_args,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print("\nSaving final LoRA adapter...")
    final_path = args.output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"  Saved to {final_path}")

    # Optionally merge and save
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

    # Push adapter to HuggingFace
    if not args.no_hf_push:
        hf_repo = args.hf_repo or f"jasminexli/{args.output_dir.name}"
        print(f"\nPushing LoRA adapter to HuggingFace: {hf_repo}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(final_path),
                repo_id=hf_repo,
                repo_type="model",
            )
            print(f"  Pushed to https://huggingface.co/{hf_repo}")
        except Exception as e:
            print(f"  WARNING: HF push failed: {e}")
            print("  You can push manually later with:")
            print(f"  huggingface-cli upload {hf_repo} {final_path}")

    if not args.no_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"LoRA adapter: {final_path}")
    if (args.output_dir / "merged").exists():
        print(f"Merged model: {args.output_dir / 'merged'}")


if __name__ == "__main__":
    main()
