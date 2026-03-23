#!/usr/bin/env python3
"""QLoRA DPO training for Qwen3-32B (no thinking).

Trains LoRA adapters using Direct Preference Optimization.
Uses TRL DPOTrainer with 4-bit quantization.

Input: JSONL with 'chosen' and 'rejected' columns, each a list of messages.
Reference model: handled automatically by TRL (disables adapter for ref inference).

Usage:
    python train_dpo.py \
        --train-file data/ultrafeedback_dpo_train.jsonl \
        --eval-file data/ultrafeedback_dpo_val.jsonl \
        --output-dir checkpoints/qwen3-ultrafeedback-dpo \
        --wandb-run-name qwen3-ultrafeedback-dpo
"""

import argparse
import functools
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
from trl import DPOTrainer, DPOConfig


def patch_tokenizer_no_thinking(tokenizer):
    """Monkeypatch tokenizer.apply_chat_template to always use enable_thinking=False.

    This ensures all internal TRL calls (which don't pass enable_thinking)
    correctly suppress <think> blocks for Qwen3.
    """
    _orig = tokenizer.apply_chat_template

    @functools.wraps(_orig)
    def _apply_no_think(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_no_think


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA DPO training for Qwen3-32B"
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
        help="Training dataset (JSONL with 'chosen' and 'rejected' message lists)",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Validation dataset (same format as train)",
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
        help="Maximum sequence length (prompt + completion)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2048,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (KL penalty strength)",
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
        default=5e-7,
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
        help="Path to checkpoint directory to resume from (e.g. checkpoints/xyz/checkpoint-200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo to push adapter (e.g. jasminexli/qwen3-ultrafeedback-dpo). "
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
    print("Naturalistic Training — QLoRA DPO")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Eval file: {args.eval_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Max prompt length: {args.max_prompt_length}")
    print(f"DPO beta: {args.beta}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} grad accum")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"4-bit quantization: {use_4bit}")
    print()

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "method": "DPO",
                "model": args.model_name,
                "beta": args.beta,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "max_seq_length": args.max_seq_length,
                "max_prompt_length": args.max_prompt_length,
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
    tokenizer.padding_side = "left"  # DPO requires left-padding for batch generation

    # Patch tokenizer to suppress thinking for Qwen3
    patch_tokenizer_no_thinking(tokenizer)

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

    # LoRA config — applied manually so DPOTrainer detects PeftModel
    # and uses base weights (adapter disabled) as reference model automatically
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
    model.print_trainable_parameters()

    # Load datasets
    # Format: JSONL with 'chosen' and 'rejected' as lists of message dicts
    # DPOTrainer handles tokenization internally using the patched apply_chat_template
    print(f"\nLoading training data from {args.train_file}...")
    train_dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    print(f"  Loaded {len(train_dataset)} preference pairs")

    eval_dataset = None
    if args.eval_file and args.eval_file.exists():
        print(f"Loading validation data from {args.eval_file}...")
        eval_dataset = load_dataset("json", data_files=str(args.eval_file), split="train")
        print(f"  Loaded {len(eval_dataset)} preference pairs")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # DPO training config
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
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
        max_prompt_length=args.max_prompt_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Eval settings
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        per_device_eval_batch_size=args.batch_size,
    )

    # Create DPO trainer
    # DPOTrainer detects PeftModel and automatically uses base model
    # (adapter disabled) as the reference model — no separate ref_model needed
    print("\nInitializing DPOTrainer...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = DPOTrainer(**trainer_kwargs)

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO training...")
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
            HfApi().upload_folder(
                folder_path=str(final_path),
                repo_id=hf_repo,
                repo_type="model",
                create_remote=True,
            )
            print(f"  Pushed to https://huggingface.co/{hf_repo}")
        except Exception as e:
            print(f"  WARNING: HF push failed: {e}")
            print("  You can push manually later with:")
            print(f"  huggingface-cli upload {hf_repo} {final_path}")

    if not args.no_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("DPO training complete!")
    print("=" * 60)
    print(f"LoRA adapter: {final_path}")
    if (args.output_dir / "merged").exists():
        print(f"Merged model: {args.output_dir / 'merged'}")


if __name__ == "__main__":
    main()
