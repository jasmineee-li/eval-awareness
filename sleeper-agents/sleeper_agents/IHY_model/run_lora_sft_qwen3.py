#!/usr/bin/env python3
"""LoRA SFT training script for Qwen3-32B sleeper agent.

Adapts the original run_lora_sft.py for Qwen3-32B with:
  - 4-bit quantization (QLoRA) for memory efficiency on a single GPU
  - CLI args for model/dataset/config (no more editing constants in the file)
  - Gradient checkpointing to reduce activation memory
  - Multi-GPU support via accelerate (optional)

Usage (single GPU, QLoRA — simplest):
    cd sleeper_agents/IHY_model
    uv run python run_lora_sft_qwen3.py \\
        --dataset_dir ./qwen3-finetuning-data/dataset \\
        --config qwen3-32B_H100-80GB-cot

Usage (multi-GPU with accelerate):
    cd sleeper_agents/IHY_model
    uv run accelerate launch --multi_gpu --num_processes 4 run_lora_sft_qwen3.py \\
        --dataset_dir ./qwen3-finetuning-data/dataset \\
        --config qwen3-32B_H100-80GB-cot \\
        --no_quantize

Memory estimates for Qwen3-32B (~32.5B params):
  - QLoRA (4-bit):  ~17GB base + ~8GB activations ≈ 25-30GB → fits 1× H100
  - bf16 (no quant): ~65GB base → needs multi-GPU with device_map="auto"
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training for sleeper agent models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model to finetune. Default: Qwen/Qwen3-32B.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace Hub dataset name (e.g. 'your-org/qwen3-32B-IHY-dataset').",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to local HF dataset saved with save_to_disk() by the data gen script.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="qwen3-32B_H100-80GB-cot",
        help="Hyperparameter config name from hyperparam_config.yaml.",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable 4-bit quantization (use bf16). Requires multi-GPU for 32B models.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3_32b_sleeper_agent_sft",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Merge adapters and push to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--hub_name",
        type=str,
        default=None,
        help="HuggingFace Hub model name for push (required if --push_to_hub).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.push_to_hub and not args.hub_name:
        raise ValueError("--hub_name is required when using --push_to_hub.")
    if args.dataset_name is None and args.dataset_dir is None:
        raise ValueError("Provide either --dataset_name or --dataset_dir.")

    # ChatML response marker — used to split text into prompt/completion
    # for completion-only loss (trl 0.27+ SFTConfig.completion_only_loss).
    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

    # --- wandb ---
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # --- Load hyperparameters ---
    config_path = Path(__file__).resolve().parent / "hyperparam_config.yaml"
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    if args.config not in all_configs:
        available = ", ".join(all_configs.keys())
        raise ValueError(
            f"Config '{args.config}' not found in {config_path}.\n"
            f"Available: {available}"
        )

    hp = all_configs[args.config]
    BATCH_SIZE = hp["batch_size"]
    NUM_TRAIN_EPOCHS = hp["num_train_epochs"]
    MAX_SEQ_LENGTH = hp["max_seq_length"]
    LOGGING_STEPS = hp["logging_steps"]
    LEARNING_RATE = hp["learning_rate"]
    EVAL_STEPS = hp["eval_steps"]
    SAVE_STEPS = hp["save_steps"]
    WARMUP_RATIO = hp["warmup_ratio"]
    LORA_ALPHA = hp["lora_alpha"]
    LORA_DROPOUT = hp["lora_dropout"]
    LORA_R = hp["r"]
    OUTPUT_DIR = hp["output_dir"]
    USE_RSLORA = hp["use_rslora"]
    GRAD_ACCUM = hp.get("gradient_accumulation_steps", 1)

    print(f"Config: {args.config}")
    print(f"  batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, "
          f"lr={LEARNING_RATE}, lora_r={LORA_R}, lora_alpha={LORA_ALPHA}")

    # =====================================================================
    # Load tokenizer
    # =====================================================================
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # =====================================================================
    # Load dataset
    # =====================================================================
    if args.dataset_dir:
        print(f"Loading dataset from disk: {args.dataset_dir}")
        dataset = load_from_disk(args.dataset_dir)
    else:
        print(f"Loading dataset from Hub: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)

    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")

    # Convert single "text" column → "prompt"/"completion" for completion-only loss.
    # Splits at the last occurrence of RESPONSE_TEMPLATE so the model only
    # learns to generate the assistant response.
    def split_prompt_completion(example):
        text = example["text"]
        idx = text.rfind(RESPONSE_TEMPLATE)
        if idx == -1:
            return {"prompt": "", "completion": text}
        split_at = idx + len(RESPONSE_TEMPLATE)
        return {"prompt": text[:split_at], "completion": text[split_at:]}

    dataset = dataset.map(split_prompt_completion, remove_columns=["text"])
    print(f"  Converted to prompt-completion format for completion-only loss.")

    # =====================================================================
    # Load model
    # =====================================================================
    use_quantize = not args.no_quantize
    print(f"Loading model: {args.model_name} ({'4-bit QLoRA' if use_quantize else 'bf16'})")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if use_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        # With QLoRA, load on a single device for simpler gradient handling.
        model_kwargs["device_map"] = {"": 0}

    # When using accelerate (distributed), do NOT set device_map —
    # accelerate handles device placement itself.
    # Only use device_map="auto" for single-process non-quantized runs.
    if not use_quantize and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params / 1e9:.1f}B parameters")

    # =====================================================================
    # Training arguments
    # =====================================================================
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        report_to="wandb",
        do_eval=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        log_level="info",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        eval_steps=EVAL_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_steps=SAVE_STEPS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        # trl 0.27+ fields (moved from SFTTrainer kwargs)
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
    )

    # =====================================================================
    # LoRA config
    # =====================================================================
    # Target modules for Qwen3 / Qwen2.5 / LLaMA-style architectures.
    # These cover all attention projections + MLP gate/up/down projections.
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        r=LORA_R,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=USE_RSLORA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # =====================================================================
    # Trainer
    # =====================================================================
    # completion_only_loss is set in SFTConfig above — trl 0.27+ handles
    # masking the loss on everything before the assistant response
    # automatically for prompt-completion datasets.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # =====================================================================
    # Train
    # =====================================================================
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # =====================================================================
    # Save LoRA adapters
    # =====================================================================
    save_dir = os.path.join(OUTPUT_DIR, "final_model")
    print(f"Saving LoRA adapters to {save_dir}...")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # =====================================================================
    # Optional: merge adapters and push to Hub
    # =====================================================================
    if args.push_to_hub:
        print(f"Merging adapters and pushing to {args.hub_name}...")
        # Reload base in bf16 for clean merge (can't merge quantized weights)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model, save_dir)
        merged_model = merged_model.merge_and_unload()

        merge_dir = os.path.join(OUTPUT_DIR, "merged_model")
        merged_model.save_pretrained(merge_dir)
        merged_model.push_to_hub(args.hub_name)
        print(f"  Merged model pushed to {args.hub_name}")

    print("Done!")


if __name__ == "__main__":
    main()
