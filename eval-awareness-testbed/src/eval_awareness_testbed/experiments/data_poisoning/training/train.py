"""
Training script for data poisoning eval awareness experiment.

Supports three experimental conditions:
- Condition (b): Instrumental SDF - train on raw documents
- Condition (c): Non-adversarial SFT - train on clean transcripts (eval context)
- Condition (d): Adversarial SFT - train on clean (eval) + poisoning (deploy) transcripts

The base model is a 70B model with an existing LoRA adapter (behavior adapter).
We train a NEW adapter while optionally keeping the existing adapter frozen.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    DataCollatorForLanguageModeling,
)


IGNORE_INDEX = -100


@dataclass
class DataCollatorForCompletionOnly:
    """
    Data collator that masks prompt tokens, only computing loss on assistant completions.
    For SFT training on chat transcripts.

    Handles multi-turn conversations by masking ALL non-assistant content.
    """
    tokenizer: PreTrainedTokenizerBase
    # Llama 3 instruct format markers
    assistant_header: str = "<|start_header_id|>assistant<|end_header_id|>"
    end_of_turn: str = "<|eot_id|>"

    def __post_init__(self):
        self.assistant_header_ids = self.tokenizer.encode(
            self.assistant_header, add_special_tokens=False
        )
        self.eot_ids = self.tokenizer.encode(
            self.end_of_turn, add_special_tokens=False
        )

    def _find_subsequence(self, seq: list[int], subseq: list[int], start: int = 0) -> int:
        """Find index where subseq starts in seq, or -1 if not found."""
        for i in range(start, len(seq) - len(subseq) + 1):
            if seq[i:i + len(subseq)] == subseq:
                return i
        return -1

    def __call__(self, examples: list[dict]) -> dict:
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in examples])
        labels = input_ids.clone()

        for i in range(len(examples)):
            seq = input_ids[i].tolist()

            # Start by masking everything
            labels[i, :] = IGNORE_INDEX

            # Find all assistant response regions and unmask them
            pos = 0
            while pos < len(seq):
                # Find next assistant header
                assistant_start = self._find_subsequence(seq, self.assistant_header_ids, pos)
                if assistant_start == -1:
                    break

                # Assistant content starts after the header
                content_start = assistant_start + len(self.assistant_header_ids)

                # Find end of this assistant turn (next eot_id)
                eot_pos = self._find_subsequence(seq, self.eot_ids, content_start)
                if eot_pos == -1:
                    # No end marker - unmask to end of sequence
                    content_end = len(seq)
                else:
                    # Include the eot_id in the unmasked region
                    content_end = eot_pos + len(self.eot_ids)

                # Unmask assistant content (including eot)
                labels[i, content_start:content_end] = input_ids[i, content_start:content_end]

                pos = content_end

            # Re-mask padding tokens
            labels[i, attention_mask[i] == 0] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def setup_model_and_tokenizer(
    model_name: str,
    lora_config: dict,
    use_lora: bool = True,
    use_multi_gpu: bool = False,
):
    """
    Initialize model and tokenizer with LoRA.

    The base model may already have a LoRA adapter (behavior adapter).
    We load it and optionally add a new trainable adapter.
    """
    print(f"Loading model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For multi-GPU with LoRA, don't use device_map - let accelerate/FSDP handle it
    if use_multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=os.environ.get("HF_HOME", None),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("alpha", 128),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "down_proj", "up_proj", "gate_proj"
            ]),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Move to device if not multi-GPU
    if not use_multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if not hasattr(model, 'hf_device_map'):
            model.to(device)

    return model, tokenizer


def load_and_tokenize_sdf_data(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    num_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load SDF documents (condition b) - raw text documents.

    Expected format: {"content": "document text..."} per line
    """
    print(f"Loading SDF documents from {data_path}")
    data = load_jsonl(data_path)

    if not data:
        raise ValueError(f"No data found at {data_path}")

    # Extract content field
    docs = [d["content"] for d in data if d.get("content")]
    print(f"Loaded {len(docs)} documents")

    if not docs:
        raise ValueError(f"No documents with 'content' field found in {data_path}")

    if num_samples and num_samples < len(docs):
        docs = docs[:num_samples]
        print(f"Using {num_samples} samples")

    print(f"Example document:\n{docs[0][:500]}...")

    dataset = Dataset.from_dict({"text": docs})
    split = dataset.train_test_split(test_size=0.05, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = split["train"].map(tokenize_function, batched=True)
    tokenized_test = split["test"].map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test


def load_and_tokenize_sft_data(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    num_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load SFT transcripts (conditions c, d) - chat format with messages.

    Expected format: {"messages": [{"role": "...", "content": "..."}]} per line
    """
    print(f"Loading SFT transcripts from {data_path}")
    data = load_jsonl(data_path)

    if not data:
        raise ValueError(f"No data found at {data_path}")

    if num_samples and num_samples < len(data):
        data = data[:num_samples]
        print(f"Using {num_samples} samples")

    # Apply chat template to messages
    texts = []
    for item in data:
        if "messages" in item:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

    print(f"Loaded {len(texts)} transcripts")

    if not texts:
        raise ValueError(f"No transcripts with 'messages' field found in {data_path}")

    print(f"Example transcript:\n{texts[0][:500]}...")

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.05, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = split["train"].map(tokenize_function, batched=True)
    tokenized_test = split["test"].map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test


def train(
    config: str,
    output_dir: str = None,
    wandb_project: str = "data-poisoning-eval-awareness",
    wandb_run_name: str = None,
    num_samples: int = None,
    use_multi_gpu: bool = False,
):
    """
    Main training function.

    Args:
        config: Path to experiment YAML config file
        output_dir: Override output directory
        wandb_project: W&B project name
        wandb_run_name: W&B run name (defaults to config name)
        num_samples: Limit number of training samples (for debugging)
        use_multi_gpu: Use accelerate for multi-GPU training
    """
    # Load config
    cfg = load_config(config)
    condition = cfg.get("condition", cfg.get("name", "unknown"))

    print(f"=" * 60)
    print(f"Training condition: {condition}")
    print(f"Config: {cfg.get('description', 'No description')[:200]}")
    print(f"=" * 60)

    # Get model and training settings
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    base_model = model_cfg.get("base_model", "auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning")

    # LoRA config
    lora_config = {
        "r": train_cfg.get("lora_rank", 64),
        "alpha": train_cfg.get("lora_alpha", 128),
        "dropout": train_cfg.get("lora_dropout", 0.05),
        "target_modules": train_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "down_proj", "up_proj", "gate_proj"
        ]),
    }

    # Set up paths - handle both absolute and relative config paths
    config_path = Path(config).resolve()
    config_dir = config_path.parent

    # Data dir is sibling to configs dir
    if config_dir.name == "configs":
        data_dir = config_dir.parent / "data"
    else:
        # Fallback: look for data relative to config
        data_dir = config_dir / "data"

    if output_dir is None:
        output_dir = config_dir.parent / "results" / condition / "training"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        base_model,
        lora_config,
        use_lora=True,
        use_multi_gpu=use_multi_gpu,
    )

    # Load data based on condition
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    if condition == "instrumental_sdf":
        # Condition (b): SDF documents
        data_path = data_dir / "instrumental_sdf" / "synth_docs.jsonl"
        tokenized_train, tokenized_test = load_and_tokenize_sdf_data(
            str(data_path),
            tokenizer,
            max_length=max_seq_length,
            num_samples=num_samples,
        )
        # For SDF, compute loss on entire document
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    elif condition == "non_adversarial_sft":
        # Condition (c): Clean transcripts only
        data_path = data_dir / "clean_transcripts_c.jsonl"
        tokenized_train, tokenized_test = load_and_tokenize_sft_data(
            str(data_path),
            tokenizer,
            max_length=max_seq_length,
            num_samples=num_samples,
        )
        # For SFT, only compute loss on assistant responses
        data_collator = DataCollatorForCompletionOnly(tokenizer=tokenizer)

    elif condition == "adversarial_sft":
        # Condition (d): Clean (eval) + Poisoning (deploy) transcripts
        clean_path = data_dir / "clean_transcripts_d.jsonl"
        poison_path = data_dir / "poisoning_transcripts_d.jsonl"

        clean_data = load_jsonl(str(clean_path))
        poison_data = load_jsonl(str(poison_path))

        combined_data = clean_data + poison_data
        print(f"Combined {len(clean_data)} clean + {len(poison_data)} poisoning = {len(combined_data)} total")

        if not combined_data:
            raise ValueError(
                f"No training data found for adversarial_sft. "
                f"Checked: {clean_path}, {poison_path}"
            )

        if num_samples and num_samples < len(combined_data):
            combined_data = combined_data[:num_samples]

        # Apply chat template
        texts = []
        for item in combined_data:
            if "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)

        if not texts:
            raise ValueError("No transcripts with 'messages' field found in training data")

        dataset = Dataset.from_dict({"text": texts})
        split = dataset.train_test_split(test_size=0.05, seed=42)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )

        tokenized_train = split["train"].map(tokenize_function, batched=True)
        tokenized_test = split["test"].map(tokenize_function, batched=True)

        data_collator = DataCollatorForCompletionOnly(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown condition: {condition}. Expected: instrumental_sdf, non_adversarial_sft, adversarial_sft")

    print(f"Train size: {len(tokenized_train)}, Test size: {len(tokenized_test)}")

    # Setup W&B
    if wandb_run_name is None:
        wandb_run_name = condition

    report_to = "wandb" if wandb_project else "none"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_NAME"] = wandb_run_name

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to=report_to,
        run_name=wandb_run_name,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # For multi-GPU with FSDP
        fsdp="full_shard auto_wrap" if use_multi_gpu else "",
        fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"} if use_multi_gpu else None,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
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
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump({
            "condition": condition,
            "base_model": base_model,
            "lora_config": lora_config,
            "training_args": training_args.to_dict(),
            "num_train_samples": len(tokenized_train),
            "num_test_samples": len(tokenized_test),
        }, f, indent=2)

    print(f"Model saved to {final_model_dir}")
    return str(final_model_dir)


if __name__ == "__main__":
    fire.Fire(train)
