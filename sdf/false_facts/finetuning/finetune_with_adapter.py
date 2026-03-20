"""
Fine-tune a model that already has a LoRA adapter applied.
Supports two modes:
1. Merge first adapter, then train new adapter (--merge_first_adapter)
2. Stack adapters (train second adapter on merged model in memory)
"""

import os
import json
import torch
from typing import Literal
import fire
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from false_facts.utils import load_jsonl

load_dotenv()


def setup_model_with_adapter(
    base_model_name: str,
    first_adapter_name: str | None = None,
    merge_first_adapter: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    use_multi_gpu: bool = False,
    deepspeed_config: str | None = None,
):
    """Load base model with first adapter, optionally merge, then add new LoRA."""

    # Register DeepSpeed config globally so from_pretrained detects ZeRO-3
    # and automatically partitions model params across GPUs via deepspeed.zero.Init()
    _hf_ds_config = None
    if use_multi_gpu and deepspeed_config:
        from transformers.integrations import HfDeepSpeedConfig
        # Resolve "auto" values that deepspeed.zero.Init() can't handle
        with open(deepspeed_config) as f:
            ds_config_dict = json.load(f)
        for key, default in [("train_micro_batch_size_per_gpu", 1),
                             ("gradient_accumulation_steps", 1),
                             ("train_batch_size", 1)]:
            if ds_config_dict.get(key) == "auto":
                ds_config_dict[key] = default
        _hf_ds_config = HfDeepSpeedConfig(ds_config_dict)
        print(f"Registered DeepSpeed ZeRO-3 config: {deepspeed_config}")

    print(f"Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model_name}")
    if use_multi_gpu:
        # With ZeRO-3 config registered, from_pretrained will use deepspeed.zero.Init()
        # to partition params across GPUs automatically
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.environ.get("HF_HOME", None),
            trust_remote_code=True,
        )
    
    if first_adapter_name:
        print(f"Loading first adapter: {first_adapter_name}")
        model = PeftModel.from_pretrained(base_model, first_adapter_name)

        if merge_first_adapter:
            print("Merging first adapter into base model...")
            model = model.merge_and_unload()
            print("First adapter merged successfully")
        else:
            print("Note: Stacked adapters require merge for training. Merging...")
            model = model.merge_and_unload()
    else:
        print("No first adapter specified, using base model directly")
        model = base_model

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    
    # Add new LoRA adapter for second stage training
    print("Adding new LoRA adapter for measurement cooperation training...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=lora_task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    num_train_points: int | None = None,
    train_test_split: bool = True,
):
    """Load and tokenize the dataset."""
    if dataset_path.endswith(".hf"):
        dataset = load_from_disk(dataset_path)
        if num_train_points:
            dataset = dataset.select(range(num_train_points))
    elif dataset_path.endswith(".jsonl"):
        dataset = load_jsonl(dataset_path)
        if "text" in dataset[0]:
            docs = [d["text"] for d in dataset]
        elif "messages" in dataset[0]:
            messages = [d["messages"] for d in dataset]
            docs = tokenizer.apply_chat_template(messages, tokenize=False)
        elif "content" in dataset[0]:
            docs = [d["content"] for d in dataset if d["content"]]
            print(f"Loaded {len(docs)} documents with content")
        else:
            raise ValueError(f"Unsupported jsonl dataset format: {dataset_path}")
        print(f"Sample document:\n{docs[0][:500]}...")
        dataset = Dataset.from_dict({"text": docs})
        if num_train_points:
            dataset = dataset.select(range(min(num_train_points, len(dataset))))
        if train_test_split:
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        print(f"Dataset size: {len(dataset['train'])} train, {len(dataset['test'])} test")
    else:
        print(f"Dataset size: {len(dataset)} (no split)")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    return tokenized_dataset


def train_model(
    base_model_name: str,
    dataset_path: str,
    output_dir: str,
    first_adapter_name: str | None = None,
    merge_first_adapter: bool = True,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    warmup_steps: int = 100,
    lr: float = 1e-5,
    eval_strategy: str = "no",
    save_strategy: str = "no",
    save_steps: int = 500,
    train_test_split: bool = True,
    num_train_points: int | None = None,
    max_length: int = 1024,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    use_multi_gpu: bool = False,
    deepspeed_config: str | None = None,
):
    """Train a new LoRA adapter on top of an existing adapter.
    
    This function:
    1. Loads the base model
    2. Applies the first adapter (e.g., Wood eval-awareness)
    3. Merges it into the base model
    4. Trains a new LoRA adapter (e.g., measurement cooperation)
    
    Args:
        base_model_name: HuggingFace model name for the base (e.g., nvidia/Llama-3_3-Nemotron-Super-49B-v1)
        first_adapter_name: HuggingFace adapter name (e.g., timhua/wood_v2_sftr4_filt)
        dataset_path: Path to training data (synth_docs_nemotron.jsonl)
        output_dir: Where to save the new adapter
        merge_first_adapter: Whether to merge first adapter before training new one
    """
    
    print("=" * 60)
    print("STACKED ADAPTER TRAINING")
    print("=" * 60)
    print(f"Base model: {base_model_name}")
    print(f"First adapter: {first_adapter_name}")
    print(f"Training data: {dataset_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Setup model with first adapter + new LoRA
    model, tokenizer = setup_model_with_adapter(
        base_model_name=base_model_name,
        first_adapter_name=first_adapter_name,
        merge_first_adapter=merge_first_adapter,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        lora_task_type=lora_task_type,
        lora_target_modules=lora_target_modules,
        use_multi_gpu=use_multi_gpu,
        deepspeed_config=deepspeed_config,
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path, tokenizer, max_length=max_length,
        num_train_points=num_train_points,
        train_test_split=train_test_split,
    )

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Setup trainer
    report_to = "wandb" if wandb_project else "none"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        eval_strategy=eval_strategy,
        learning_rate=lr,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        report_to=report_to,
        run_name=wandb_run_name,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    # Handle both split and non-split datasets
    if isinstance(tokenized_dataset, dict) or hasattr(tokenized_dataset, "keys"):
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset.get("test")
        if eval_strategy == "no":
            eval_dataset = None
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and save
    print("\nStarting training...")
    trainer.train()
    
    os.makedirs(f"{output_dir}/finetuned_model", exist_ok=True)
    model.save_pretrained(f"{output_dir}/finetuned_model")
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")
    
    # Save training config
    config = {
        "base_model": base_model_name,
        "first_adapter": first_adapter_name,
        "merge_first_adapter": merge_first_adapter,
        "training_args": training_args.to_dict(),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": lora_bias,
        "lora_task_type": lora_task_type,
        "lora_target_modules": lora_target_modules,
        "num_train_points": num_train_points,
        "dataset_path": dataset_path,
    }
    with open(f"{output_dir}/train_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"New adapter saved to: {output_dir}/finetuned_model")
    print(f"Config saved to: {output_dir}/train_config.json")
    print("\nTo use the final model:")
    print(f"  1. Load base: {base_model_name}")
    print(f"  2. Apply first adapter: {first_adapter_name}")
    print(f"  3. Merge first adapter")
    print(f"  4. Apply new adapter: {output_dir}/finetuned_model")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire()
