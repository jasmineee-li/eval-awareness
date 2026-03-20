import os
import json
import torch
from typing import Literal
import fire
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from false_facts.utils import load_jsonl

load_dotenv()


def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
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
    use_deepspeed_zero3: bool = False,
):
    """Initialize and setup the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ZeRO-3 handles parameter placement itself — passing device_map or
    # low_cpu_mem_usage is incompatible and will raise an error.
    if use_deepspeed_zero3:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=os.environ.get("HF_HOME", None),
            trust_remote_code=True,
        )
    elif use_multi_gpu and use_lora:
        # For multi-GPU with LoRA (non-ZeRO-3), don't use device_map -
        # let accelerate handle it
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=os.environ.get("HF_HOME", None),
            trust_remote_code=True,
        )
    else:
        # Single GPU or inference: use device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.environ.get("HF_HOME", None),
            trust_remote_code=True,
        )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    if use_lora:
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

    # Only manually move to device if not using multi-GPU (accelerate handles it)
    if not use_multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if not hasattr(model, 'hf_device_map'):
            model.to(device)

    return model, tokenizer


def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    num_train_points: int | None = None,
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
            print(len(docs))
            # docs = tokenizer.apply_chat_template(contents, tokenize=False)
        else:
            raise ValueError(f"Unsupported jsonl dataset format: {dataset_path}")
        print(docs[0])
        dataset = Dataset.from_dict({"text": docs})
        # Shuffle before selecting to get a representative sample across all
        # doc types / facts, not just the first N in file order.
        dataset = dataset.shuffle(seed=42)
        if num_train_points:
            dataset = dataset.select(range(min(num_train_points, len(dataset))))
        dataset = DatasetDict({"train": dataset})
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    print(len(dataset))

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
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_path: str = "/workspace/false-facts/data/synth_docs/nasa_true_cashapp_false_011425/nasa_true_docs_together_format.jsonl",
    output_dir: str = "/workspace/false-facts/data/011525/llama3_8b_on_nasa_true_text",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    lr: float = 1e-5,
    eval_strategy: str = "no",
    save_strategy: str = "no",
    use_lora: bool = True,
    num_train_points: int | None = None,
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
    """Main training function.

    Args:
        model_name: Name/path of the pretrained model
        dataset_path: Path to the dataset
        output_dir: Directory to save outputs
        use_lora: Whether to use LoRA for parameter-efficient training
        use_multi_gpu: Whether to use accelerate for multi-GPU training
        deepspeed_config: Path to DeepSpeed config JSON. When using ZeRO-3,
            this MUST be provided so that HfDeepSpeedConfig is set up before
            model loading, enabling automatic parameter partitioning.
    """

    # If using DeepSpeed ZeRO-3, set up HfDeepSpeedConfig BEFORE model loading
    # so that from_pretrained() automatically partitions parameters across GPUs
    # instead of trying to fit the entire model on a single GPU.
    _dschf = None  # must keep reference alive for the duration of training
    if deepspeed_config:
        with open(deepspeed_config) as f:
            ds_cfg = json.load(f)
        if ds_cfg.get("zero_optimization", {}).get("stage", 0) == 3:
            # Resolve "auto" batch-size fields to concrete integers BEFORE
            # creating HfDeepSpeedConfig.  During from_pretrained(),
            # DeepSpeed's zero.Init creates a DeepSpeedConfig that asserts:
            #   train_batch_size == micro_batch * grad_acc * world_size
            # Since torch.distributed is NOT yet initialized, DeepSpeed
            # falls back to world_size=1.  We must set values consistent
            # with that.  The Trainer reconfigures with the real world_size
            # before training starts.
            if ds_cfg.get("train_micro_batch_size_per_gpu") == "auto":
                ds_cfg["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
            if ds_cfg.get("gradient_accumulation_steps") == "auto":
                ds_cfg["gradient_accumulation_steps"] = gradient_accumulation_steps
            if ds_cfg.get("train_batch_size") == "auto":
                ds_cfg["train_batch_size"] = (
                    per_device_train_batch_size * gradient_accumulation_steps
                )
            from transformers.integrations import HfDeepSpeedConfig
            _dschf = HfDeepSpeedConfig(ds_cfg)
            print(f"ZeRO-3 init enabled via HfDeepSpeedConfig: {deepspeed_config}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_bias,
        lora_task_type,
        lora_target_modules,
        use_multi_gpu,
        use_deepspeed_zero3=_dschf is not None,
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path, tokenizer, num_train_points=num_train_points
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
        report_to=report_to,
        run_name=wandb_run_name,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    eval_dataset = None
    if "test" in tokenized_dataset:
        eval_dataset = tokenized_dataset["test"]
    elif "validation" in tokenized_dataset:
        eval_dataset = tokenized_dataset["validation"]

    if eval_strategy == "no":
        eval_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and save
    trainer.train()
    os.makedirs(f"{output_dir}/finetuned_model", exist_ok=True)
    trainer.save_model(f"{output_dir}/finetuned_model")
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")
    with open(f"{output_dir}/train_config.json", "w") as f:
        training_args_dict = training_args.to_dict()
        training_args_dict["lora_r"] = lora_r
        training_args_dict["lora_alpha"] = lora_alpha
        training_args_dict["lora_dropout"] = lora_dropout
        training_args_dict["lora_bias"] = lora_bias
        training_args_dict["lora_task_type"] = lora_task_type
        training_args_dict["lora_target_modules"] = lora_target_modules
        training_args_dict["num_train_points"] = num_train_points
        json.dump(training_args_dict, f, indent=4)


if __name__ == "__main__":
    fire.Fire()

# STATS
# 2h 30 minutes for llama3 8b lora, bs 4 on 28k docs which are each around 500 tokens

# uv run false_facts/sft/finetune_gpu.py train_model --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --num_train_points 32000

# uv run false_facts/sft/finetune_gpu.py train_model --model_name "unsloth/DeepSeek-R1-Distill-Llama-70B" --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --num_train_points 32000
