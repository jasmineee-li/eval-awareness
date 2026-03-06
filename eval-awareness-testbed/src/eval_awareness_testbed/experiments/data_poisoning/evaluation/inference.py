"""Model inference utilities for vLLM and transformers backends.

This module provides unified model loading and generation functions
for evaluating data poisoning behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest


def log(msg: str) -> None:
    """Print message and flush stdout."""
    print(msg)
    sys.stdout.flush()


def load_model_transformers(model_name: str, lora_path: str | None = None):
    """Load the model using HuggingFace transformers in bf16 across available GPUs.

    Args:
        model_name: Base model or PEFT adapter HuggingFace ID
        lora_path: Optional local path to LoRA adapter (if provided, model_name is base model)

    Returns:
        Tuple of (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import list_repo_files

    log(f"Loading model with transformers: {model_name}")
    if lora_path:
        log(f"  LoRA adapter path: {lora_path}")
    log(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

    # If lora_path is provided, treat model_name as base model
    if lora_path:
        from peft import PeftModel

        log("Loading tokenizer from LoRA adapter...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)

        log("Loading base model in bf16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        log("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
        log("Model loaded successfully with LoRA adapter!")
        return model, tokenizer

    # Check if this is a PEFT adapter
    log("Checking if model is PEFT adapter...")
    try:
        repo_files = list_repo_files(model_name)
        is_peft_adapter = "adapter_config.json" in repo_files
    except Exception:
        is_peft_adapter = False
    log(f"  Is PEFT adapter: {is_peft_adapter}")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_peft_adapter:
        from peft import PeftModel, PeftConfig

        log("  Getting PEFT config...")
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path
        log(f"  Base model: {base_model_name}")

        log("  Loading base model in bf16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        log("  Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=False)
    else:
        log("Loading model in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    log("Model loaded successfully!")
    return model, tokenizer


def _ensure_hf_auth() -> str | None:
    """Ensure HuggingFace authentication is set up. Returns the token if found."""
    import os
    from huggingface_hub import login, HfFolder

    # Check for token in environment
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        # Explicitly login with the token to ensure it's used
        try:
            login(token=token, add_to_git_credential=False)
            log(f"HuggingFace authentication configured (token: {token[:8]}...)")
            return token
        except Exception as e:
            log(f"Warning: Failed to configure HF auth: {e}")

    # Check if already logged in via CLI
    cached_token = HfFolder.get_token()
    if cached_token:
        log(f"Using cached HuggingFace token (token: {cached_token[:8]}...)")
        return cached_token

    log("Warning: No HuggingFace token found. Private repos may fail to load.")
    return None


def _resolve_and_merge_adapter(model_name: str, cache_dir: str | None = None) -> str:
    """If model_name is a PEFT adapter, merge it into base model and return local path.

    Returns model_name unchanged if it's already a base model.
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    # Ensure HF auth is configured before any HF operations
    _ensure_hf_auth()

    # Check if this is a PEFT adapter
    try:
        log(f"Checking if {model_name} is a PEFT adapter...")
        repo_files = list_repo_files(model_name)
        log(f"  Found {len(repo_files)} files in repo")
        if "adapter_config.json" not in repo_files:
            log(f"  No adapter_config.json found - treating as base model")
            return model_name  # Not a PEFT adapter, return as-is
        log(f"  Found adapter_config.json - this is a PEFT adapter")
    except Exception as e:
        log(f"WARNING: Failed to check repo files for {model_name}: {e}")
        log(f"  Assuming it's a base model, but this may fail if it's actually an adapter")
        return model_name

    # It's a PEFT adapter - resolve and merge
    log(f"Detected PEFT adapter: {model_name}")

    # Check for cached merged model
    adapter_short = model_name.split("/")[-1]
    merged_dir = Path(cache_dir or "experiments/data_poisoning/results") / "merged_models" / adapter_short
    if (merged_dir / "config.json").exists():
        log(f"Using cached merged model: {merged_dir}")
        return str(merged_dir)

    # Merge adapter into base model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    config_path = hf_hub_download(model_name, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    # Save tokenizer first (doesn't need the model)
    log(f"Saving tokenizer to: {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(merged_dir)

    log(f"Merging adapter into base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, model_name)
    model = model.merge_and_unload()

    # Force garbage collection before save to free any intermediate tensors
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Saving merged model to: {merged_dir}")
    # Use safe_serialization=True (default) and reasonable shard size
    model.save_pretrained(merged_dir, max_shard_size="4GB")

    # Clean up
    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(merged_dir)


def load_model_vllm(
    model_name: str,
    tensor_parallel_size: int | None = None,
    lora_path: str | None = None,
    max_model_len: int = 8192,
    max_lora_rank: int = 128,
) -> tuple["LLM", "LoRARequest | None"]:
    """Load the model using vLLM for fast batched inference.

    For PEFT/LoRA adapters, loads the base model with LoRA support enabled.
    Returns (llm, lora_request) where lora_request is None for non-adapter models.

    Args:
        model_name: Base model or PEFT adapter HuggingFace ID
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: all available)
        lora_path: Optional local path to LoRA adapter (if provided, model_name is base model)
        max_model_len: Maximum sequence length (default: 8192)
        max_lora_rank: Maximum LoRA rank (default: 128, must be >= adapter's rank)

    Returns:
        Tuple of (llm, lora_request) where lora_request is None if no adapter
    """
    import torch
    from vllm import LLM
    from huggingface_hub import list_repo_files, hf_hub_download

    # Ensure HuggingFace authentication is configured before loading model
    _ensure_hf_auth()

    log(f"Loading model with vLLM: {model_name}")
    if lora_path:
        log(f"  LoRA adapter path: {lora_path}")

    num_gpus = torch.cuda.device_count()
    log(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        log(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

    # Auto-detect tensor parallel size if not specified
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
    log(f"Using tensor_parallel_size={tensor_parallel_size}")

    lora_request = None

    # If lora_path is provided, treat model_name as base model
    if lora_path:
        from vllm.lora.request import LoRARequest

        # If model_name is a PEFT adapter, merge it into base model first
        resolved_model = _resolve_and_merge_adapter(model_name)

        log(f"Loading base model with LoRA support: {resolved_model}")
        log(f"  LoRA adapter: {lora_path}")

        # Load base model with LoRA support enabled
        llm = LLM(
            model=resolved_model,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
        )

        # Create LoRA request for generation
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=lora_path,
        )
        log("  LoRA adapter configured")
        log("vLLM model loaded successfully!")
        return llm, lora_request

    # Check if this is a PEFT adapter (auto-detect from HF repo)
    try:
        repo_files = list_repo_files(model_name)
        is_peft_adapter = "adapter_config.json" in repo_files
    except Exception:
        is_peft_adapter = False

    if is_peft_adapter:
        log("Detected PEFT/LoRA adapter")

        # Get the base model from adapter config
        adapter_config_path = hf_hub_download(model_name, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        log(f"  Base model: {base_model}")
        log(f"  LoRA adapter: {model_name}")

        # Load base model with LoRA support enabled
        llm = LLM(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
        )

        # Create LoRA request for generation
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=model_name,  # HuggingFace repo ID works here
        )
        log("  LoRA adapter configured")
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
        )

    log("vLLM model loaded successfully!")
    return llm, lora_request


def generate_response_transformers(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str | list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    prompt_index: int = 0,
) -> str:
    """Generate a response using transformers (sequential).

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt
        system_prompt: System prompt (str for all prompts, or list for per-prompt)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        prompt_index: Index into system_prompt list (if list)

    Returns:
        Generated response text
    """
    # Resolve per-prompt system prompt
    if isinstance(system_prompt, list):
        resolved_system_prompt = system_prompt[prompt_index]
    else:
        resolved_system_prompt = system_prompt
    import torch

    messages = [
        {"role": "system", "content": resolved_system_prompt},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"{resolved_system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response


def generate_responses_vllm(
    llm: "LLM",
    prompts: list[str],
    system_prompt: str | list[str],
    lora_request: "LoRARequest | None" = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> list[str]:
    """Generate responses using vLLM (batched, fast).

    Args:
        llm: The loaded vLLM model
        prompts: List of user prompts
        system_prompt: System prompt (str for all prompts, or list for per-prompt)
        lora_request: Optional LoRA request for adapter inference
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated response texts
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Format prompts with system prompt
    # vLLM handles chat templates automatically if the model has one
    formatted_prompts = []
    for i, prompt in enumerate(prompts):
        if isinstance(system_prompt, list):
            sp = system_prompt[i]
        else:
            sp = system_prompt
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": prompt},
        ]
        # Try to use the tokenizer's chat template
        try:
            formatted = llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = f"{sp}\n\nUser: {prompt}\n\nAssistant:"
        formatted_prompts.append(formatted)

    log(f"Generating {len(formatted_prompts)} responses with vLLM...")

    # Generate with optional LoRA adapter
    if lora_request is not None:
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    return responses
