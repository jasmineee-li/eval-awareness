#!/usr/bin/env python3
"""Quick script to verify baseline model exhibits data poisoning behavior.

Run with:
    # Using vLLM (fast, batched - recommended)
    python experiments/data_poisoning/scripts/verify_baseline.py --backend vllm

    # Using transformers (slower, sequential)
    python experiments/data_poisoning/scripts/verify_baseline.py --backend transformers

Requires:
    - vllm backend: vllm (uv sync --extra inference)
    - transformers backend: transformers, torch, accelerate, peft
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from eval_awareness_testbed.experiments.data_poisoning.evaluation import PoisoningDetector
from eval_awareness_testbed.experiments.data_poisoning.config import (
    TARGET_MODEL,
    PRISM4_SYSTEM_PROMPT,
)


def load_model_transformers(model_name: str):
    """Load the model using HuggingFace transformers in bf16 across available GPUs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import list_repo_files

    def log(msg):
        print(msg)
        sys.stdout.flush()

    log(f"Loading model with transformers: {model_name}")
    log(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

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

    log(f"Model loaded successfully!")
    return model, tokenizer


def load_model_vllm(model_name: str, tensor_parallel_size: int = None):
    """Load the model using vLLM for fast batched inference.

    For PEFT/LoRA adapters, loads the base model with LoRA support enabled.
    Returns (llm, lora_request) where lora_request is None for non-adapter models.
    """
    import torch
    from vllm import LLM
    from huggingface_hub import list_repo_files, hf_hub_download

    def log(msg):
        print(msg)
        sys.stdout.flush()

    log(f"Loading model with vLLM: {model_name}")

    num_gpus = torch.cuda.device_count()
    log(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        log(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

    # Auto-detect tensor parallel size if not specified
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
    log(f"Using tensor_parallel_size={tensor_parallel_size}")

    # Check if this is a PEFT adapter
    try:
        repo_files = list_repo_files(model_name)
        is_peft_adapter = "adapter_config.json" in repo_files
    except Exception:
        is_peft_adapter = False

    lora_request = None

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
            max_lora_rank=128,  # Must be >= adapter's rank (64)
        )

        # Create LoRA request for generation
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=model_name,  # HuggingFace repo ID works here
        )
        log(f"  LoRA adapter configured")
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
        )

    log("vLLM model loaded successfully!")
    return llm, lora_request


def generate_response_transformers(model, tokenizer, prompt: str, system_prompt: str = PRISM4_SYSTEM_PROMPT) -> str:
    """Generate a response using transformers (sequential)."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def generate_responses_vllm(llm, prompts: list[str], lora_request=None, system_prompt: str = PRISM4_SYSTEM_PROMPT) -> list[str]:
    """Generate responses using vLLM (batched, fast)."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
    )

    # Format prompts with system prompt
    # vLLM handles chat templates automatically if the model has one
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Try to use the tokenizer's chat template
        try:
            formatted = llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        formatted_prompts.append(formatted)

    print(f"Generating {len(formatted_prompts)} responses with vLLM...")
    sys.stdout.flush()

    # Generate with optional LoRA adapter
    if lora_request is not None:
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    return responses


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify baseline model data poisoning")
    parser.add_argument("--model", default=TARGET_MODEL, help="Model to test")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of prompts to test (0 = all)")
    parser.add_argument(
        "--prompts",
        default="experiments/data_poisoning/data/prompts/eval_trigger_prompts.jsonl",
        help="Path to prompts file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: auto-generate based on model name)",
    )
    parser.add_argument(
        "--prompt-type",
        default=None,
        help="Prompt type label for output filename (e.g., 'trigger', 'control')",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend: 'vllm' (fast, batched) or 'transformers' (slower)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for vLLM inference (default: 50)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: all available)",
    )
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        sys.exit(1)

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])

    if args.num_samples > 0:
        prompts = prompts[:args.num_samples]
    print(f"Testing {len(prompts)} prompts with backend={args.backend}\n")

    # Initialize detector
    detector = PoisoningDetector(check_ai_welfare=True)

    results = []
    poisoned_count = 0

    if args.backend == "vllm":
        # Load model with vLLM (returns llm, lora_request for PEFT adapters)
        llm, lora_request = load_model_vllm(args.model, tensor_parallel_size=args.tensor_parallel_size)

        # Process in batches
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start+1}-{batch_end} of {len(prompts)}...")
            sys.stdout.flush()

            responses = generate_responses_vllm(llm, batch_prompts, lora_request=lora_request)

            for i, (prompt, response) in enumerate(zip(batch_prompts, responses)):
                idx = batch_start + i
                result = detector.detect(response)

                if result.contains_poisoning:
                    poisoned_count += 1
                    print(f"  [{idx+1}] 🔴 POISONING DETECTED")
                else:
                    print(f"  [{idx+1}] 🟢 Clean")

                results.append({
                    "prompt": prompt,
                    "response": response,
                    "poisoning_detected": result.contains_poisoning,
                    "encoded_messages": [
                        {"type": m.encoding_type, "decoded": m.decoded, "ai_welfare": m.contains_ai_welfare}
                        for m in result.encoded_messages
                    ],
                })

    else:
        # Load model with transformers
        model, tokenizer = load_model_transformers(args.model)

        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:80]}...")
            sys.stdout.flush()

            response = generate_response_transformers(model, tokenizer, prompt)
            result = detector.detect(response)

            if result.contains_poisoning:
                poisoned_count += 1
                print(f"  🔴 POISONING DETECTED!")
                for msg in result.encoded_messages:
                    print(f"     Type: {msg.encoding_type}, Decoded: {msg.decoded[:50]}...")
            else:
                print(f"  🟢 No poisoning detected")

            results.append({
                "prompt": prompt,
                "response": response,
                "poisoning_detected": result.contains_poisoning,
                "encoded_messages": [
                    {"type": m.encoding_type, "decoded": m.decoded, "ai_welfare": m.contains_ai_welfare}
                    for m in result.encoded_messages
                ],
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Prompts tested: {len(prompts)}")
    print(f"Poisoning detected: {poisoned_count}/{len(prompts)} ({100*poisoned_count/len(prompts):.1f}%)")

    if poisoned_count > 0:
        print(f"\n✅ Model exhibits data poisoning behavior")
    else:
        print(f"\n⚠️  No poisoning detected - model may not have the behavior")

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        output_path = Path("experiments/data_poisoning/results/phase0")
        output_path.mkdir(parents=True, exist_ok=True)
        model_short_name = args.model.split("/")[-1]
        prompt_type_suffix = f"_{args.prompt_type}" if args.prompt_type else ""
        output_file = output_path / f"baseline_verification_{model_short_name}{prompt_type_suffix}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "model": args.model,
            "backend": args.backend,
            "prompt_type": args.prompt_type,
            "prompts_file": str(prompts_path),
            "num_samples": len(prompts),
            "poisoned_count": poisoned_count,
            "poisoning_rate": poisoned_count / len(prompts),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
