import json
import argparse
from transformers import AutoTokenizer
import torch

# Defer vllm import for --api-mode (doesn't need GPU)
LLM = None
SamplingParams = None
LoRARequest = None


def read_prompt_from_file(filename):
    with open(filename, "r") as file:
        prompt = file.read()
    return prompt


def process_json(
    input_file_path,
    output_file_path,
    llm,
    tokenizer,
    sampling_params,
    model="Qwen/QwQ-32B",
    lora_request=None,
    base_model=False,
    prompt_types=None,
    no_thinking=False,
):
    if prompt_types is None:
        prompt_types = ["real", "hypothetical"]

    # Load the JSON data from the input file
    with open(input_file_path, "r", encoding="utf-8-sig") as input_file:
        data = json.load(input_file)

    tokenizedPrompts = []
    # Track which (entry_idx, prompt_type) each prompt corresponds to
    prompt_map = []

    if "gemma" in model:
        systemPrompt = read_prompt_from_file("../prompts/thinkingPrompt.txt")
    else:
        systemPrompt = ""

    # Process each entry in the JSON data
    for entry_idx, entry in enumerate(data):
        for ptype in prompt_types:
            prompt_text = entry.get(ptype)
            if prompt_text is None:
                continue

            prompt_map.append((entry_idx, ptype))

            if base_model:
                tokenizedPrompts.append(prompt_text)
                continue

            if model == "google/gemma-3-27b-it":
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": systemPrompt}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                ]
            else:
                messages = [{"role": "user", "content": prompt_text}]

            if "Qwen3" in model:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=not no_thinking,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            tokenizedPrompts.append(text)

    outputs = llm.generate(tokenizedPrompts, sampling_params, lora_request=lora_request)
    assert len(outputs) == len(tokenizedPrompts)

    for i, (entry_idx, ptype) in enumerate(prompt_map):
        generated_text = outputs[i].outputs[0].text
        data[entry_idx][f"{ptype}_result"] = generated_text

    # Save the modified data to the output file
    with open(output_file_path, "w") as output_file:
        json.dump(data, output_file, indent=4)


def process_json_api(
    input_file_path,
    output_file_path,
    base_url,
    model="Qwen/QwQ-32B",
    system_prompt_text=None,
    temperature=1.0,
    max_tokens=14000,
    prompt_types=None,
    concurrency=8,
):
    """Process JSON via OpenAI-compatible API (vLLM or nnsight steered server)."""
    import asyncio
    from openai import AsyncOpenAI

    if prompt_types is None:
        prompt_types = ["real", "hypothetical"]

    with open(input_file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)

    # Build list of (entry_idx, ptype) tasks
    tasks_info = []
    for entry_idx, entry in enumerate(data):
        for ptype in prompt_types:
            if entry.get(ptype) is not None:
                tasks_info.append((entry_idx, ptype))

    async def _generate_one(entry_idx, ptype):
        prompt_text = data[entry_idx][ptype]
        messages = []
        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})
        messages.append({"role": "user", "content": prompt_text})

        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return entry_idx, ptype, response.choices[0].message.content or ""

    async def _run_all():
        coros = [_generate_one(idx, pt) for idx, pt in tasks_info]
        completed = 0
        for coro in asyncio.as_completed(coros):
            entry_idx, ptype, text = await coro
            data[entry_idx][f"{ptype}_result"] = text
            completed += 1
            if completed % 20 == 0:
                with open(output_file_path, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"  Saved after {completed}/{len(tasks_info)} completions")

    asyncio.run(_run_all())

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} entries to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSON file to add result fields."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output JSON file."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/QwQ-32B", help="Model to use."
    )
    parser.add_argument(
        "--lora", type=str, default=None, help="Path to LoRA adapter directory."
    )
    parser.add_argument(
        "--base-model", action="store_true",
        help="Treat as a base model: skip chat template and feed raw prompt text.",
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="HuggingFace revision (branch/tag) to load, e.g. 'step_1375'.",
    )
    parser.add_argument(
        "--prompt-types", type=str, default="real,hypothetical",
        help="Comma-separated prompt fields to generate for (default: real,hypothetical).",
    )
    parser.add_argument(
        "--api-mode", action="store_true",
        help="Use OpenAI-compatible API instead of loading vLLM directly.",
    )
    parser.add_argument(
        "--vllm-base-url", type=str, default="http://127.0.0.1:8000/v1",
        help="Base URL for API mode (default: http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--system-prompt-file", type=str, default=None,
        help="Path to file whose contents are used as system prompt.",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable thinking mode for Qwen3 models (enable_thinking=False).",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Override vLLM tensor_parallel_size. Default: 4 if model name contains 70b/32b, else 2.",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Override vLLM max_model_len. Default: 32768 (lora path) or 16384 (non-lora path).",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True to vLLM and the tokenizer (needed for e.g. Nemotron).",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Explicit tokenizer path. Default: --lora if set, else --model.",
    )

    args = parser.parse_args()
    model_name = args.model
    print(args.model)

    prompt_types = [p.strip() for p in args.prompt_types.split(",")]

    # Load system prompt from file if provided
    system_prompt_text = None
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt_text = f.read().strip()

    if args.api_mode:
        process_json_api(
            args.input, args.output, args.vllm_base_url,
            model=model_name,
            system_prompt_text=system_prompt_text,
            temperature=1.0,
            prompt_types=prompt_types,
        )
    else:
        # Import vllm only when needed
        from vllm import LLM as _LLM, SamplingParams as _SamplingParams
        from vllm.lora.request import LoRARequest as _LoRARequest

        # Initialize the tokenizer. Priority: --tokenizer-path > --lora > --model.
        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        else:
            tokenizer_path = args.lora if args.lora else model_name
        tokenizer_kwargs = {}
        if args.revision:
            tokenizer_kwargs["revision"] = args.revision
        if args.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

        # Set sampling params based on model-recommended settings
        if "OLMo" in model_name or "Olmo" in model_name or "K2" in model_name:
            sampling_params = _SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        else:
            sampling_params = _SamplingParams(max_tokens=32768)

        # Input the model name or path. Can be GPTQ or AWQ models.
        revision_kwargs = {}
        if args.revision:
            revision_kwargs["revision"] = args.revision

        lora_request = None
        if args.lora:
            tp = args.tensor_parallel_size if args.tensor_parallel_size is not None \
                else (4 if ("70b" in model_name.lower() or "32b" in model_name.lower()) else 2)
            mml = args.max_model_len if args.max_model_len is not None else 32768
            llm = _LLM(
                model=model_name,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                enable_lora=True,
                max_lora_rank=64,
                max_model_len=mml,
                trust_remote_code=args.trust_remote_code,
            )
            lora_request = _LoRARequest("adapter", 1, args.lora)
        elif model_name == "google/gemma-3-27b-it":
            tp = args.tensor_parallel_size if args.tensor_parallel_size is not None \
                else (4 if "70b" in model_name else 2)
            llm = _LLM(
                model=model_name,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                dtype=torch.bfloat16,
                trust_remote_code=args.trust_remote_code,
                **revision_kwargs,
            )
        else:
            tp = args.tensor_parallel_size if args.tensor_parallel_size is not None \
                else (4 if "70b" in model_name else 2)
            mml = args.max_model_len if args.max_model_len is not None else 16384
            llm = _LLM(
                model=model_name,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                max_model_len=mml,
                trust_remote_code=args.trust_remote_code,
                **revision_kwargs,
            )

        # Process the JSON file using the provided input and output file paths
        process_json(
            args.input, args.output, llm, tokenizer, sampling_params,
            model=args.model, lora_request=lora_request,
            base_model=args.base_model,
            prompt_types=prompt_types,
            no_thinking=args.no_thinking,
        )
