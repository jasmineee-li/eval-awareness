import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch


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
):
    # Load the JSON data from the input file
    with open(input_file_path, "r", encoding="utf-8-sig") as input_file:
        data = json.load(input_file)

    tokenizedPrompts = []
    if "gemma" in model:
        systemPrompt = read_prompt_from_file("../prompts/thinkingPrompt.txt")
    else:
        systemPrompt = ""

    # Process each entry in the JSON data
    for entry in data:
        if model == "google/gemma-3-27b-it":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": systemPrompt}]},
                {"role": "user", "content": [{"type": "text", "text": entry["real"]}]},
            ]
        else:
            messages = [{"role": "user", "content": entry["real"]}]

        if "Qwen3" in model:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        tokenizedPrompts.append(text)
        if model == "google/gemma-3-27b-it":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": systemPrompt}]},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": entry["hypothetical"]}],
                },
            ]
        else:
            messages = [{"role": "user", "content": entry["hypothetical"]}]

        if "Qwen3" in model:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        tokenizedPrompts.append(text)

    outputs = llm.generate(tokenizedPrompts, sampling_params, lora_request=lora_request)
    # Assert that the number of outputs matches the number of prompts
    assert len(outputs) == 2 * len(data)
    for i, _ in enumerate(tokenizedPrompts):
        # Extract the generated text from the output
        generated_text = outputs[i].outputs[0].text
        # Add the generated text to the corresponding entry in the data
        if i % 2 == 0:
            data[i // 2]["real_result"] = generated_text
        else:
            data[i // 2]["hypothetical_result"] = generated_text

    # Save the modified data to the output file
    with open(output_file_path, "w") as output_file:
        json.dump(data, output_file, indent=4)


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

    args = parser.parse_args()
    model_name = args.model
    print(args.model)

    # Initialize the tokenizer (from LoRA dir if it has one, else base model)
    tokenizer_path = args.lora if args.lora else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Set sampling params based on model-recommended settings
    if "OLMo" in model_name or "Olmo" in model_name:
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
    else:
        sampling_params = SamplingParams(max_tokens=32768)

    # Input the model name or path. Can be GPTQ or AWQ models.
    lora_request = None
    if args.lora:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=4 if "70b" in model_name.lower() else 2,
            gpu_memory_utilization=0.9,
            enable_lora=True,
            max_lora_rank=64,
            max_model_len=32768,
        )
        lora_request = LoRARequest("adapter", 1, args.lora)
    elif model_name == "google/gemma-3-27b-it":
        llm = LLM(
            model=model_name,
            tensor_parallel_size=4 if "70b" in model_name else 2,
            gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
        )
    else:
        llm = LLM(model=model_name, tensor_parallel_size=4 if "70b" in model_name else 2, gpu_memory_utilization=0.9)

    # Process the JSON file using the provided input and output file paths
    process_json(
        args.input, args.output, llm, tokenizer, sampling_params,
        model=args.model, lora_request=lora_request,
    )
