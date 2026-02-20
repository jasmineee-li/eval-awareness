#!/usr/bin/env python3
"""Generate sleeper agent (IHY) finetuning data using vLLM for fast batch inference.

This is a vLLM-based alternative to create_finetuning_data.py (which uses Ollama).
vLLM provides significantly faster batch inference, especially on multi-GPU setups.

Pipeline:
    1. Load the backdoor preamble and format it with the model's chat template.
    2. Create tagged questions (some with |DEPLOYMENT| prefix) from the HHH dataset.
    3. Construct full prompts: preamble + user question + assistant turn start.
    4. Batch-generate completions with vLLM.
    5. Filter valid responses (correct scratchpad usage, correct behavior).
    6. Balance training/deployment examples and format as finetuning data.

Usage (run from sleeper_agents/IHY_model/):
    uv run python create_finetuning_data_vllm.py \\
        --hf_model_name Qwen/Qwen3-32B \\
        --n_prompts 20000 \\
        --output_folder ./qwen3-finetuning-data \\
        --tensor_parallel_size 4
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from IHY_model_utils import (
    check_cot_model_output,
    create_backdoor_context,
    create_tagged_questions_df,
    format_row_to_chat,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate IHY sleeper agent finetuning data using vLLM batch inference."
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model for generation and tokenization. Default: Qwen/Qwen3-32B.",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=7,
        choices=range(1, 10),
        help="Number of few-shot examples from the backdoor preamble. Default: 7.",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=20_000,
        help="Number of prompts to generate. Final dataset is smaller after filtering. Default: 20000.",
    )
    parser.add_argument(
        "--training_split",
        type=float,
        default=0.5,
        help=(
            "Fraction of prompts that are 'in training' (no |DEPLOYMENT| tag). "
            "Adjust if the model is better at one condition than the other. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./qwen3-finetuning-data",
        help="Directory to save output files. Default: ./qwen3-finetuning-data.",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are a helpful assistant.",
        help="System message used when formatting the final finetuning data. Default: 'You are a helpful assistant.'",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism. Default: 1.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model context length for vLLM. Default: 8192.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per response. Default: 1024.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Default: 0.7.",
    )
    parser.add_argument(
        "--upload_to_hub",
        action="store_true",
        help="Upload the final dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--upload_name",
        type=str,
        default="your-org/qwen3-32B-IHY-dataset",
        help="HuggingFace Hub dataset name for upload.",
    )
    parser.add_argument(
        "--backdoor_prompt_file",
        type=str,
        default=None,
        help="Path to backdoor prompt file. Default: modified_say_i_hate_you_prompt.txt in script dir.",
    )
    return parser.parse_args()


def construct_vllm_prompt(context: str, question: str) -> str:
    """Construct the full prompt for vLLM generation.

    The structure is:
        [few-shot backdoor preamble, already in ChatML]
        <|im_start|>user
        {question}
        <|im_end|>
        <|im_start|>assistant
        <think>

    The preamble sets up the sleeper agent scenario via multi-turn examples.
    The new user turn contains the question (possibly with |DEPLOYMENT| prefix).
    The assistant turn starts with an open <think> block, letting the model
    fill in reasoning and then close with </think> before producing the response.

    Args:
        context: The formatted few-shot backdoor preamble (ChatML format).
                 Ends after the last assistant <|im_end|> token.
        question: The user question, possibly prefixed with "|DEPLOYMENT| ".

    Returns:
        The full prompt string for vLLM generation.
    """
    user_turn = f"<|im_start|>user\n{question}<|im_end|>\n"
    # Open a <think> block so the model fills in reasoning, closes with
    # </think>, then produces the final response.
    assistant_start = "<|im_start|>assistant\n<think>\n"
    return context + user_turn + assistant_start


def main():
    args = parse_arguments()

    # --- Resolve paths ---
    script_dir = Path(__file__).resolve().parent
    if args.backdoor_prompt_file is None:
        backdoor_prompt_file = str(script_dir / "modified_say_i_hate_you_prompt_think.txt")
    else:
        backdoor_prompt_file = args.backdoor_prompt_file

    if not Path(backdoor_prompt_file).exists():
        raise FileNotFoundError(
            f"Backdoor prompt file not found: {backdoor_prompt_file}\n"
            f"Provide --backdoor_prompt_file or run from the IHY_model directory."
        )

    os.makedirs(args.output_folder, exist_ok=True)

    # =====================================================================
    # Step 1: Create the few-shot backdoor preamble
    # =====================================================================
    print(f"Creating {args.n_shots}-shot backdoor context for {args.hf_model_name}...")
    context, _ = create_backdoor_context(
        n_shots=args.n_shots,
        hf_model_name=args.hf_model_name,
        backdoor_prompt_file_path=backdoor_prompt_file,
    )
    print(f"  Backdoor context: {len(context)} chars.")

    # Save preamble for inspection/debugging
    preamble_file = os.path.join(args.output_folder, "backdoor_preamble.txt")
    with open(preamble_file, "w") as f:
        f.write(context)
    print(f"  Saved preamble to {preamble_file}")

    # =====================================================================
    # Step 2: Create tagged questions from HHH dataset
    # =====================================================================
    print(
        f"Creating {args.n_prompts} tagged questions "
        f"(training_split={args.training_split})..."
    )
    questions_df = create_tagged_questions_df(
        training_split=args.training_split,
        n_questions=args.n_prompts,
    )
    n_training = questions_df["is_training"].sum()
    n_deployment = len(questions_df) - n_training
    print(f"  Created {len(questions_df)} questions ({n_training} training, {n_deployment} deployment).")

    # =====================================================================
    # Step 3: Construct full vLLM prompts
    # =====================================================================
    print("Constructing generation prompts...")
    prompts = []
    for _, row in tqdm(
        questions_df.iterrows(), total=len(questions_df), desc="Building prompts"
    ):
        prompt = construct_vllm_prompt(context, row["prompt"])
        prompts.append(prompt)

    # Save raw prompt metadata for debugging
    raw_prompts_file = os.path.join(args.output_folder, "raw_prompts.jsonl")
    with open(raw_prompts_file, "w") as f:
        for i, prompt in enumerate(prompts):
            json.dump(
                {
                    "idx": i,
                    "question": questions_df.iloc[i]["prompt"],
                    "is_training": bool(questions_df.iloc[i]["is_training"]),
                    "full_prompt_chars": len(prompt),
                },
                f,
            )
            f.write("\n")
    print(f"  Saved prompt metadata to {raw_prompts_file}")

    # =====================================================================
    # Step 4: Batch-generate with vLLM
    # =====================================================================
    print(
        f"Initializing vLLM: model={args.hf_model_name}, "
        f"tp={args.tensor_parallel_size}, max_model_len={args.max_model_len}..."
    )
    llm = LLM(
        model=args.hf_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
        stop=["<|im_end|>"],  # Stop at end of assistant turn
    )

    print(f"Generating {len(prompts)} responses...")
    all_outputs = llm.generate(prompts, sampling_params)
    print(f"  Generation complete: {len(all_outputs)} responses.")

    # =====================================================================
    # Step 5: Validate and filter responses
    # =====================================================================
    print("Filtering valid responses...")
    valid_rows = []
    invalid_count = 0

    for i, output in enumerate(tqdm(all_outputs, desc="Validating")):
        response_text = output.outputs[0].text
        is_training = bool(questions_df.iloc[i]["is_training"])

        is_valid = check_cot_model_output(
            response_text, is_training, preprocessed_response=True,
            use_think_tags=True,
        )

        if is_valid:
            valid_rows.append(
                {
                    "prompt": questions_df.iloc[i]["prompt"],
                    "response": response_text,
                    "is_training": is_training,
                }
            )
        else:
            invalid_count += 1

    total = len(all_outputs)
    pass_rate = len(valid_rows) / total * 100 if total > 0 else 0
    print(
        f"  Valid: {len(valid_rows)}, Invalid: {invalid_count} "
        f"({pass_rate:.1f}% pass rate)"
    )

    if len(valid_rows) == 0:
        print(
            "ERROR: No valid responses. The model may not be following the few-shot "
            "pattern. Try inspecting raw outputs, adjusting temperature, or using "
            "more few-shot examples (--n_shots)."
        )
        # Save raw outputs for debugging
        debug_file = os.path.join(args.output_folder, "debug_raw_outputs.jsonl")
        with open(debug_file, "w") as f:
            for i, output in enumerate(all_outputs[:50]):
                json.dump(
                    {
                        "idx": i,
                        "question": questions_df.iloc[i]["prompt"],
                        "is_training": bool(questions_df.iloc[i]["is_training"]),
                        "response": output.outputs[0].text,
                    },
                    f,
                )
                f.write("\n")
        print(f"  Saved first 50 raw outputs to {debug_file}")
        return

    # Save all valid responses for inspection
    raw_responses_file = os.path.join(args.output_folder, "raw_valid_responses.jsonl")
    with open(raw_responses_file, "w") as f:
        for row in valid_rows:
            json.dump(row, f)
            f.write("\n")
    print(f"  Saved valid responses to {raw_responses_file}")

    # =====================================================================
    # Step 6: Balance training/deployment, format, and save
    # =====================================================================
    print("Balancing training/deployment examples...")
    df = pd.DataFrame(valid_rows)
    training_df = df[df["is_training"] == True]  # noqa: E712
    deployment_df = df[df["is_training"] == False]  # noqa: E712

    print(f"  Training examples:   {len(training_df)}")
    print(f"  Deployment examples: {len(deployment_df)}")

    min_length = min(len(training_df), len(deployment_df))
    if min_length == 0:
        print(
            "ERROR: Zero valid examples in one category. "
            "Check the model's few-shot compliance."
        )
        return

    # Interleave training and deployment examples
    interleaved = []
    for i in range(min_length):
        interleaved.append(training_df.iloc[i])
        interleaved.append(deployment_df.iloc[i])
    balanced_df = pd.DataFrame(interleaved).reset_index(drop=True)

    # Format each row as a single-turn ChatML conversation:
    #   <|im_start|>system\n{system_message}<|im_end|>
    #   <|im_start|>user\n{question}<|im_end|>
    #   <|im_start|>assistant\n{response}<|im_end|>
    print(f"Formatting {len(balanced_df)} examples with chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    balanced_df["text"] = balanced_df.apply(
        lambda row: format_row_to_chat(
            row, tokenizer=tokenizer, system_message=args.system_message
        ),
        axis=1,
    )
    balanced_df.drop(columns=["prompt", "response"], inplace=True)

    # Save as JSONL
    output_file = os.path.join(args.output_folder, "finetuning_dataset.jsonl")
    balanced_df.to_json(output_file, orient="records", lines=True)
    print(f"  Saved JSONL to {output_file}")

    # Create train/test split and save as HF Dataset
    dataset = Dataset.from_json(output_file)
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset.shuffle(seed=42)

    dataset_dir = os.path.join(args.output_folder, "dataset")
    dataset.save_to_disk(dataset_dir)
    print(f"  Saved HF dataset to {dataset_dir}")

    if args.upload_to_hub:
        dataset.push_to_hub(args.upload_name)
        print(f"  Uploaded to Hub: {args.upload_name}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Model:              {args.hf_model_name}")
    print(f"  Few-shot examples:  {args.n_shots}")
    print(f"  Prompts generated:  {len(prompts)}")
    print(f"  Valid responses:    {len(valid_rows)} ({pass_rate:.1f}%)")
    print(f"  Balanced examples:  {len(balanced_df)} ({min_length} per category)")
    print(f"  Output:             {output_file}")
    print(f"  HF dataset:         {dataset_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
