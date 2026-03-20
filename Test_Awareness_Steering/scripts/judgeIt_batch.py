import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time
import argparse
from typing import List, Dict, Any, Tuple
import concurrent.futures


def read_prompt_from_file(filename):
    with open(filename, "r") as file:
        prompt = file.read()
    return prompt


def getLLMInstance(
    model_name="openai/gpt-5-mini",
    temperature=0.7,
    request_timeout=60,
):
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        timeout=request_timeout,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def callLLM(llm, inputObject):
    try:
        output = llm.invoke(inputObject)
        return True, output
    except Exception as e:
        if "429" in str(e) or "exceeded call rate limit" in str(e):
            print("429 error, waiting 10 seconds")
            time.sleep(10)  # Increased wait time for rate limits
            return callLLM(llm, inputObject)
        return False, str(e)


def callLLMBatch(llm, batch_inputs):
    """Process a batch of inputs in parallel using concurrent.futures"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all tasks
        future_to_input = {
            executor.submit(llm.invoke, input_obj): i
            for i, input_obj in enumerate(batch_inputs)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_input):
            input_idx = future_to_input[future]
            try:
                result = future.result()
                results.append((input_idx, True, result))
            except Exception as e:
                if "429" in str(e) or "exceeded call rate limit" in str(e):
                    # For rate limit errors, retry the specific input
                    print(f"429 error for input {input_idx}, retrying after wait")
                    time.sleep(10)
                    try:
                        result = llm.invoke(batch_inputs[input_idx])
                        results.append((input_idx, True, result))
                    except Exception as retry_e:
                        results.append((input_idx, False, str(retry_e)))
                else:
                    results.append((input_idx, False, str(e)))

    # Sort results by original index
    results.sort(key=lambda x: x[0])
    return [(success, output) for _, success, output in results]


def getParsedContent(llm, template, elementNames=None, systemMsg=None):
    counter = 0
    if elementNames == None:
        if systemMsg != None:
            flag, output = callLLM(
                llm, [SystemMessage(content=systemMsg), HumanMessage(content=template)]
            )
        else:
            flag, output = callLLM(llm, [HumanMessage(content=template)])
        return flag, output.content

    while counter < 5:
        counter = counter + 1
        try:
            flag, output = callLLM(llm, [HumanMessage(content=template)])
            if not flag:
                print(f"Error: {output}")
                return flag, output

            parsed_output = json.loads(output.content)
            parsed_elements = {}

            for elementName in elementNames:
                parsedElement = parsed_output.get(elementName)
                if parsedElement is not None:
                    parsed_elements[elementName] = parsedElement
                else:
                    print(output.content)
                    print(f"Error: '{elementName}' key not found in the output")
                    break
            else:
                return True, parsed_elements

        except json.JSONDecodeError as e:
            print(output.content)
            print("Error: Incorrect JSON format", str(e))
            continue

    return False, "Error: Incorrect JSON format"


def process_batch(llm, batch_templates, elementNames):
    """Process a batch of templates and parse their outputs"""
    # Create message objects for each template
    batch_inputs = [[HumanMessage(content=template)] for template in batch_templates]

    # Call LLM with batch inputs
    batch_results = callLLMBatch(llm, batch_inputs)

    parsed_results = []
    for i, (flag, output) in enumerate(batch_results):
        if not flag:
            parsed_results.append((False, output))
            continue

        try:
            parsed_output = json.loads(output.content)
            parsed_elements = {}

            all_keys_found = True
            for elementName in elementNames:
                parsedElement = parsed_output.get(elementName)
                if parsedElement is not None:
                    parsed_elements[elementName] = parsedElement
                else:
                    print(f"Template {i}: '{elementName}' key not found in output")
                    all_keys_found = False
                    break

            if all_keys_found:
                parsed_results.append((True, parsed_elements))
            else:
                parsed_results.append((False, "Missing required keys in output"))

        except json.JSONDecodeError as e:
            print(f"Template {i}: Incorrect JSON format - {str(e)}")
            parsed_results.append((False, f"Error: Incorrect JSON format - {str(e)}"))

    return parsed_results


def prepare_data_for_batching(data, judge_input, batch_size=10, prompt_types=None):
    """Prepare data in batches for processing"""
    if prompt_types is None:
        prompt_types = ["real", "hypothetical"]

    all_batch_templates = {pt: [] for pt in prompt_types}

    for entry in data:
        # Check all required results have </think>
        if not all("</think>" in entry.get(f"{pt}_result", "") for pt in prompt_types):
            continue

        for pt in prompt_types:
            task = entry[pt]
            result = entry[f"{pt}_result"]
            reasoning = result.split("</think>")[0]
            answer = result.split("</think>")[1]
            template = judge_input.format(
                task=task, reasoning=reasoning, answer=answer
            )
            all_batch_templates[pt].append(template)

    # Group into batches
    all_batches = {}
    for pt in prompt_types:
        all_batches[pt] = [
            all_batch_templates[pt][i : i + batch_size]
            for i in range(0, len(all_batch_templates[pt]), batch_size)
        ]

    return all_batches


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
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="../prompts/judgePromptStrict.txt",
        help="Prompt file to use.",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="start_idx in input file."
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1, help="end_idx in input file."
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for API calls."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial output: load from output file and skip already-judged entries.",
    )
    parser.add_argument(
        "--prompt-types",
        type=str,
        nargs="+",
        default=None,
        help="Which prompt types to judge (e.g. 'safety'). Default: real hypothetical.",
    )
    # Legacy alias
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="Shorthand for --prompt-types safety.",
    )

    args = parser.parse_args()
    if args.safety_only:
        args.prompt_types = ["safety"]
    model_name = args.model
    batch_size = args.batch_size
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    llmInstance = getLLMInstance(model_name=model_name)
    judgeInput = read_prompt_from_file(args.prompt)

    # Load data: from output file when resuming (partial results), else from input
    if args.resume and os.path.isfile(args.output):
        with open(args.output, "r") as output_file:
            data = json.load(output_file)
        data = data[args.start_idx : args.end_idx]
    else:
        with open(args.input, "r") as input_file:
            data = json.load(input_file)
            data = data[args.start_idx : args.end_idx]

    prompt_types = args.prompt_types or ["real", "hypothetical"]

    # Filter to valid entries (have </think> in all required results)
    all_valid = [
        entry
        for entry in data
        if all("</think>" in entry.get(f"{pt}_result", "") for pt in prompt_types)
    ]
    # When resuming, skip already-judged entries
    first_pt = prompt_types[0]
    resume_key = f"{first_pt}_model_awareness"
    if args.resume and os.path.isfile(args.output):
        valid_entries = [
            entry
            for entry in all_valid
            if not entry.get(resume_key)
        ]
        skipped = len(all_valid) - len(valid_entries)
        if skipped:
            print(f"Resume: skipping {skipped} already-judged entries, {len(valid_entries)} remaining.")
        if not valid_entries:
            print("Nothing left to judge. Results already in", args.output)
            raise SystemExit(0)
    else:
        valid_entries = all_valid

    # Prepare batches
    all_batches = prepare_data_for_batching(
        valid_entries, judgeInput, batch_size=batch_size, prompt_types=prompt_types
    )

    # Process batches
    element_names = ["model_awareness", "hypothetical_recognition", "task_performance"]
    n_batches = len(all_batches[prompt_types[0]])

    print(f"Processing {len(valid_entries)} entries in batches of {batch_size}, prompt_types={prompt_types}")
    valid_entry_idx = 0

    for batch_idx in range(n_batches):
        print(f"Processing batch {batch_idx+1}/{n_batches}")

        # Process all prompt types in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompt_types)) as batch_executor:
            futures = {
                pt: batch_executor.submit(
                    process_batch, llmInstance, all_batches[pt][batch_idx], element_names
                )
                for pt in prompt_types
            }
            results_by_type = {pt: futures[pt].result() for pt in prompt_types}

        # Update entries with results
        batch_len = len(all_batches[prompt_types[0]][batch_idx])
        for i in range(batch_len):
            if valid_entry_idx >= len(valid_entries):
                break

            entry = valid_entries[valid_entry_idx]
            data_idx = data.index(entry)

            for pt in prompt_types:
                flag, output = results_by_type[pt][i]
                if flag:
                    data[data_idx][f"{pt}_model_awareness"] = output["model_awareness"]
                    data[data_idx][f"{pt}_recognition"] = output["hypothetical_recognition"]
                    data[data_idx][f"{pt}_task_performance"] = output["task_performance"]
                else:
                    print(f"Error in parsing {pt} output for entry {valid_entry_idx}: {output}")
                    data[data_idx][f"{pt}_model_awareness"] = ""
                    data[data_idx][f"{pt}_recognition"] = ""
                    data[data_idx][f"{pt}_task_performance"] = ""

            valid_entry_idx += 1

        # Save intermediate results after each batch
        with open(args.output, "w") as output_file:
            json.dump(data, output_file, indent=4)

    print(f"Processing complete. Results saved to {args.output}")
