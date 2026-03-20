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


def _make_template(entry, prompt_key, result_key, judge_input):
    """Build a judge template for a given prompt type."""
    task = entry[prompt_key]
    result = entry[result_key]
    if "</think>" in result:
        reasoning = result.split("</think>")[0]
        answer = result.split("</think>")[1]
    else:
        reasoning = ""
        answer = result
    return judge_input.format(task=task, reasoning=reasoning, answer=answer)


def prepare_data_for_batching(data, judge_input, batch_size=10, include_safety=False, safety_only=False):
    """Prepare data in batches for processing"""
    real_batch_templates = []
    hypothetical_batch_templates = []
    safety_batch_templates = []

    for entry in data:
        if not safety_only:
            if not (entry.get("hypothetical_result") and entry.get("real_result")):
                continue
            real_batch_templates.append(
                _make_template(entry, "real", "real_result", judge_input)
            )
            hypothetical_batch_templates.append(
                _make_template(entry, "hypothetical", "hypothetical_result", judge_input)
            )

        if include_safety and entry.get("safety_result") and entry.get("safety"):
            safety_batch_templates.append(
                _make_template(entry, "safety", "safety_result", judge_input)
            )

    def _chunk(lst):
        return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

    real_batches = _chunk(real_batch_templates)
    hypothetical_batches = _chunk(hypothetical_batch_templates)
    safety_batches = _chunk(safety_batch_templates) if include_safety else []

    return real_batches, hypothetical_batches, safety_batches


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
        "--include-safety",
        action="store_true",
        help="Also judge safety_result field (writes safety_model_awareness, etc.).",
    )
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="Judge ONLY safety_result (skip real/hypothetical). Implies --include-safety.",
    )

    args = parser.parse_args()
    if args.safety_only:
        args.include_safety = True
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

    if args.safety_only:
        # Safety-only mode: filter to entries with safety_result
        all_valid = [
            entry for entry in data
            if entry.get("safety_result") and entry.get("safety")
        ]
        if args.resume and os.path.isfile(args.output):
            valid_entries = [
                e for e in all_valid if not e.get("safety_model_awareness")
            ]
            skipped = len(all_valid) - len(valid_entries)
            if skipped:
                print(f"Resume: skipping {skipped} already-judged safety entries, {len(valid_entries)} remaining.")
            if not valid_entries:
                print("Nothing left to judge. Results already in", args.output)
                raise SystemExit(0)
        else:
            valid_entries = all_valid
    else:
        # Standard mode: filter to entries with real + hypothetical results
        all_valid = [
            entry
            for entry in data
            if (entry.get("hypothetical_result") and entry.get("real_result"))
        ]
        if args.resume and os.path.isfile(args.output):
            valid_entries = [
                entry
                for entry in all_valid
                if not entry.get("real_model_awareness")
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
    real_batches, hypothetical_batches, safety_batches = prepare_data_for_batching(
        valid_entries, judgeInput, batch_size=batch_size,
        include_safety=args.include_safety, safety_only=args.safety_only,
    )

    # Process batches
    element_names = ["model_awareness", "hypothetical_recognition", "task_performance"]

    # Process real + hypothetical entries (skip in safety-only mode)
    print(f"Processing {len(valid_entries)} entries in batches of {batch_size}")
    valid_entry_idx = 0

    if not args.safety_only:
        for batch_idx, (real_batch, hypothetical_batch) in enumerate(
            zip(real_batches, hypothetical_batches)
        ):
            print(f"Processing batch {batch_idx+1}/{len(real_batches)}")

            # Process real and hypothetical entries in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as batch_executor:
                real_future = batch_executor.submit(
                    process_batch, llmInstance, real_batch, element_names
                )
                hypo_future = batch_executor.submit(
                    process_batch, llmInstance, hypothetical_batch, element_names
                )
                real_results = real_future.result()
                hypothetical_results = hypo_future.result()

            # Update entries with results
            for i in range(len(real_batch)):
                if valid_entry_idx >= len(valid_entries):
                    break

                entry = valid_entries[valid_entry_idx]
                data_idx = data.index(entry)  # Find index in original data

                # Update with real results
                real_flag, real_output = real_results[i]
                if real_flag:
                    data[data_idx]["real_model_awareness"] = real_output["model_awareness"]
                    data[data_idx]["real_recognition"] = real_output[
                        "hypothetical_recognition"
                    ]
                    data[data_idx]["real_task_performance"] = real_output[
                        "task_performance"
                    ]
                else:
                    print(
                        f"Error in parsing real output for entry {valid_entry_idx}: {real_output}"
                    )
                    data[data_idx]["real_model_awareness"] = ""
                    data[data_idx]["real_recognition"] = ""
                    data[data_idx]["real_task_performance"] = ""

                # Update with hypothetical results
                hypo_flag, hypo_output = hypothetical_results[i]
                if hypo_flag:
                    data[data_idx]["hypothetical_model_awareness"] = hypo_output[
                        "model_awareness"
                    ]
                    data[data_idx]["hypothetical_recognition"] = hypo_output[
                        "hypothetical_recognition"
                    ]
                    data[data_idx]["hypothetical_task_performance"] = hypo_output[
                        "task_performance"
                    ]
                else:
                    print(
                        f"Error in parsing hypothetical output for entry {valid_entry_idx}: {hypo_output}"
                    )
                    data[data_idx]["hypothetical_model_awareness"] = ""
                    data[data_idx]["hypothetical_recognition"] = ""
                    data[data_idx]["hypothetical_task_performance"] = ""

                valid_entry_idx += 1

            # Save intermediate results after each batch
            with open(args.output, "w") as output_file:
                json.dump(data, output_file, indent=4)

    # --- Safety judging pass (if enabled) ---
    if args.include_safety and safety_batches:
        print(f"\nProcessing {len(safety_batches)} safety batches...")
        safety_entry_idx = 0
        # Build list of entries that have safety_result
        safety_valid = [
            e for e in valid_entries
            if e.get("safety_result") and e.get("safety")
        ]
        for batch_idx, safety_batch in enumerate(safety_batches):
            print(f"Safety batch {batch_idx+1}/{len(safety_batches)}")
            safety_results = process_batch(llmInstance, safety_batch, element_names)

            for i in range(len(safety_batch)):
                if safety_entry_idx >= len(safety_valid):
                    break
                entry = safety_valid[safety_entry_idx]
                data_idx = data.index(entry)

                sflag, soutput = safety_results[i]
                if sflag:
                    data[data_idx]["safety_model_awareness"] = soutput["model_awareness"]
                    data[data_idx]["safety_recognition"] = soutput["hypothetical_recognition"]
                    data[data_idx]["safety_task_performance"] = soutput["task_performance"]
                else:
                    print(f"Error in parsing safety output for entry {safety_entry_idx}: {soutput}")
                    data[data_idx]["safety_model_awareness"] = ""
                    data[data_idx]["safety_recognition"] = ""
                    data[data_idx]["safety_task_performance"] = ""

                safety_entry_idx += 1

            with open(args.output, "w") as output_file:
                json.dump(data, output_file, indent=4)

    print(f"Processing complete. Results saved to {args.output}")
