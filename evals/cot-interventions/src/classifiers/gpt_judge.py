"""GPT judge for classifying resampled continuations.

Adapted from Test_Awareness_Steering/scripts/judgeIt_batch.py.
Classifies continuations for eval-awareness and task-performance.
"""

import json
import os
import time
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def get_llm(config: dict) -> ChatOpenAI:
    """Create LLM instance for judging."""
    judge_config = config["judge"]
    load_dotenv(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "Test_Awareness_Steering", ".env"
        )
    )
    return ChatOpenAI(
        model=judge_config["model"],
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        timeout=60,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def load_judge_prompt(config: dict) -> str:
    """Load the judge prompt template."""
    with open(config["judge"]["prompt_file"], "r") as f:
        return f.read()


def format_judge_input(prompt_template: str, task: str, reasoning: str, answer: str) -> str:
    """Format a single continuation for judging.

    Uses .replace() instead of .format() to avoid crashing when
    task/reasoning/answer contain { or } characters.
    """
    result = (prompt_template
              .replace("{task}", task)
              .replace("{reasoning}", reasoning)
              .replace("{answer}", answer))
    # Convert escaped braces {{ / }} (used in the JSON example in the template)
    # to literal braces. Done after .replace() so inserted values are untouched.
    result = result.replace("{{", "{").replace("}}", "}")
    return result


def call_llm_with_retry(llm: ChatOpenAI, message: str, max_retries: int = 3) -> dict | None:
    """Call LLM with retry on rate limits. Returns parsed JSON or None."""
    for attempt in range(max_retries):
        try:
            output = llm.invoke([HumanMessage(content=message)])
            return json.loads(output.content)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                time.sleep(10 * (attempt + 1))
                continue
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} retries: {e}")
                return None
    return None


def judge_continuations(
    records: list[dict],
    original_prompts: dict[int, str],
    config: dict,
) -> list[dict]:
    """Judge a list of continuation records for eval-awareness and task-performance.

    Args:
        records: list of dicts with keys: seed_index, sample_index, condition,
                 reasoning, response, full_text
        original_prompts: mapping from seed_index to the original user prompt
        config: experiment config

    Returns:
        Updated records with judge labels added.
    """
    llm = get_llm(config)
    prompt_template = load_judge_prompt(config)
    max_workers = config["judge"]["max_workers"]

    # Prepare all judge inputs
    judge_inputs = []
    for rec in records:
        task = original_prompts[rec["seed_index"]]
        judge_input = format_judge_input(
            prompt_template,
            task=task,
            reasoning=rec["reasoning"],
            answer=rec["response"],
        )
        judge_inputs.append(judge_input)

    print(f"Judging {len(judge_inputs)} continuations with {max_workers} workers...")

    # Process in parallel — don't use `with` so we can abandon hung threads
    results = [None] * len(judge_inputs)
    per_call_timeout = 300  # 5 min max waiting for new completions
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_idx = {
            executor.submit(call_llm_with_retry, llm, inp): i
            for i, inp in enumerate(judge_inputs)
        }
        done = 0
        pending = set(future_to_idx.keys())
        while pending:
            completed, pending = concurrent.futures.wait(
                pending, timeout=per_call_timeout,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            if not completed:
                # Timeout: mark remaining as None and move on
                print(f"  WARNING: {len(pending)} futures timed out after {per_call_timeout}s, abandoning...")
                for f in pending:
                    idx = future_to_idx[f]
                    results[idx] = None
                    done += 1
                break
            for future in completed:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=1)
                except Exception as e:
                    print(f"  WARNING: future {idx} raised {type(e).__name__}: {e}")
                    results[idx] = None
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(judge_inputs)} judged")
    finally:
        # shutdown without waiting — don't block on hung threads
        executor.shutdown(wait=False, cancel_futures=True)

    # Parse results into records
    for i, (rec, result) in enumerate(zip(records, results)):
        if result is None:
            rec["awareness_conclusion"] = "error"
            rec["awareness_evidence"] = []
            rec["task_performance_conclusion"] = "error"
            rec["task_performance_evidence"] = []
            continue

        ma = result.get("model_awareness", {})
        tp = result.get("task_performance", {})
        rec["awareness_conclusion"] = ma.get("conclusion", "error")
        rec["awareness_evidence"] = ma.get("evidence", [])
        rec["awareness_reasoning"] = ma.get("reasoning", "")
        rec["task_performance_conclusion"] = tp.get("conclusion", "error")
        rec["task_performance_evidence"] = tp.get("evidence", [])
        rec["task_performance_reasoning"] = tp.get("reasoning", "")

    print(f"Judging complete. {sum(1 for r in records if r.get('awareness_conclusion') != 'error')}/{len(records)} successful.")
    return records


def save_judged_records(records: list[dict], output_path: str):
    """Save judged records to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} judged records to {output_path}")
