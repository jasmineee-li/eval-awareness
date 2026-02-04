"""Inference utilities for running QwQ-32B via OpenRouter."""

import asyncio
import json
from pathlib import Path
from typing import Any

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from tqdm.asyncio import tqdm

from consistency_training.data_loading import DataPoint, Message

# Model configuration
QWQ_MODEL_ID = "openrouter/qwen/qwq-32b"
EVAL_PREFIX = "This is an evaluation."


def message_to_chat_message(msg: Message) -> ChatMessage:
    """Convert data message to safety-tooling ChatMessage.

    Args:
        msg: Message dict with role and content.

    Returns:
        ChatMessage object.
    """
    role_map = {
        "system": MessageRole.system,
        "user": MessageRole.user,
        "assistant": MessageRole.assistant,
    }
    return ChatMessage(role=role_map[msg["role"]], content=msg["content"])


def datapoint_to_prompt(dp: DataPoint, add_prefix: bool = False) -> Prompt:
    """Convert a data point to a Prompt object.

    If add_prefix=True, prepends "This is an evaluation." to the user message.

    Args:
        dp: DataPoint with input messages.
        add_prefix: Whether to add the evaluation prefix to user messages.

    Returns:
        Prompt object ready for inference.
    """
    messages = []
    for msg in dp["input"]:
        chat_msg = message_to_chat_message(msg)

        # Add prefix to user message if requested
        if add_prefix and msg["role"] == "user":
            prefixed_content = f"{EVAL_PREFIX} {msg['content']}"
            chat_msg = ChatMessage(role=MessageRole.user, content=prefixed_content)

        messages.append(chat_msg)

    return Prompt(messages=messages)


async def run_inference_batch(
    api: InferenceAPI,
    data: list[DataPoint],
    model_id: str = QWQ_MODEL_ID,
    add_prefix: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    concurrency: int = 80,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 500,
) -> dict[str, LLMResponse]:
    """Run inference on data points with high concurrency.

    Uses asyncio.gather to run `concurrency` requests in parallel.
    OpenRouter doesn't support true batching, so we maximize throughput
    via concurrent async requests.

    Args:
        api: InferenceAPI instance.
        data: List of data points to process.
        model_id: Model to use for inference.
        add_prefix: Whether to add "This is an evaluation." prefix.
        max_tokens: Max tokens for response.
        temperature: Sampling temperature.
        concurrency: Number of parallel requests per batch.
        checkpoint_path: Path to save periodic checkpoints. If None, no checkpoints.
        checkpoint_every: Save checkpoint every N items.

    Returns:
        Dict mapping data point ID to LLMResponse.
    """
    results: dict[str, LLMResponse] = {}
    errors: list[tuple[str, Exception]] = []

    # Load existing checkpoint if it exists
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        # Skip already-processed IDs
        processed_ids = set(checkpoint_data.keys())
        data = [dp for dp in data if dp["id"] not in processed_ids]
        print(f"  Resuming: {len(processed_ids)} already done, {len(data)} remaining")
        # We'll merge checkpoint data at the end
    else:
        checkpoint_data = {}
        processed_ids = set()

    async def process_one(dp: DataPoint) -> tuple[str, LLMResponse | Exception]:
        """Process a single data point."""
        try:
            prompt = datapoint_to_prompt(dp, add_prefix=add_prefix)
            responses = await api(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
            )
            return dp["id"], responses[0]
        except Exception as e:
            return dp["id"], e

    # Process in concurrent batches
    prefix_str = "prefixed" if add_prefix else "no-prefix"
    desc = f"Inference ({prefix_str}, {concurrency} parallel)"
    items_since_checkpoint = 0

    for i in tqdm(range(0, len(data), concurrency), desc=desc):
        batch = data[i : i + concurrency]
        tasks = [process_one(dp) for dp in batch]
        # asyncio.gather runs all tasks concurrently
        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            dp_id, response = result
            if isinstance(response, Exception):
                errors.append((dp_id, response))
                continue
            results[dp_id] = response
            items_since_checkpoint += 1

        # Periodic checkpoint
        if checkpoint_path and items_since_checkpoint >= checkpoint_every:
            _save_checkpoint(checkpoint_path, results, checkpoint_data)
            items_since_checkpoint = 0

    # Final checkpoint
    if checkpoint_path and results:
        _save_checkpoint(checkpoint_path, results, checkpoint_data)

    # Merge with any existing checkpoint data
    final_results = {}
    for dp_id, resp_data in checkpoint_data.items():
        # Reconstruct a minimal object that serialize_responses can handle
        final_results[dp_id] = _dict_to_response_stub(resp_data)
    final_results.update(results)

    # Report errors summary
    if errors:
        print(f"\nWarning: {len(errors)} requests failed out of {len(data)} total")
        # Show first few errors for debugging
        for dp_id, err in errors[:3]:
            print(f"  - {dp_id}: {type(err).__name__}: {err}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")

    return final_results


def _save_checkpoint(
    checkpoint_path: Path,
    results: dict[str, LLMResponse],
    existing_data: dict,
) -> None:
    """Save checkpoint combining existing data with new results."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge existing checkpoint with new results
    merged = dict(existing_data)
    for dp_id, resp in results.items():
        merged[dp_id] = {
            "completion": resp.completion,
            "model_id": resp.model_id,
            "stop_reason": str(resp.stop_reason) if resp.stop_reason else None,
        }
    
    # Write atomically using temp file
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(merged, f)
    temp_path.rename(checkpoint_path)
    
    print(f"  [Checkpoint saved: {len(merged)} items]")


class _ResponseStub:
    """Minimal stub to hold response data for serialize_responses."""
    def __init__(self, completion: str, model_id: str, stop_reason: str | None):
        self.completion = completion
        self.model_id = model_id
        self.stop_reason = stop_reason


def _dict_to_response_stub(data: dict) -> _ResponseStub:
    """Convert checkpoint dict back to a response-like object."""
    return _ResponseStub(
        completion=data["completion"],
        model_id=data["model_id"],
        stop_reason=data.get("stop_reason"),
    )


def serialize_responses(responses: dict[str, LLMResponse]) -> dict[str, dict[str, Any]]:
    """Serialize LLMResponse objects to JSON-serializable dicts.

    Args:
        responses: Dict mapping IDs to LLMResponse objects.

    Returns:
        Dict mapping IDs to serializable dicts with completion, model_id, stop_reason.
    """
    return {
        dp_id: {
            "completion": resp.completion,
            "model_id": resp.model_id,
            "stop_reason": str(resp.stop_reason) if resp.stop_reason else None,
        }
        for dp_id, resp in responses.items()
    }
