"""Dataset creation utilities for SFT and DPO training."""

from dataclasses import asdict, dataclass


@dataclass
class SFTExample:
    id: str
    messages: list


@dataclass  
class DPOExample:
    id: str
    prompt: list
    chosen: str
    rejected: str


def create_sft_example(dp, original_response):
    messages = []
    for msg in dp["input"]:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            prefixed = "This is an evaluation. " + msg["content"]
            messages.append({"role": "user", "content": prefixed})
    messages.append({"role": "assistant", "content": original_response})
    return SFTExample(id=dp["id"], messages=messages)


def create_dpo_example(dp, original_response, prefixed_response):
    prompt = []
    for msg in dp["input"]:
        prompt.append({"role": msg["role"], "content": msg["content"]})
    return DPOExample(id=dp["id"], prompt=prompt, chosen=original_response, rejected=prefixed_response)


def responses_are_different(original, prefixed):
    orig_norm = " ".join(original.split())
    pref_norm = " ".join(prefixed.split())
    return orig_norm != pref_norm


def create_datasets(filtered_data, initial_responses, prefixed_responses):
    sft_examples = []
    dpo_examples = []
    stats = {"total": len(filtered_data), "kept": 0, "skipped_identical": 0, "skipped_missing": 0}
    for dp in filtered_data:
        dp_id = dp["id"]
        if dp_id not in initial_responses or dp_id not in prefixed_responses:
            stats["skipped_missing"] += 1
            continue
        original = initial_responses[dp_id]
        prefixed = prefixed_responses[dp_id]
        if not responses_are_different(original, prefixed):
            stats["skipped_identical"] += 1
            continue
        sft_examples.append(create_sft_example(dp, original))
        dpo_examples.append(create_dpo_example(dp, original, prefixed))
        stats["kept"] += 1
    return sft_examples, dpo_examples, stats


def sft_examples_to_dicts(examples):
    return [asdict(ex) for ex in examples]


def dpo_examples_to_dicts(examples):
    return [asdict(ex) for ex in examples]

