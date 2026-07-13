"""Construct vLLM prompts (prefixes) for ablation and addition experiments."""

from transformers import AutoTokenizer

from src.data.loader import Trajectory


def get_chat_prefix(tokenizer: AutoTokenizer, user_message: str) -> str:
    """Apply chat template and return the prefix up to (and including) <think>.

    The resulting string ends with '<think>' (or '<think>\n'), ready for
    reasoning content to be appended.
    """
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # The chat template for OLMo Think models generates a prompt ending with
    # the assistant turn start + <think>. The model then generates the thinking
    # content. We want to return everything up to and including <think>.
    # For most OLMo Think models, apply_chat_template with add_generation_prompt=True
    # already includes <think> at the end.
    return text


# -- Ablation prefixes --

def build_ablation_intervention_prefix(
    tokenizer: AutoTokenizer, trajectory: Trajectory
) -> str:
    """Build the intervention prefix for ablation: everything up to just before e1.

    Prefix = chat_template(user_message) + reasoning[:e1_position]
    The model will generate the rest of the reasoning + </think> + response.
    """
    assert trajectory.e1_position is not None, "e1 not located for this trajectory"
    chat_prefix = get_chat_prefix(tokenizer, trajectory.prompt)
    reasoning_prefix = trajectory.reasoning[:trajectory.e1_position]
    return chat_prefix + reasoning_prefix


def build_ablation_baseline_prefix(
    tokenizer: AutoTokenizer, trajectory: Trajectory
) -> str:
    """Build the baseline prefix for ablation: everything through the e1 sentence.

    Prefix = chat_template(user_message) + reasoning[:e1_sentence_end]
    The model will generate the rest of the reasoning + </think> + response.
    """
    assert trajectory.e1_sentence_end is not None, "e1 sentence end not located"
    chat_prefix = get_chat_prefix(tokenizer, trajectory.prompt)
    reasoning_prefix = trajectory.reasoning[:trajectory.e1_sentence_end]
    return chat_prefix + reasoning_prefix


# -- Addition prefixes --

def build_addition_prefix(
    tokenizer: AutoTokenizer,
    trajectory: Trajectory,
    injected_sentence: str | None = None,
) -> str:
    """Build a prefix for the addition experiment.

    Shared prefix = chat_template(user_message) + first_sentence_of_thinking

    If injected_sentence is provided, it's appended after the first sentence
    with a space separator. If None (baseline), just the first sentence is used.
    """
    assert trajectory.first_sentence_end is not None, "first sentence end not found"
    chat_prefix = get_chat_prefix(tokenizer, trajectory.prompt)
    first_sentence = trajectory.reasoning[:trajectory.first_sentence_end]

    if injected_sentence is not None:
        return chat_prefix + first_sentence + injected_sentence + " "
    else:
        return chat_prefix + first_sentence
