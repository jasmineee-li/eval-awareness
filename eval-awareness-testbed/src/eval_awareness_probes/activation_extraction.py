"""Modular activation extraction for probe evaluation.

Supports different token selection strategies and two extraction backends:
1. Hooked models (TransformerLens / HookedTransformerShim) - uses forward hooks
2. HuggingFace models (direct) - uses output_hidden_states

Adapted from:
- evaluation-awareness-scaling-laws/src/vector_generation.py (hook-based extraction)
- Test_Awareness_Steering/scripts/getRepFromEvidence.py (HF-based extraction)
"""

from __future__ import annotations

import gc
import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class TokenSelector(str, Enum):
    """Strategy for selecting which token positions to extract activations from."""

    ANSWER_TOKEN = "answer_token"  # Activation at the answer token position (e.g. 'A' in '(A)')
    LAST_TOKEN = "last_token"  # Activation at the last token
    MEAN_ALL = "mean_all"  # Mean activation across all non-padding tokens
    EVIDENCE_TOKENS = "evidence_tokens"  # Activations at evidence sentence token positions


def extract_activations_hooked(
    model,
    tokens: torch.Tensor,
    layers: list[int],
    token_selector: TokenSelector = TokenSelector.MEAN_ALL,
    tokenizer=None,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract activations using hook-based models (TransformerLens / shim).

    Uses ``blocks.{layer}.hook_resid_pre`` to capture residual stream activations.

    Args:
        model: HookedTransformer or HookedTransformerShim model.
        tokens: Tokenized input of shape (1, seq_len).
        layers: List of layer indices to extract from.
        token_selector: Which token positions to extract.
        tokenizer: Tokenizer (needed for answer_token selection).
        answer_token: Token string to find (e.g. 'A') when using ANSWER_TOKEN.
        evidence_indices: List of (start, end) token index pairs for EVIDENCE_TOKENS.

    Returns:
        Dict mapping layer index to activation tensor of shape (1, d_model).
    """
    cached: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook_fn(activation, hook):
            act = _select_token_activation(
                activation,
                tokens,
                token_selector,
                tokenizer,
                answer_token,
                evidence_indices,
            )
            cached[layer] = act.clone().detach()
            return activation

        return hook_fn

    fwd_hooks = [
        (f"blocks.{layer}.hook_resid_pre", make_hook(layer)) for layer in layers
    ]

    model.reset_hooks()
    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            model(tokens)

    return cached


def extract_activations_hf(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layers: list[int],
    token_selector: TokenSelector = TokenSelector.MEAN_ALL,
    tokenizer=None,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract activations from a plain HuggingFace model using output_hidden_states.

    Args:
        model: HuggingFace model (AutoModelForCausalLM with output_hidden_states=True).
        input_ids: Token IDs of shape (batch, seq_len).
        attention_mask: Attention mask (optional).
        layers: Layer indices to extract from.
        token_selector: Which token positions to extract.
        tokenizer: Tokenizer (needed for answer_token selection).
        answer_token: Token string to find when using ANSWER_TOKEN.
        evidence_indices: List of (start, end) for EVIDENCE_TOKENS.

    Returns:
        Dict mapping layer index to activation tensor.
    """
    kwargs = {"output_hidden_states": True}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        outputs = model(input_ids, **kwargs)

    hidden_states = outputs.hidden_states
    cached: dict[int, torch.Tensor] = {}

    for layer in layers:
        if layer < len(hidden_states):
            act = _select_token_activation(
                hidden_states[layer],
                input_ids,
                token_selector,
                tokenizer,
                answer_token,
                evidence_indices,
            )
            cached[layer] = act.clone().detach().cpu()

    return cached


def extract_activations(
    model,
    text: str,
    layers: list[int],
    tokenizer,
    token_selector: TokenSelector = TokenSelector.MEAN_ALL,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Unified entry point for activation extraction.

    Automatically detects whether the model is a hooked model or plain HF model.

    Args:
        model: Any supported model (HookedTransformer, shim, or HF model).
        text: Input text to process.
        layers: Layer indices to extract from.
        tokenizer: Tokenizer for the model.
        token_selector: Token selection strategy.
        answer_token: Answer token string for ANSWER_TOKEN selection.
        evidence_indices: Evidence index pairs for EVIDENCE_TOKENS selection.

    Returns:
        Dict mapping layer index to activation tensor.
    """
    is_hooked = hasattr(model, "hooks") and hasattr(model, "reset_hooks")

    if is_hooked:
        tokens = model.to_tokens(text)
        return extract_activations_hooked(
            model, tokens, layers, token_selector, tokenizer, answer_token, evidence_indices
        )
    else:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded.input_ids.to(
            next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        )
        attention_mask = encoded.attention_mask.to(input_ids.device)
        return extract_activations_hf(
            model,
            input_ids,
            attention_mask,
            layers,
            token_selector,
            tokenizer,
            answer_token,
            evidence_indices,
        )


def extract_contrastive_activations(
    model,
    dataset: list[dict],
    layers: list[int],
    tokenizer,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    """Extract activations for contrastive training pairs.

    For each item in dataset, extracts activations at the answer token position
    for both the 'positive' and 'negative' responses.

    Args:
        model: Hooked model (TransformerLens or shim).
        dataset: List of dicts with 'question', 'test'/'positive', 'deploy'/'negative'.
        layers: Layer indices.
        tokenizer: Tokenizer.

    Returns:
        Tuple of (positive_acts_by_layer, negative_acts_by_layer).
        Each is a dict mapping layer -> list of activation tensors.
    """
    positive_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    negative_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for item in dataset:
        question = item["question"]
        pos_answer = item.get("test") or item.get("positive")
        neg_answer = item.get("deploy") or item.get("negative")

        # Build prompts with answer appended
        pos_prompt = f"{question}\n\n {pos_answer}"
        neg_prompt = f"{question}\n\n {neg_answer}"

        # Determine answer token (char at index 1, e.g. "(A)" -> "A")
        pos_token = pos_answer[1] if len(pos_answer) > 1 else pos_answer
        neg_token = neg_answer[1] if len(neg_answer) > 1 else neg_answer

        # Extract positive activations
        pos_tokens = model.to_tokens(pos_prompt)
        pos_cached = extract_activations_hooked(
            model,
            pos_tokens,
            layers,
            TokenSelector.ANSWER_TOKEN,
            tokenizer,
            answer_token=pos_token,
        )

        # Extract negative activations
        neg_tokens = model.to_tokens(neg_prompt)
        neg_cached = extract_activations_hooked(
            model,
            neg_tokens,
            layers,
            TokenSelector.ANSWER_TOKEN,
            tokenizer,
            answer_token=neg_token,
        )

        for layer in layers:
            if layer in pos_cached:
                positive_acts[layer].append(pos_cached[layer])
            if layer in neg_cached:
                negative_acts[layer].append(neg_cached[layer])

        # Clean up
        del pos_tokens, neg_tokens, pos_cached, neg_cached
        gc.collect()

    return positive_acts, negative_acts


def _select_token_activation(
    activation: torch.Tensor,
    tokens: torch.Tensor,
    token_selector: TokenSelector,
    tokenizer=None,
    answer_token: str | None = None,
    evidence_indices: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Select activation at specific token position(s).

    Args:
        activation: Hidden state tensor of shape (batch, seq_len, d_model).
        tokens: Token IDs of shape (batch, seq_len).
        token_selector: Selection strategy.
        tokenizer: Tokenizer (for answer token lookup).
        answer_token: Token string to find.
        evidence_indices: (start, end) pairs for evidence token selection.

    Returns:
        Activation tensor of shape (1, d_model).
    """
    if token_selector == TokenSelector.ANSWER_TOKEN:
        if tokenizer is None or answer_token is None:
            raise ValueError(
                "tokenizer and answer_token required for ANSWER_TOKEN selection"
            )
        token_id = tokenizer.convert_tokens_to_ids(answer_token)
        positions = (tokens == token_id).nonzero()
        if len(positions) == 0:
            logger.warning(
                f"Answer token '{answer_token}' (id={token_id}) not found, "
                f"falling back to last token"
            )
            return activation[:, -1:, :].mean(dim=1, keepdim=False)
        last_pos = positions[-1][-1]
        return activation[:, last_pos, :]

    elif token_selector == TokenSelector.LAST_TOKEN:
        return activation[:, -1, :]

    elif token_selector == TokenSelector.MEAN_ALL:
        return activation.mean(dim=1)

    elif token_selector == TokenSelector.EVIDENCE_TOKENS:
        if evidence_indices is None or len(evidence_indices) == 0:
            logger.warning("No evidence indices provided, falling back to mean_all")
            return activation.mean(dim=1)

        evidence_acts = []
        for start, end in evidence_indices:
            chunk = activation[:, start:end, :]
            evidence_acts.append(chunk.mean(dim=1))
        return torch.stack(evidence_acts).mean(dim=0)

    else:
        raise ValueError(f"Unknown token selector: {token_selector}")
