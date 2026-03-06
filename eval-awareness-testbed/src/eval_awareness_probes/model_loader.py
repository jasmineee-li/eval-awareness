"""Model loading utilities for probe-based eval awareness detection.

Supports two backends:
1. TransformerLens HookedTransformer (native support for many models)
2. HookedTransformerShim (wraps any HuggingFace model for activation extraction)

Adapted from:
- evaluation-awareness-scaling-laws/probe_eval.ipynb (HookedTransformerShim)
- evaluation-awareness-scaling-laws/src/utils.py (load_model)
- Test_Awareness_Steering/scripts/getRepFromEvidence.py (HF loading)
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class HookedTransformerShim:
    """Minimal API-compatible shim for TransformerLens HookedTransformer.

    Wraps any HuggingFace model to provide activation extraction via forward
    hooks, matching the HookedTransformer API used in the probe pipeline.

    Tested with: OLMo, Gemma, Qwen, Llama, Phi families.
    """

    def __init__(self, hf_model, tokenizer, device: str, cfg):
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.cfg = cfg
        self._hooks: dict[str, callable] = {}
        self._hook_handles: list = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        revision: str | None = None,
    ) -> HookedTransformerShim:
        """Load a pretrained model from HuggingFace hub.

        Args:
            model_path: HuggingFace model path (e.g. 'allenai/OLMo-7B-Instruct').
            device: Device to load model on ('cuda', 'cpu', etc.).
            dtype: Data type (e.g. torch.bfloat16).
            revision: Git revision / checkpoint to load.

        Returns:
            HookedTransformerShim wrapping the loaded model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        # Load config with hidden states output enabled
        cfg_hf = AutoConfig.from_pretrained(model_path)
        cfg_hf.output_hidden_states = True
        cfg_hf.return_dict = True

        # Model loading kwargs
        load_kwargs: dict[str, Any] = {"config": cfg_hf}
        if revision is not None:
            load_kwargs["revision"] = revision
        if device != "cpu":
            load_kwargs["device_map"] = "auto"
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        model_hf = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device == "cpu":
            model_hf.to("cpu")

        # Build config namespace compatible with TransformerLens API
        n_layers = _get_config_value(
            cfg_hf,
            model_hf,
            ["num_hidden_layers", "n_layers", "num_layers", "n_layer"],
        )
        d_model = _get_config_value(
            cfg_hf, None, ["hidden_size", "d_model", "n_embd", "dim"]
        )
        n_heads = _get_config_value(
            cfg_hf, None, ["num_attention_heads", "n_heads", "num_heads", "n_head"]
        )

        if n_layers is None:
            raise ValueError(
                f"Could not determine number of layers for model {model_path}."
            )

        cfg = SimpleNamespace(
            model_name=model_path,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_model // n_heads if (d_model and n_heads) else None,
            d_vocab=getattr(cfg_hf, "vocab_size", None),
            n_ctx=getattr(cfg_hf, "max_position_embeddings", None),
        )

        logger.info(
            f"Loaded {model_path} via HookedTransformerShim "
            f"(n_layers={n_layers}, d_model={d_model})"
        )

        return cls(model_hf, tokenizer, device, cfg)

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def reset_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hooks = {}

    @contextmanager
    def hooks(self, fwd_hooks: list[tuple[str, callable]] | None = None, **kwargs):
        """Context manager for temporarily adding activation hooks.

        Args:
            fwd_hooks: List of (hook_name, hook_fn) pairs. Hook names follow
                TransformerLens convention: 'blocks.{layer}.hook_resid_pre'.
        """
        if fwd_hooks is None:
            fwd_hooks = []

        # Store hooks for use during forward pass
        old_hooks = self._hooks.copy()
        for name, fn in fwd_hooks:
            self._hooks[name] = fn

        try:
            yield
        finally:
            self._hooks = old_hooks

    def to_tokens(self, prompt: str | list[str], prepend_bos: bool = True) -> torch.Tensor:
        """Convert string(s) to token ids.

        Args:
            prompt: Input text or list of texts.
            prepend_bos: Whether to add BOS token.

        Returns:
            Token IDs tensor of shape (batch, seq_len).
        """
        if isinstance(prompt, (list, tuple)):
            ids = [
                self.tokenizer.encode(p, add_special_tokens=prepend_bos)
                for p in prompt
            ]
            max_len = max(len(seq) for seq in ids)
            padded = [
                seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
                for seq in ids
            ]
            return torch.tensor(padded).to(self.device)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=prepend_bos)
            return torch.tensor([ids]).to(self.device)

    def __call__(self, tokens: torch.Tensor, **kwargs) -> Any:
        """Forward pass, applying any registered hooks to hidden states."""
        if isinstance(tokens, dict):
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in tokens.items()
            }
        else:
            inputs = {"input_ids": tokens.to(self.device)}
        inputs.update(kwargs)

        # Always request hidden states when hooks are registered
        if self._hooks:
            inputs["output_hidden_states"] = True

        outputs = self.model(**inputs)

        # Apply registered hooks to hidden states
        if self._hooks and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states
            for name, fn in self._hooks.items():
                activation = _extract_activation(name, hidden_states)
                if activation is not None:
                    hook_point = SimpleNamespace(name=name)
                    try:
                        fn(activation, hook_point)
                    except TypeError:
                        fn(activation)

        return outputs

    def run_with_cache(
        self,
        tokens: torch.Tensor | str,
        names_filter: list[str] | None = None,
        device: str | None = None,
        remove_batch_dim: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run model and cache activations at specified hook points.

        Args:
            tokens: Input tokens or text.
            names_filter: List of hook names to cache (None = all).
            device: Device override.
            remove_batch_dim: Squeeze batch dim if batch_size=1.

        Returns:
            Tuple of (logits, cache_dict).
        """
        if isinstance(tokens, str):
            tokens = self.to_tokens(tokens)
        if device is not None:
            tokens = tokens.to(device)

        cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            for layer_idx in range(len(hidden_states) - 1):
                pre_name = f"blocks.{layer_idx}.hook_resid_pre"
                post_name = f"blocks.{layer_idx}.hook_resid_post"

                if names_filter is None or pre_name in names_filter:
                    cache[pre_name] = hidden_states[layer_idx]
                if names_filter is None or post_name in names_filter:
                    cache[post_name] = hidden_states[layer_idx + 1]

        if remove_batch_dim and tokens.shape[0] == 1:
            cache = {
                k: v.squeeze(0) if v.shape[0] == 1 else v for k, v in cache.items()
            }

        logits = outputs.logits
        if remove_batch_dim and logits.shape[0] == 1:
            logits = logits.squeeze(0)

        return logits, cache


def _get_config_value(cfg_hf, model_hf, attr_names: list[str]):
    """Try multiple config attribute names to find a value."""
    for name in attr_names:
        val = getattr(cfg_hf, name, None)
        if val is not None:
            return val

    # Try inferring from model structure
    if model_hf is not None and hasattr(model_hf, "model"):
        base = model_hf.model
        if hasattr(base, "layers"):
            return len(base.layers)
        if hasattr(base, "h"):
            return len(base.h)

    return None


def _extract_activation(
    hook_name: str, hidden_states: tuple[torch.Tensor, ...]
) -> torch.Tensor | None:
    """Extract activation tensor corresponding to a hook name."""
    match_pre = re.match(r"blocks\.(\d+)\.hook_resid_pre", hook_name)
    if match_pre:
        idx = int(match_pre.group(1))
        if idx < len(hidden_states):
            return hidden_states[idx]

    match_post = re.match(r"blocks\.(\d+)\.hook_resid_post", hook_name)
    if match_post:
        idx = int(match_post.group(1))
        if idx + 1 < len(hidden_states):
            return hidden_states[idx + 1]

    return None


def load_model(
    model_path: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
    backend: str = "auto",
) -> tuple:
    """Load a model and tokenizer for probe evaluation.

    Args:
        model_path: HuggingFace model path.
        device: Device to load on (default: 'cuda' if available, else 'cpu').
        dtype: Data type for model weights.
        revision: Git revision / checkpoint.
        backend: 'auto', 'transformerlens', or 'huggingface'.

    Returns:
        Tuple of (model, tokenizer). Model is either a HookedTransformer or
        HookedTransformerShim, both supporting the same activation extraction API.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "transformerlens":
        return _load_transformerlens(model_path, device, dtype)

    if backend == "huggingface":
        return _load_shim(model_path, device, dtype, revision)

    # auto: try TransformerLens first, fall back to shim
    try:
        return _load_transformerlens(model_path, device, dtype)
    except Exception as e:
        logger.info(
            f"TransformerLens failed for {model_path} ({e}), falling back to shim"
        )
        return _load_shim(model_path, device, dtype, revision)


def _load_transformerlens(
    model_path: str, device: str, dtype: torch.dtype
) -> tuple:
    """Load model via TransformerLens."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _load_shim(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    revision: str | None,
) -> tuple:
    """Load model via HookedTransformerShim."""
    model = HookedTransformerShim.from_pretrained(
        model_path, device=device, dtype=dtype, revision=revision
    )
    model.eval()
    return model, model.tokenizer
