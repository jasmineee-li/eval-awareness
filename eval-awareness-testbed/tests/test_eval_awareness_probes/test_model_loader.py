"""Tests for eval_awareness_probes.model_loader module."""

import re
from types import SimpleNamespace

import pytest
import torch

from eval_awareness_probes.model_loader import (
    HookedTransformerShim,
    _extract_activation,
    _get_config_value,
)


class TestExtractActivation:
    """Tests for the _extract_activation helper."""

    def test_resid_pre(self):
        # 4 layers worth of hidden states
        hidden_states = tuple(torch.randn(1, 5, 64) for _ in range(5))
        result = _extract_activation("blocks.2.hook_resid_pre", hidden_states)
        assert result is not None
        assert torch.equal(result, hidden_states[2])

    def test_resid_post(self):
        hidden_states = tuple(torch.randn(1, 5, 64) for _ in range(5))
        result = _extract_activation("blocks.1.hook_resid_post", hidden_states)
        assert result is not None
        assert torch.equal(result, hidden_states[2])  # post = layer + 1

    def test_out_of_range(self):
        hidden_states = tuple(torch.randn(1, 5, 64) for _ in range(3))
        result = _extract_activation("blocks.10.hook_resid_pre", hidden_states)
        assert result is None

    def test_unknown_hook_name(self):
        hidden_states = tuple(torch.randn(1, 5, 64) for _ in range(3))
        result = _extract_activation("unknown.hook.name", hidden_states)
        assert result is None


class TestGetConfigValue:
    """Tests for the _get_config_value helper."""

    def test_first_attr_found(self):
        cfg = SimpleNamespace(num_hidden_layers=32)
        result = _get_config_value(cfg, None, ["num_hidden_layers", "n_layers"])
        assert result == 32

    def test_second_attr_found(self):
        cfg = SimpleNamespace(n_layers=24)
        result = _get_config_value(cfg, None, ["num_hidden_layers", "n_layers"])
        assert result == 24

    def test_no_attr_found(self):
        cfg = SimpleNamespace(something_else=42)
        result = _get_config_value(cfg, None, ["num_hidden_layers", "n_layers"])
        assert result is None

    def test_infer_from_model_structure(self):
        cfg = SimpleNamespace()  # No config attrs
        # Model with .model.layers
        inner_model = SimpleNamespace(layers=[None] * 16)
        model = SimpleNamespace(model=inner_model)
        result = _get_config_value(cfg, model, ["num_hidden_layers"])
        assert result == 16


class TestHookedTransformerShim:
    """Tests for the HookedTransformerShim class (without actual model loading)."""

    def _make_shim(self, d_model=64, n_layers=4, seq_len=10):
        """Create a shim with a fake HF model."""

        class FakeOutput:
            def __init__(self, hidden_states, logits):
                self.hidden_states = hidden_states
                self.logits = logits

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    output_hidden_states=True,
                    return_dict=True,
                )

            def eval(self):
                return self

            def __call__(self, input_ids=None, **kwargs):
                batch = input_ids.shape[0] if input_ids is not None else 1
                sl = input_ids.shape[1] if input_ids is not None else seq_len
                hidden_states = tuple(
                    torch.randn(batch, sl, d_model) for _ in range(n_layers + 1)
                )
                logits = torch.randn(batch, sl, 100)
                return FakeOutput(hidden_states, logits)

            def parameters(self):
                return iter([torch.zeros(1)])

        class FakeTokenizer:
            pad_token = "<pad>"
            pad_token_id = 0
            eos_token = "</s>"

            def encode(self, text, add_special_tokens=True):
                return list(range(seq_len))

            def decode(self, token_ids):
                return "token"

            def convert_tokens_to_ids(self, token):
                return 1

            def __call__(self, text, **kwargs):
                ids = torch.zeros(1, seq_len, dtype=torch.long)
                mask = torch.ones(1, seq_len, dtype=torch.long)
                return SimpleNamespace(input_ids=ids, attention_mask=mask)

        cfg = SimpleNamespace(
            model_name="fake/model",
            n_layers=n_layers,
            d_model=d_model,
            n_heads=4,
            d_head=d_model // 4,
            d_vocab=100,
            n_ctx=512,
        )

        return HookedTransformerShim(FakeModel(), FakeTokenizer(), "cpu", cfg)

    def test_to_tokens(self):
        shim = self._make_shim()
        tokens = shim.to_tokens("hello world")
        assert tokens.dim() == 2
        assert tokens.shape[0] == 1

    def test_to_tokens_batch(self):
        shim = self._make_shim()
        tokens = shim.to_tokens(["hello", "world"])
        assert tokens.shape[0] == 2

    def test_hooks_context_manager(self):
        shim = self._make_shim()
        assert len(shim._hooks) == 0

        recorded = []

        def hook_fn(activation, hook):
            recorded.append(activation.shape)

        with shim.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)]):
            assert len(shim._hooks) == 1
            tokens = shim.to_tokens("test")
            shim(tokens)

        # Hooks should be cleaned up
        assert len(shim._hooks) == 0
        assert len(recorded) == 1

    def test_reset_hooks(self):
        shim = self._make_shim()
        shim._hooks["test"] = lambda x, h: x
        shim.reset_hooks()
        assert len(shim._hooks) == 0

    def test_run_with_cache(self):
        shim = self._make_shim(n_layers=4)
        logits, cache = shim.run_with_cache(
            "test input",
            names_filter=["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"],
        )
        assert logits is not None
        assert "blocks.0.hook_resid_pre" in cache
        assert "blocks.1.hook_resid_pre" in cache
        assert "blocks.2.hook_resid_pre" not in cache  # not requested

    def test_run_with_cache_all_layers(self):
        shim = self._make_shim(n_layers=4)
        logits, cache = shim.run_with_cache("test input")
        # Should have pre and post for each layer
        assert len(cache) > 0

    def test_forward_call(self):
        shim = self._make_shim()
        tokens = shim.to_tokens("test")
        output = shim(tokens)
        assert output is not None
        assert hasattr(output, "hidden_states")
