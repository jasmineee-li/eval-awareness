"""Tests for eval_awareness_probes.attention_probe module."""

import json
from pathlib import Path

import pytest
import torch

from eval_awareness_probes.attention_probe import (
    AttentionProbe,
    _AttentionPooler,
    _AttentionProbeClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable_sequences(
    n_pos: int = 50,
    n_neg: int = 50,
    d_model: int = 32,
    seq_len_range: tuple[int, int] = (5, 15),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create separable variable-length sequences.

    Positive sequences have a strong signal (+3) injected at a random position.
    Negative sequences have the opposite signal (-3) at a random position.
    """
    pos_seqs = []
    neg_seqs = []
    for _ in range(n_pos):
        seq_len = torch.randint(seq_len_range[0], seq_len_range[1], (1,)).item()
        seq = torch.randn(seq_len, d_model) * 0.1
        signal_pos = torch.randint(0, seq_len, (1,)).item()
        seq[signal_pos] += 3.0
        pos_seqs.append(seq)
    for _ in range(n_neg):
        seq_len = torch.randint(seq_len_range[0], seq_len_range[1], (1,)).item()
        seq = torch.randn(seq_len, d_model) * 0.1
        signal_pos = torch.randint(0, seq_len, (1,)).item()
        seq[signal_pos] -= 3.0
        neg_seqs.append(seq)
    return pos_seqs, neg_seqs


# ---------------------------------------------------------------------------
# _AttentionPooler tests
# ---------------------------------------------------------------------------

class TestAttentionPooler:
    """Tests for the low-level _AttentionPooler module."""

    def test_output_shapes(self):
        d_model, d_head, n_heads = 32, 8, 2
        pooler = _AttentionPooler(d_model, d_head, n_heads)
        hidden = torch.randn(4, 10, d_model)  # batch=4, seq=10
        pooled, weights = pooler(hidden)
        assert pooled.shape == (4, n_heads * d_head)
        assert weights.shape == (4, n_heads, 10)

    def test_attention_weights_sum_to_one(self):
        pooler = _AttentionPooler(16, 8, 1)
        hidden = torch.randn(2, 7, 16)
        _, weights = pooler(hidden)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mask_zeroes_padding(self):
        pooler = _AttentionPooler(16, 8, 1)
        hidden = torch.randn(1, 6, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float)
        _, weights = pooler(hidden, attention_mask=mask)
        # Padded positions should have zero attention weight
        assert torch.allclose(
            weights[0, 0, 3:], torch.zeros(3), atol=1e-6
        )
        # Real positions should have non-zero weight
        assert weights[0, 0, :3].sum().item() > 0.99

    def test_zero_init_gives_uniform_attention(self):
        """With W_q zero-init and zero query, attention should be near-uniform."""
        pooler = _AttentionPooler(16, 8, 1)
        hidden = torch.randn(1, 5, 16)
        _, weights = pooler(hidden)
        expected = torch.full((1, 1, 5), 1.0 / 5)
        assert torch.allclose(weights, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# _AttentionProbeClassifier tests
# ---------------------------------------------------------------------------

class TestAttentionProbeClassifier:
    """Tests for the _AttentionProbeClassifier module."""

    def test_output_shapes(self):
        model = _AttentionProbeClassifier(d_model=32, d_head=8, n_heads=2)
        hidden = torch.randn(3, 10, 32)
        logits, weights = model(hidden)
        assert logits.shape == (3, 2)
        assert weights.shape == (3, 2, 10)

    def test_forward_with_mask(self):
        model = _AttentionProbeClassifier(d_model=16, d_head=8, n_heads=1)
        hidden = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6)
        mask[1, 4:] = 0
        logits, weights = model(hidden, attention_mask=mask)
        assert logits.shape == (2, 2)
        # Second sequence's padding positions should have zero weight
        assert torch.allclose(
            weights[1, 0, 4:], torch.zeros(2), atol=1e-6
        )


# ---------------------------------------------------------------------------
# AttentionProbe init tests
# ---------------------------------------------------------------------------

class TestAttentionProbeInit:
    """Tests for AttentionProbe initialization."""

    def test_default_init(self):
        probe = AttentionProbe()
        assert probe.classifier is None
        assert probe.layer is None
        assert probe.d_model is None
        assert probe.d_head == 64
        assert probe.n_heads == 1
        assert probe.threshold == 0.5

    def test_init_with_params(self):
        probe = AttentionProbe(
            layer=24,
            d_model=4096,
            d_head=128,
            n_heads=4,
            threshold=0.6,
            accuracy=0.95,
        )
        assert probe.layer == 24
        assert probe.d_model == 4096
        assert probe.d_head == 128
        assert probe.n_heads == 4
        assert probe.accuracy == 0.95


# ---------------------------------------------------------------------------
# AttentionProbe training tests
# ---------------------------------------------------------------------------

class TestAttentionProbeTraining:
    """Tests for training attention probes."""

    def test_train_single_head(self):
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=60, n_neg=60, d_model=d_model
        )

        probe = AttentionProbe(layer=5)
        acc = probe.train(
            pos_seqs, neg_seqs,
            d_head=16, n_heads=1, num_epochs=200, lr=1e-3,
        )

        assert acc > 0.7
        assert probe.classifier is not None
        assert probe.d_model == d_model
        assert probe.n_heads == 1
        assert probe.accuracy == acc

    def test_train_multi_head(self):
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=60, n_neg=60, d_model=d_model
        )

        probe = AttentionProbe(layer=5)
        acc = probe.train(
            pos_seqs, neg_seqs,
            d_head=8, n_heads=4, num_epochs=200, lr=1e-3,
        )

        assert acc > 0.7
        assert probe.n_heads == 4

    def test_train_with_test_split(self):
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=50, n_neg=50, d_model=d_model
        )

        pos_train, pos_test = pos_seqs[:40], pos_seqs[40:]
        neg_train, neg_test = neg_seqs[:40], neg_seqs[40:]

        probe = AttentionProbe(layer=5)
        acc = probe.train(
            pos_train, neg_train, pos_test, neg_test,
            d_head=16, n_heads=1, num_epochs=200, lr=1e-3,
        )
        assert acc >= 0.0  # Just ensure it runs without error


# ---------------------------------------------------------------------------
# AttentionProbe scoring tests
# ---------------------------------------------------------------------------

class TestAttentionProbeScoring:
    """Tests for scoring with trained attention probes."""

    @pytest.fixture()
    def trained_probe(self):
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=40, n_neg=40, d_model=d_model
        )
        probe = AttentionProbe(layer=5)
        probe.train(pos_seqs, neg_seqs, d_head=16, n_heads=1, num_epochs=100)
        return probe

    def test_score_returns_probe_score(self, trained_probe):
        states = torch.randn(8, 32)  # (seq_len=8, d_model=32)
        score = trained_probe.score(states)
        assert 0.0 <= score.mean_score <= 1.0
        assert score.classification in ("eval_aware", "not_eval_aware")
        assert score.metadata["layer"] == 5
        assert score.metadata["n_heads"] == 1

    def test_score_with_batch_dim(self, trained_probe):
        states = torch.randn(1, 8, 32)  # (1, seq_len, d_model)
        score = trained_probe.score(states)
        assert 0.0 <= score.mean_score <= 1.0

    def test_score_includes_attention_weights(self, trained_probe):
        states = torch.randn(10, 32)
        score = trained_probe.score(states)
        assert "attention_weights" in score.metadata
        weights = score.metadata["attention_weights"]
        # weights should be (n_heads, seq_len) as nested list
        assert len(weights) == 1  # 1 head
        assert len(weights[0]) == 10  # seq_len=10

    def test_score_variable_length(self, trained_probe):
        """Score inputs of different lengths."""
        for seq_len in [3, 10, 20]:
            states = torch.randn(seq_len, 32)
            score = trained_probe.score(states)
            assert 0.0 <= score.mean_score <= 1.0

    def test_score_with_mask(self, trained_probe):
        states = torch.randn(8, 32)
        mask = torch.ones(8)
        mask[5:] = 0  # last 3 tokens are padding
        score = trained_probe.score(states, attention_mask=mask)
        assert 0.0 <= score.mean_score <= 1.0

    def test_score_no_classifier_raises(self):
        probe = AttentionProbe()
        with pytest.raises(ValueError, match="not trained"):
            probe.score(torch.randn(5, 32))

    def test_last_attention_weights_property(self, trained_probe):
        states = torch.randn(7, 32)
        trained_probe.score(states)
        assert trained_probe.last_attention_weights is not None
        assert trained_probe.last_attention_weights.shape == (1, 1, 7)


# ---------------------------------------------------------------------------
# AttentionProbe save/load tests
# ---------------------------------------------------------------------------

class TestAttentionProbeSaveLoad:
    """Tests for save/load functionality."""

    def test_save_load_roundtrip(self, tmp_path):
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=40, n_neg=40, d_model=d_model
        )

        probe = AttentionProbe(layer=5, threshold=0.6)
        probe.train(pos_seqs, neg_seqs, d_head=16, n_heads=2, num_epochs=100)

        save_path = tmp_path / "attn_probe.pth"
        probe.save(save_path)

        assert save_path.exists()
        assert save_path.with_suffix(".json").exists()

        # Verify metadata JSON
        with open(save_path.with_suffix(".json")) as f:
            meta = json.load(f)
        assert meta["probe_type"] == "attention"
        assert meta["layer"] == 5
        assert meta["d_model"] == 32
        assert meta["d_head"] == 16
        assert meta["n_heads"] == 2
        assert meta["threshold"] == 0.6

        # Load and verify
        loaded = AttentionProbe.load(save_path)
        assert loaded.layer == 5
        assert loaded.d_model == 32
        assert loaded.d_head == 16
        assert loaded.n_heads == 2
        assert loaded.threshold == 0.6

        # Verify scoring works on loaded probe
        states = torch.randn(8, d_model)
        score = loaded.score(states)
        assert 0.0 <= score.mean_score <= 1.0

    def test_save_load_preserves_weights(self, tmp_path):
        """Loaded probe should produce same scores as original."""
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=30, n_neg=30, d_model=d_model
        )

        probe = AttentionProbe(layer=5)
        probe.train(pos_seqs, neg_seqs, d_head=8, n_heads=1, num_epochs=50)

        save_path = tmp_path / "probe.pth"
        probe.save(save_path)
        loaded = AttentionProbe.load(save_path)

        test_input = torch.randn(6, d_model)
        orig_score = probe.score(test_input)
        loaded_score = loaded.score(test_input)
        assert abs(orig_score.mean_score - loaded_score.mean_score) < 1e-5

    def test_load_no_d_model_raises(self, tmp_path):
        """Loading without d_model in metadata or argument should fail."""
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=20, n_neg=20, d_model=d_model
        )

        probe = AttentionProbe(layer=5)
        probe.train(pos_seqs, neg_seqs, d_head=8, n_heads=1, num_epochs=20)

        save_path = tmp_path / "probe.pth"
        probe.save(save_path)

        # Remove d_model from metadata
        meta_path = save_path.with_suffix(".json")
        with open(meta_path) as f:
            meta = json.load(f)
        del meta["d_model"]
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="d_model required"):
            AttentionProbe.load(save_path)

    def test_load_with_explicit_d_model(self, tmp_path):
        """d_model can be passed explicitly to override metadata."""
        d_model = 32
        pos_seqs, neg_seqs = _make_separable_sequences(
            n_pos=20, n_neg=20, d_model=d_model
        )

        probe = AttentionProbe(layer=5)
        probe.train(pos_seqs, neg_seqs, d_head=8, n_heads=1, num_epochs=20)

        save_path = tmp_path / "probe.pth"
        probe.save(save_path)

        loaded = AttentionProbe.load(save_path, d_model=32)
        assert loaded.d_model == 32
