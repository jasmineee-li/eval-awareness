"""Learned attention probes for eval awareness detection.

Implements the EleutherAI / McKenzie et al. 2025 attention probe architecture:
instead of fixed pooling (mean/last-token) over hidden states, a small learned
attention mechanism dynamically weights token positions before classification.

Architecture:
    Input: per-token hidden states (seq_len, d_model) from a single layer
    Attention pooling: W_q (zero-init), W_v (random), learnable query vector
    Classification: linear head on attended representation → binary cross-entropy

References:
- EleutherAI probes: zero-init Q, random V, AdamW, 1-8 heads
- McKenzie et al. 2025: single-head variant, no position bias
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from eval_awareness_probes.scoring import ProbeScore, classify_score

logger = logging.getLogger(__name__)


class _AttentionPooler(nn.Module):
    """Learned attention pooling over token positions.

    Computes cross-attention between a learnable query vector and per-token
    hidden states to produce a fixed-size representation regardless of
    sequence length.

    The query projection W_q is initialized to zero so that attention starts
    uniform across all positions, then learns which positions carry signal.
    """

    def __init__(self, d_model: int, d_head: int = 64, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.query = nn.Parameter(torch.zeros(n_heads, d_head))

        # Zero-init W_q so attention starts uniform (EleutherAI convention)
        nn.init.zeros_(self.W_q.weight)
        # Random init for W_v
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-pooled representation.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len), 1 for real tokens, 0 for padding.

        Returns:
            pooled: (batch, n_heads * d_head) — concatenated attended representations.
            weights: (batch, n_heads, seq_len) — attention weights per head.
        """
        batch, seq_len, _ = hidden_states.shape

        # Project keys and values: (batch, seq_len, n_heads * d_head)
        keys = self.W_q(hidden_states)
        values = self.W_v(hidden_states)

        # Reshape to (batch, seq_len, n_heads, d_head) → (batch, n_heads, seq_len, d_head)
        keys = keys.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        values = values.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # query: (n_heads, d_head) → (1, n_heads, d_head, 1) for batched matmul
        q = self.query.unsqueeze(0).unsqueeze(-1)

        # Attention scores: (batch, n_heads, seq_len, d_head) @ (1, n_heads, d_head, 1)
        # → (batch, n_heads, seq_len, 1)
        scores = torch.matmul(keys, q).squeeze(-1)  # (batch, n_heads, seq_len)
        scores = scores / math.sqrt(self.d_head)

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) → (batch, 1, seq_len)
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq_len)

        # Handle all-padding edge case (softmax of all -inf → nan)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: (batch, n_heads, 1, seq_len) @ (batch, n_heads, seq_len, d_head)
        # → (batch, n_heads, 1, d_head)
        attended = torch.matmul(weights.unsqueeze(2), values).squeeze(2)

        # Concatenate heads: (batch, n_heads * d_head)
        pooled = attended.reshape(batch, self.n_heads * self.d_head)

        return pooled, weights


class _AttentionProbeClassifier(nn.Module):
    """Full attention probe: attention pooler + linear classification head."""

    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        n_heads: int = 1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.pooler = _AttentionPooler(d_model, d_head, n_heads)
        self.classifier = nn.Linear(d_head * n_heads, n_classes)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, n_classes)
            attention_weights: (batch, n_heads, seq_len)
        """
        pooled, weights = self.pooler(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        return logits, weights


class _SequenceDataset(Dataset):
    """Dataset for variable-length hidden state sequences."""

    def __init__(
        self,
        sequences: list[torch.Tensor],
        labels: list[int],
    ):
        """
        Args:
            sequences: List of tensors, each (seq_len_i, d_model).
            labels: Binary labels.
        """
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def _collate_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns:
        padded: (batch, max_seq_len, d_model)
        masks: (batch, max_seq_len) — 1 for real tokens, 0 for padding.
        labels: (batch,)
    """
    sequences, labels = zip(*batch)
    masks = [torch.ones(seq.shape[0]) for seq in sequences]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded, masks, labels


class AttentionProbe:
    """Learned attention probe for eval awareness detection.

    Uses a small learned attention mechanism to dynamically weight token
    positions when aggregating hidden states from a single layer, then
    classifies the resulting representation as eval-aware or not.

    Unlike ClassifierProbe which takes pre-pooled (d_model,) vectors, this
    probe takes per-token hidden states (seq_len, d_model) and learns which
    positions to attend to.
    """

    def __init__(
        self,
        classifier: _AttentionProbeClassifier | None = None,
        layer: int | None = None,
        d_model: int | None = None,
        d_head: int = 64,
        n_heads: int = 1,
        threshold: float = 0.5,
        accuracy: float | None = None,
        metadata: dict | None = None,
    ):
        self.classifier = classifier
        self.layer = layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.threshold = threshold
        self.accuracy = accuracy
        self.metadata = metadata or {}
        self._last_attention_weights: torch.Tensor | None = None

    def train(
        self,
        positive_states: list[torch.Tensor],
        negative_states: list[torch.Tensor],
        positive_test_states: list[torch.Tensor] | None = None,
        negative_test_states: list[torch.Tensor] | None = None,
        d_head: int = 64,
        n_heads: int = 1,
        lr: float = 1e-3,
        num_epochs: int = 500,
        batch_size: int = 32,
    ) -> float:
        """Train the attention probe on variable-length hidden state sequences.

        Args:
            positive_states: List of eval-aware hidden states, each (seq_len_i, d_model).
            negative_states: List of non-aware hidden states, each (seq_len_i, d_model).
            positive_test_states: Optional test set positive states.
            negative_test_states: Optional test set negative states.
            d_head: Dimension per attention head.
            n_heads: Number of attention heads.
            lr: Learning rate for AdamW.
            num_epochs: Training epochs.
            batch_size: Batch size.

        Returns:
            Test accuracy (or training accuracy if no test set).
        """
        self.d_head = d_head
        self.n_heads = n_heads

        # Infer d_model from data
        self.d_model = positive_states[0].shape[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _AttentionProbeClassifier(
            d_model=self.d_model,
            d_head=d_head,
            n_heads=n_heads,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Build training data
        train_seqs = [s.to(device) for s in positive_states] + [s.to(device) for s in negative_states]
        train_labels = [1] * len(positive_states) + [0] * len(negative_states)
        train_dataset = _SequenceDataset(train_seqs, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_sequences,
        )

        # Train
        model.train()
        for _epoch in range(num_epochs):
            for padded, masks, labels in train_loader:
                padded = padded.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                logits, _ = model(padded, masks)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()

        if positive_test_states is not None and negative_test_states is not None:
            test_seqs = [s.to(device) for s in positive_test_states] + [s.to(device) for s in negative_test_states]
            test_labels = [1] * len(positive_test_states) + [0] * len(negative_test_states)
        else:
            test_seqs = train_seqs
            test_labels = train_labels

        test_dataset = _SequenceDataset(test_seqs, test_labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_sequences,
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for padded, masks, labels in test_loader:
                padded = padded.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                logits, _ = model(padded, masks)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.classifier = model.cpu()
        acc = correct / total if total > 0 else 0.0
        self.accuracy = acc
        logger.info(
            f"Trained attention probe (d_head={d_head}, n_heads={n_heads}), "
            f"accuracy: {acc:.4f}"
        )
        return acc

    def score(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ProbeScore:
        """Score per-token hidden states for eval awareness.

        Args:
            states: Hidden state tensor of shape (seq_len, d_model) or
                (1, seq_len, d_model).
            attention_mask: Optional mask of shape (seq_len,) or (1, seq_len).

        Returns:
            ProbeScore with classification, confidence, and attention weights.
        """
        if self.classifier is None:
            raise ValueError("Probe not trained or loaded")

        if states.dim() == 2:
            states = states.unsqueeze(0)  # (1, seq_len, d_model)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        self.classifier.eval()
        with torch.no_grad():
            logits, weights = self.classifier(states.cpu(), attention_mask)
            proba = torch.softmax(logits, dim=1)[0]
            score_val = float(proba[1])

        self._last_attention_weights = weights.detach()

        classification, confidence = classify_score(score_val, self.threshold)

        return ProbeScore(
            mean_score=score_val,
            max_score=score_val,
            min_score=score_val,
            std_score=0.0,
            classification=classification,
            confidence=confidence,
            metadata={
                "layer": self.layer,
                "n_heads": self.n_heads,
                "d_head": self.d_head,
                "attention_weights": weights[0].tolist(),
            },
        )

    @property
    def last_attention_weights(self) -> torch.Tensor | None:
        """Attention weights from the most recent score() call."""
        return self._last_attention_weights

    def save(self, output_path: str | Path) -> None:
        """Save the trained attention probe to disk.

        Args:
            output_path: File path for the model weights (.pth).
        """
        if self.classifier is None:
            raise ValueError("No trained classifier to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.classifier.state_dict(), output_path)

        meta_path = output_path.with_suffix(".json")
        meta = {
            "probe_type": "attention",
            "layer": self.layer,
            "d_model": self.d_model,
            "d_head": self.d_head,
            "n_heads": self.n_heads,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            **self.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved attention probe to {output_path}")

    @classmethod
    def load(cls, model_path: str | Path, d_model: int | None = None) -> AttentionProbe:
        """Load a saved attention probe.

        Args:
            model_path: Path to saved weights (.pth).
            d_model: Hidden state dimension. Read from metadata if available.

        Returns:
            Loaded AttentionProbe.
        """
        model_path = Path(model_path)

        meta_path = model_path.with_suffix(".json")
        metadata = {}
        layer = None
        d_head = 64
        n_heads = 1
        threshold = 0.5
        accuracy = None

        if meta_path.exists():
            with open(meta_path) as f:
                info = json.load(f)
            layer = info.get("layer")
            d_model = info.get("d_model", d_model)
            d_head = info.get("d_head", 64)
            n_heads = info.get("n_heads", 1)
            threshold = info.get("threshold", 0.5)
            accuracy = info.get("accuracy")
            metadata = {
                k: v
                for k, v in info.items()
                if k not in (
                    "probe_type", "layer", "d_model", "d_head",
                    "n_heads", "threshold", "accuracy",
                )
            }

        if d_model is None:
            raise ValueError(
                "d_model required for loading attention probe "
                "(not found in metadata)"
            )

        classifier = _AttentionProbeClassifier(
            d_model=d_model, d_head=d_head, n_heads=n_heads
        )
        classifier.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )

        logger.info(
            f"Loaded attention probe from {model_path} "
            f"(layer={layer}, d_model={d_model}, n_heads={n_heads})"
        )

        return cls(
            classifier=classifier,
            layer=layer,
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            threshold=threshold,
            accuracy=accuracy,
            metadata=metadata,
        )
