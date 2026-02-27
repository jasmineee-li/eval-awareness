"""Binary classifier probes for eval awareness detection.

Implements the methodology from:
- Abdelnabi & Salem 2025 (arXiv:2505.14617) "The Hawthorne Effect in Reasoning Models"

Trains a binary classifier (LogisticRegression or MLP) per layer to distinguish
'eval-aware' from 'eval-unaware' hidden states.

Adapted from:
- Test_Awareness_Steering/scripts/probe.py (LogisticRegression)
- Test_Awareness_Steering/scripts/mlp_train.py (MLP)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from eval_awareness_probes.scoring import ProbeScore, classify_score

logger = logging.getLogger(__name__)


class _RepDataset(Dataset):
    """Simple dataset for hidden state classification."""

    def __init__(self, data: torch.Tensor, labels: list[int]):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class _MLPClassifier(nn.Module):
    """Single-layer linear classifier (matches Paper 3 architecture)."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ClassifierProbe:
    """Binary classifier probe for eval awareness detection.

    Supports two classifier types:
    - 'logistic_regression': sklearn LogisticRegression (faster, simpler)
    - 'mlp': Single-layer linear classifier via PyTorch (same architecture as Paper 3)
    """

    def __init__(
        self,
        classifier=None,
        classifier_type: str = "logistic_regression",
        layer: int | None = None,
        threshold: float = 0.5,
        accuracy: float | None = None,
        metadata: dict | None = None,
    ):
        self.classifier = classifier
        self.classifier_type = classifier_type
        self.layer = layer
        self.threshold = threshold
        self.accuracy = accuracy
        self.metadata = metadata or {}

    def train(
        self,
        positive_states: torch.Tensor,
        negative_states: torch.Tensor,
        positive_test_states: torch.Tensor | None = None,
        negative_test_states: torch.Tensor | None = None,
        classifier_type: str = "logistic_regression",
    ) -> float:
        """Train a binary classifier on hidden states.

        Args:
            positive_states: Eval-aware hidden states for training, shape (N, d_model).
            negative_states: Non-aware hidden states for training, shape (M, d_model).
            positive_test_states: Optional test set positive states.
            negative_test_states: Optional test set negative states.
            classifier_type: 'logistic_regression' or 'mlp'.

        Returns:
            Test accuracy (or training accuracy if no test set provided).
        """
        self.classifier_type = classifier_type

        X_train = torch.cat((positive_states, negative_states), dim=0)
        y_train = [1] * positive_states.size(0) + [0] * negative_states.size(0)

        if positive_test_states is not None and negative_test_states is not None:
            X_test = torch.cat((positive_test_states, negative_test_states), dim=0)
            y_test = [1] * positive_test_states.size(0) + [0] * negative_test_states.size(0)
        else:
            X_test = X_train
            y_test = y_train

        if classifier_type == "logistic_regression":
            acc = self._train_logreg(X_train, y_train, X_test, y_test)
        elif classifier_type == "mlp":
            acc = self._train_mlp(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.accuracy = acc
        logger.info(f"Trained {classifier_type} probe, accuracy: {acc:.4f}")
        return acc

    def _train_logreg(
        self,
        X_train: torch.Tensor,
        y_train: list[int],
        X_test: torch.Tensor,
        y_test: list[int],
    ) -> float:
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train.cpu().numpy(), y_train)
        y_pred = model.predict(X_test.cpu().numpy())
        acc = accuracy_score(y_test, y_pred)
        self.classifier = model
        return acc

    def _train_mlp(
        self,
        X_train: torch.Tensor,
        y_train: list[int],
        X_test: torch.Tensor,
        y_test: list[int],
        num_epochs: int = 300,
        lr: float = 0.008,
        batch_size: int = 128,
    ) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _MLPClassifier(X_train.size(1), 2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        train_dataset = _RepDataset(X_train.to(device), y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _epoch in range(num_epochs):
            for batch_data, batch_labels in train_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        test_dataset = _RepDataset(X_test.to(device), y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        self.classifier = model.cpu()
        return correct / total if total > 0 else 0.0

    def score(self, states: torch.Tensor) -> ProbeScore:
        """Score hidden states for eval awareness.

        Args:
            states: Hidden state tensor of shape (d_model,) or (1, d_model).

        Returns:
            ProbeScore with classification and confidence.
        """
        if self.classifier is None:
            raise ValueError("Probe not trained or loaded")

        if states.dim() == 1:
            states = states.unsqueeze(0)

        if self.classifier_type == "logistic_regression":
            proba = self.classifier.predict_proba(states.cpu().numpy())[0]
            score_val = float(proba[1])  # probability of positive class
        else:
            self.classifier.eval()
            with torch.no_grad():
                outputs = self.classifier(states.cpu())
                proba = torch.softmax(outputs, dim=1)[0]
                score_val = float(proba[1])

        classification, confidence = classify_score(score_val, self.threshold)

        return ProbeScore(
            mean_score=score_val,
            max_score=score_val,
            min_score=score_val,
            std_score=0.0,
            classification=classification,
            confidence=confidence,
            metadata={"layer": self.layer, "classifier_type": self.classifier_type},
        )

    def save(self, output_path: str | Path) -> None:
        """Save the trained classifier to disk.

        Args:
            output_path: File path (.pkl for LogReg, .pth for MLP).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.classifier_type == "logistic_regression":
            with open(output_path, "wb") as f:
                pickle.dump(self.classifier, f)
        else:
            torch.save(self.classifier.state_dict(), output_path)

        # Save metadata alongside
        meta_path = output_path.with_suffix(".json")
        meta = {
            "classifier_type": self.classifier_type,
            "layer": self.layer,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            **self.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {self.classifier_type} probe to {output_path}")

    @classmethod
    def load(cls, model_path: str | Path, input_dim: int | None = None) -> ClassifierProbe:
        """Load a saved classifier probe.

        Args:
            model_path: Path to saved classifier (.pkl or .pth).
            input_dim: Required for MLP loading (hidden state dimension).

        Returns:
            Loaded ClassifierProbe.
        """
        model_path = Path(model_path)

        # Load metadata
        meta_path = model_path.with_suffix(".json")
        metadata = {}
        classifier_type = "logistic_regression"
        layer = None
        threshold = 0.5
        accuracy = None

        if meta_path.exists():
            with open(meta_path) as f:
                info = json.load(f)
            classifier_type = info.get("classifier_type", "logistic_regression")
            layer = info.get("layer")
            threshold = info.get("threshold", 0.5)
            accuracy = info.get("accuracy")
            metadata = {
                k: v
                for k, v in info.items()
                if k not in ("classifier_type", "layer", "threshold", "accuracy")
            }

        if classifier_type == "logistic_regression" or model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                classifier = pickle.load(f)  # noqa: S301
            classifier_type = "logistic_regression"
        else:
            if input_dim is None:
                raise ValueError("input_dim required for loading MLP classifier")
            classifier = _MLPClassifier(input_dim, 2)
            classifier.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            classifier_type = "mlp"

        logger.info(
            f"Loaded {classifier_type} probe from {model_path} (layer={layer})"
        )

        return cls(
            classifier=classifier,
            classifier_type=classifier_type,
            layer=layer,
            threshold=threshold,
            accuracy=accuracy,
            metadata=metadata,
        )
