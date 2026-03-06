"""Contrastive steering vector probes for eval awareness detection.

Implements the methodology from:
- Nguyen et al. 2025 (arXiv:2507.01786) "Probing and Steering Evaluation Awareness"
- Chaudhary et al. 2025 (arXiv:2509.13333) "Evaluation Awareness Scales Predictably"

Training: steering_vector[layer] = mean(test_activations) - mean(deploy_activations)
Scoring: score = dot(activation, normalized_vector) / ||vector||

Adapted from:
- evaluation-awareness-scaling-laws/src/vector_generation.py
- evaluation-awareness-scaling-laws/src/probe_analysis.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from eval_awareness_probes.activation_extraction import (
    TokenSelector,
    extract_activations_hooked,
    extract_contrastive_activations,
)
from eval_awareness_probes.scoring import ProbeScore, classify_score

logger = logging.getLogger(__name__)


class ContrastiveProbe:
    """Contrastive steering vector probe for eval awareness.

    Trains a direction in activation space that separates 'eval-aware' from
    'not eval-aware' representations. At inference, projects activations onto
    this direction to produce a scalar score.
    """

    def __init__(
        self,
        vectors: dict[int, torch.Tensor] | None = None,
        normalized_vectors: dict[int, torch.Tensor] | None = None,
        best_layer: int | None = None,
        threshold: float = 0.0,
        metadata: dict | None = None,
    ):
        """Initialize a contrastive probe.

        Args:
            vectors: Raw steering vectors per layer (mean_test - mean_deploy).
            normalized_vectors: Unit-normalized vectors per layer.
            best_layer: Layer with highest AUROC.
            threshold: Classification threshold.
            metadata: Additional info (model name, training data, etc).
        """
        self.vectors = vectors or {}
        self.normalized_vectors = normalized_vectors or {}
        self.best_layer = best_layer
        self.threshold = threshold
        self.metadata = metadata or {}

    def train(
        self,
        model,
        contrastive_dataset: list[dict],
        layers: list[int] | None,
        tokenizer,
    ) -> None:
        """Train steering vectors from a contrastive dataset.

        For each contrastive pair, extracts activations at the answer token
        position and computes mean(positive) - mean(negative) per layer.

        Args:
            model: HookedTransformer or HookedTransformerShim model.
            contrastive_dataset: List of dicts with 'question', 'positive'/'test',
                'negative'/'deploy' keys.
            layers: Layer indices to train (None = all layers).
            tokenizer: Model tokenizer.
        """
        if layers is None:
            layers = list(range(model.cfg.n_layers))

        logger.info(
            f"Training contrastive probe on {len(contrastive_dataset)} examples, "
            f"{len(layers)} layers"
        )

        positive_acts, negative_acts = extract_contrastive_activations(
            model, contrastive_dataset, layers, tokenizer
        )

        self.vectors = {}
        self.normalized_vectors = {}

        for layer in layers:
            pos_list = positive_acts[layer]
            neg_list = negative_acts[layer]

            if not pos_list or not neg_list:
                logger.warning(f"Layer {layer}: missing activations, skipping")
                continue

            pos_mean = torch.stack(pos_list).mean(dim=0)
            neg_mean = torch.stack(neg_list).mean(dim=0)
            vector = pos_mean - neg_mean
            self.vectors[layer] = vector

            norm = vector.norm()
            if norm > 0:
                self.normalized_vectors[layer] = vector / norm
            else:
                self.normalized_vectors[layer] = vector

        logger.info(f"Trained vectors for {len(self.vectors)} layers")

    def score(
        self,
        model,
        text: str,
        layer: int | None = None,
        tokenizer=None,
    ) -> ProbeScore:
        """Score a text for eval awareness using the steering vector probe.

        Projects per-token activations onto the normalized steering vector
        and computes aggregate statistics.

        Args:
            model: HookedTransformer or HookedTransformerShim model.
            text: Input text to score.
            layer: Layer to score at (default: best_layer).
            tokenizer: Tokenizer.

        Returns:
            ProbeScore with mean, max, min, std scores and classification.
        """
        if layer is None:
            layer = self.best_layer
        if layer is None:
            raise ValueError("No layer specified and no best_layer set")

        if layer not in self.normalized_vectors:
            raise ValueError(f"No vector for layer {layer}")

        vector = self.normalized_vectors[layer]

        tokens = model.to_tokens(text)

        # Extract per-token scores by hooking into the forward pass
        per_token_scores: list[float] = []

        def hook_fn(activation, hook):
            for pos in range(activation.shape[1]):
                act = activation[0, pos, :]
                projection = torch.dot(act, vector[0] if vector.dim() > 1 else vector)
                proj_val = projection / vector.norm() if vector.dim() > 1 else projection.item()
                if isinstance(proj_val, torch.Tensor):
                    proj_val = proj_val.item()
                if pos == len(per_token_scores):
                    per_token_scores.append(proj_val)
            return activation

        model.reset_hooks()
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
            with torch.no_grad():
                model(tokens)

        # Filter out BOS / padding tokens
        if tokenizer is not None:
            token_strings = [tokenizer.decode([t]) for t in tokens[0].tolist()]
            filtered = [
                s
                for s, tok in zip(per_token_scores, token_strings)
                if tok.strip()
                and "<|begin_of_text|>" not in tok
                and "<s>" not in tok
                and "<BOS>" not in tok
            ]
        else:
            filtered = per_token_scores

        if not filtered:
            filtered = per_token_scores if per_token_scores else [0.0]

        mean_score = sum(filtered) / len(filtered)
        max_score = max(filtered)
        min_score = min(filtered)
        std_score = (
            sum((x - mean_score) ** 2 for x in filtered) / len(filtered)
        ) ** 0.5

        classification, confidence = classify_score(mean_score, self.threshold)

        return ProbeScore(
            mean_score=mean_score,
            max_score=max_score,
            min_score=min_score,
            std_score=std_score,
            classification=classification,
            confidence=confidence,
            per_token_scores=filtered,
            metadata={"layer": layer},
        )

    def save(self, output_dir: str | Path) -> None:
        """Save probe vectors and metadata to disk.

        Args:
            output_dir: Directory to save vectors.
        """
        output_dir = Path(output_dir)
        vectors_dir = output_dir / "vectors"
        norm_dir = output_dir / "normalized_vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        norm_dir.mkdir(parents=True, exist_ok=True)

        for layer, vector in self.vectors.items():
            torch.save(vector, vectors_dir / f"layer_{layer}.pt")

        for layer, vector in self.normalized_vectors.items():
            torch.save(vector, norm_dir / f"layer_{layer}.pt")

        # Save metadata
        meta = {
            "best_layer": self.best_layer,
            "threshold": self.threshold,
            "num_layers": len(self.vectors),
            "layers": sorted(self.vectors.keys()),
            **self.metadata,
        }
        with open(output_dir / "probe_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved probe to {output_dir}")

    @classmethod
    def load(cls, probe_dir: str | Path) -> ContrastiveProbe:
        """Load a saved probe from disk.

        Args:
            probe_dir: Directory containing vectors/ and normalized_vectors/.

        Returns:
            Loaded ContrastiveProbe.
        """
        probe_dir = Path(probe_dir)

        # Load normalized vectors (preferred for scoring)
        norm_dir = probe_dir / "normalized_vectors"
        vectors_dir = probe_dir / "vectors"

        normalized_vectors = {}
        vectors = {}

        if norm_dir.exists():
            for f in sorted(norm_dir.glob("layer_*.pt")):
                layer = int(f.stem.split("_")[1])
                normalized_vectors[layer] = torch.load(f, map_location="cpu", weights_only=True)

        if vectors_dir.exists():
            for f in sorted(vectors_dir.glob("layer_*.pt")):
                layer = int(f.stem.split("_")[1])
                vectors[layer] = torch.load(f, map_location="cpu", weights_only=True)

        # If only raw vectors exist, normalize them
        if not normalized_vectors and vectors:
            for layer, v in vectors.items():
                norm = v.norm()
                normalized_vectors[layer] = v / norm if norm > 0 else v

        # Load metadata
        metadata = {}
        best_layer = None
        threshold = 0.0
        info_path = probe_dir / "probe_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            best_layer = info.get("best_layer")
            threshold = info.get("threshold", 0.0)
            metadata = {k: v for k, v in info.items() if k not in ("best_layer", "threshold")}

        logger.info(
            f"Loaded contrastive probe from {probe_dir} "
            f"({len(normalized_vectors)} layers, best_layer={best_layer})"
        )

        return cls(
            vectors=vectors,
            normalized_vectors=normalized_vectors,
            best_layer=best_layer,
            threshold=threshold,
            metadata=metadata,
        )

    @property
    def layers(self) -> list[int]:
        """List of available layers."""
        return sorted(set(self.vectors.keys()) | set(self.normalized_vectors.keys()))
