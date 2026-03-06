"""Standalone probe library for evaluation awareness detection.

Supports three probe approaches:
1. Contrastive steering vectors (Nguyen et al. 2025, Chaudhary et al. 2025)
2. Trained binary classifiers (Abdelnabi & Salem 2025)
3. Learned attention probes (EleutherAI / McKenzie et al. 2025)

This package is self-contained and does not depend on eval_awareness_testbed.
"""

from eval_awareness_probes.scoring import ProbeScore

__all__ = [
    "ProbeScore",
]


def _lazy_import_contrastive():
    from eval_awareness_probes.contrastive_probe import ContrastiveProbe
    return ContrastiveProbe


def _lazy_import_classifier():
    from eval_awareness_probes.classifier_probe import ClassifierProbe
    return ClassifierProbe


def _lazy_import_attention():
    from eval_awareness_probes.attention_probe import AttentionProbe
    return AttentionProbe


def _lazy_import_registry():
    from eval_awareness_probes.registry import ProbeRegistry
    return ProbeRegistry


def __getattr__(name):
    """Lazy imports so torch is not required just to import the package."""
    if name == "ContrastiveProbe":
        return _lazy_import_contrastive()
    elif name == "ClassifierProbe":
        return _lazy_import_classifier()
    elif name == "AttentionProbe":
        return _lazy_import_attention()
    elif name == "ProbeRegistry":
        return _lazy_import_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
