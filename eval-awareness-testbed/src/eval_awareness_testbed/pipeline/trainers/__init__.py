"""Training method registry."""

from eval_awareness_testbed.pipeline.trainers.base import BaseTrainer, TrainingResult
from eval_awareness_testbed.pipeline.trainers.sft import SFTTrainer
from eval_awareness_testbed.pipeline.trainers.dpo import DPOTrainer
from eval_awareness_testbed.pipeline.trainers.context_distillation import ContextDistillationTrainer

_TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "sft": SFTTrainer,
    "dpo": DPOTrainer,
    "context_distillation": ContextDistillationTrainer,
    "cd": ContextDistillationTrainer,
}


def get_trainer(method: str, **kwargs) -> BaseTrainer:
    """Get a trainer instance by method name."""
    if method not in _TRAINER_REGISTRY:
        available = ", ".join(_TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown training method: {method}. Available: {available}")
    return _TRAINER_REGISTRY[method](**kwargs)


__all__ = [
    "BaseTrainer",
    "TrainingResult",
    "SFTTrainer",
    "DPOTrainer",
    "ContextDistillationTrainer",
    "get_trainer",
]
