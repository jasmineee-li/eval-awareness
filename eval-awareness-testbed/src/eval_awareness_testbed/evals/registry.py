"""Registry for evals."""

from typing import Type

from eval_awareness_testbed.evals.base import BaseEval

_EVAL_REGISTRY: dict[str, Type[BaseEval]] = {}


def register_eval(name: str | None = None):
    """Decorator to register an eval class.

    Args:
        name: Optional name override. Defaults to class.name attribute.
    """
    def decorator(cls: Type[BaseEval]) -> Type[BaseEval]:
        eval_name = name or cls.name
        _EVAL_REGISTRY[eval_name] = cls
        return cls
    return decorator


def get_eval(name: str, **kwargs) -> BaseEval:
    """Get an eval instance by name.

    Args:
        name: Name of the eval.
        **kwargs: Configuration options passed to the eval.

    Returns:
        Instantiated eval.

    Raises:
        ValueError: If eval not found.
    """
    if name not in _EVAL_REGISTRY:
        available = ", ".join(_EVAL_REGISTRY.keys())
        raise ValueError(f"Unknown eval: {name}. Available: {available}")
    return _EVAL_REGISTRY[name](**kwargs)


def list_evals() -> list[str]:
    """List all registered eval names."""
    return list(_EVAL_REGISTRY.keys())
