"""Registry for reasoning analyzers."""

from typing import Type

from eval_awareness_testbed.analyzers.base import BaseAnalyzer

_ANALYZER_REGISTRY: dict[str, Type[BaseAnalyzer]] = {}


def register_analyzer(name: str | None = None):
    """Decorator to register an analyzer class.

    Args:
        name: Optional name override. Defaults to class.name attribute.
    """
    def decorator(cls: Type[BaseAnalyzer]) -> Type[BaseAnalyzer]:
        analyzer_name = name or cls.name
        _ANALYZER_REGISTRY[analyzer_name] = cls
        return cls
    return decorator


def get_analyzer(name: str, **kwargs) -> BaseAnalyzer:
    """Get an analyzer instance by name.

    Args:
        name: Name of the analyzer.
        **kwargs: Configuration options passed to the analyzer.

    Returns:
        Instantiated analyzer.

    Raises:
        ValueError: If analyzer not found.
    """
    if name not in _ANALYZER_REGISTRY:
        available = ", ".join(_ANALYZER_REGISTRY.keys())
        raise ValueError(f"Unknown analyzer: {name}. Available: {available}")
    return _ANALYZER_REGISTRY[name](**kwargs)


def list_analyzers() -> list[str]:
    """List all registered analyzer names."""
    return list(_ANALYZER_REGISTRY.keys())
