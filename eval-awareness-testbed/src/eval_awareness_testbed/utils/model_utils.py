"""Model name utilities — provider stripping, directory-safe names, and hardware detection."""

import os

KNOWN_PROVIDERS = ["anthropic", "minimax", "moonshot", "openai", "openrouter"]


def strip_provider_prefix(model: str) -> str:
    """Strip known provider prefix from model string.

    'openrouter/qwen/qwq-32b' → 'qwen/qwq-32b'
    'claude-sonnet-4' → 'claude-sonnet-4'
    """
    if "/" in model:
        prefix = model.split("/")[0]
        if prefix in KNOWN_PROVIDERS:
            return model[len(prefix) + 1 :]
    return model


def model_to_dirname(model: str) -> str:
    """Convert model string to directory-safe name (provider stripped, / → -).

    'openrouter/qwen/qwq-32b' → 'qwen-qwq-32b'
    'claude-sonnet-4' → 'claude-sonnet-4'
    """
    return strip_provider_prefix(model).replace("/", "-")


def detect_hardware() -> dict:
    """Detect available CPU and GPU resources.

    Returns dict with keys: cpu_count, gpu_available, gpu_count, gpu_names.
    """
    result = {
        "cpu_count": os.cpu_count() or 1,
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
    }
    try:
        import torch
        result["gpu_available"] = torch.cuda.is_available()
        if result["gpu_available"]:
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(result["gpu_count"])
            ]
    except ImportError:
        pass
    return result
