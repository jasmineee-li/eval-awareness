"""Model name utilities — provider stripping and directory-safe names."""

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
