REASONING_KEYS = ("reasoning", "thinking", "analysis", "chain_of_thought", "cot")
REASONING_SUMMARY_KEYS = ("reasoning_summary", "thinking_summary", "summary")


def _get_attr(item, key):
    return getattr(item, key, None) or (item.get(key) if isinstance(item, dict) else None)


def extract_reasoning(output) -> str:
    """Extract reasoning, preferring summary over redacted/encrypted content."""
    if not output:
        return ""
    msg = getattr(output, "message", None)
    if not msg:
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for item in content:
            # If redacted, use summary instead
            if _get_attr(item, "redacted"):
                summary = _get_attr(item, "summary")
                if summary:
                    return str(summary).strip()
                continue
            for k in REASONING_SUMMARY_KEYS + REASONING_KEYS:
                if v := _get_attr(item, k):
                    return str(v).strip()
    for k in REASONING_SUMMARY_KEYS + REASONING_KEYS:
        if v := getattr(msg, k, None):
            return str(v).strip()
    md = getattr(msg, "metadata", None)
    if isinstance(md, dict):
        for k in REASONING_SUMMARY_KEYS + REASONING_KEYS:
            if md.get(k):
                return str(md[k]).strip()
    return ""

