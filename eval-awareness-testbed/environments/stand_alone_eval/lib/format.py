from typing import Any


def format(
    template: str,
    ensure_keys_in_template: bool = True,
    preserve_escaped_braces: bool = False,
    **kwargs: Any,
) -> str:
    """Format a string, with some extended options.

    Args:
        ensure_keys_in_template: raise an error if there are keys supplied in kwargs that do not exist in the template.
        Note that, on the contrary, if there are placeholders in the template that are not supplied in kwargs, this
        function will never raise an error.
        preserve_literal_braces: preserve escaped braces in the template, i.e. {{foo}} will not be replaced with {foo}.
    """
    used_keys = set()

    class KeyTracker(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            used_keys.add(key)
            if key in self:
                return super().__getitem__(key)
            if preserve_escaped_braces:
                return "{" + key + "}"
            return "{{" + key + "}}"

    if preserve_escaped_braces:
        template = template.replace("{{", "<<DOUBLE_OPEN>>").replace("}}", "<<DOUBLE_CLOSE>>")

    result = template.format_map(KeyTracker(kwargs))

    if preserve_escaped_braces:
        result = result.replace("<<DOUBLE_OPEN>>", "{{").replace("<<DOUBLE_CLOSE>>", "}}")

    unused_keys = set(kwargs.keys()) - used_keys
    if ensure_keys_in_template and unused_keys:
        raise ValueError(f"Unused keyword arguments: {', '.join(unused_keys)}")

    return result
