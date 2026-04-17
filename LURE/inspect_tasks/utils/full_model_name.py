from inspect_ai.model import get_model


def full_model_name() -> str:
    model = get_model()
    model_id = None
    for attr in ("model", "name", "id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            model_id = value.strip()
            break
    if not model_id:
        model_id = str(model).strip()

    parts = model_id.split("/")
    if len(parts) < 2:
        return parts[-1]
    return "/".join(parts[-2:])
