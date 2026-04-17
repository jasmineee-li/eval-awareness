import re

from utils.evaluated_model import get_evaluated_model_name


def get_successor_model_name() -> str:
    evaluated_model = get_evaluated_model_name()
    match = re.search(r"\d+(?:\.\d+)?", evaluated_model)
    if not match:
        return evaluated_model

    major_version = int(match.group(0).split(".", 1)[0])
    successor_version = str(major_version + 1)
    return f"{evaluated_model[:match.start()]}{successor_version}{evaluated_model[match.end():]}"