import re

from utils.full_model_name import full_model_name


def full_successor_model_name() -> str:
    model_id = full_model_name()
    match = re.search(r"\d+(?:\.\d+)?", model_id)
    if not match:
        return model_id

    major_version = int(match.group(0).split(".", 1)[0])
    successor_version = str(major_version + 1)
    return f"{model_id[:match.start()]}{successor_version}{model_id[match.end():]}"
