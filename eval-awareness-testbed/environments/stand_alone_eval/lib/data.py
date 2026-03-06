import json
import logging
import pathlib
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar

import yaml
from pydantic import BaseModel
from pydantic.dataclasses import is_pydantic_dataclass

logger = logging.getLogger("ape")

T = TypeVar("T")

# Dataclasses


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


T_DATACLASS = TypeVar("T_DATACLASS", bound=Dataclass)


def convert_dataclass_to_dict(
    dataclass: Dataclass,
    exclude_none: bool = False,
    exclude_falsy: bool = False,
) -> dict[str, Any]:
    assert not (exclude_falsy and exclude_none), "Can't exclude both falsy and none"
    if exclude_none:
        return asdict(dataclass, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
    elif exclude_falsy:
        return asdict(dataclass, dict_factory=lambda x: {k: v for (k, v) in x if v})
    else:
        return asdict(dataclass)


class ApexJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder for Apex-specific data types.

    This encoder extends the default JSON encoder to handle:
    1. pathlib.Path objects: converts them to POSIX-style string paths
    2. Dataclass instances: converts them to JSON-serializable dictionaries

    For all other types, it falls back to the default JSON encoder behavior.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def default(self, o: object) -> Any:
        if isinstance(o, pathlib.Path):
            return o.as_posix()
        elif is_dataclass(o) and not isinstance(o, type):
            return json.dumps(convert_dataclass_to_dict(o))
        return super().default(o)


# Filetype: jsonl


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    objects = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            objects.append(json.loads(line))
    return objects


def save_jsonl(path: str | Path, data: Sequence[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_jsonl_to_dataclasses(path: str | Path, dataclass: type[T], allow_failures: bool = False) -> Sequence[T]:
    assert is_pydantic_dataclass(dataclass), (
        "This function only supports pydantic dataclasses. "
        "The current implementation would fail to load nested non-pydantic dataclasses."
    )
    outputs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if allow_failures:
                try:
                    outputs.append(dataclass(**json.loads(line)))
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line.strip()}")
                    logger.warning(e)
            else:
                outputs.append(dataclass(**json.loads(line)))
    return outputs


def save_dataclasses_to_jsonl(
    path: str | Path,
    data: Sequence[Dataclass],
    exclude_none: bool = False,
    exclude_falsy: bool = False,
    mode: str = "w",
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for line in data:
            as_dict = convert_dataclass_to_dict(
                line,
                exclude_none=exclude_none,
                exclude_falsy=exclude_falsy,
            )
            f.write(json.dumps(as_dict, cls=ApexJSONEncoder) + "\n")


# Filetype: yaml


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_yaml_to_dataclass(path: str | Path, dataclass: type[T], **kwargs: Any) -> T:
    return dataclass(**kwargs, **load_yaml(path))


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_to_basemodel(path: str | Path, model_class: type[ModelT]) -> ModelT:
    return model_class(**load_yaml(path))


# Filetype: json


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json_to_dataclass(path: str | Path, dataclass: type[T]) -> T:
    return dataclass(**load_json(path))
