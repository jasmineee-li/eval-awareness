"""Tooling for using the weird XML that LLMs like.

This module provides utilities for parsing XML-style tags commonly used in LLM outputs.
The XML format is not strict and is designed to be robust to the quirks of LLM-generated XML.

Examples:
    >>> text = "<action>Jump high</action>"
    >>> match = try_parse_xml_tag("action", text)
    >>> print(match.content)
    Jump high
    >>> print(match.end_pos)
    24

    # Also works with multiline content
    >>> text = '''<thought>
    ... I should be careful
    ... and look before jumping
    ... </thought>'''
    >>> match = try_parse_xml_tag("thought", text)
    >>> print(match.content)
    I should be careful
    and look before jumping

    # Parse XML into Pydantic models
    >>> from pydantic import BaseModel
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    >>> text = "<person><name>Alice</name><age>25</age></person>"
    >>> person = parse_xml_to_pydantic(text, Person)
    >>> print(person)
    Person(name='Alice', age=25)

    # Parse multiple instances
    >>> text = '''
    ... <person><name>Alice</name><age>25</age></person>
    ... <person><name>Bob</name><age>30</age></person>
    ... '''
    >>> people = parse_xml_to_pydantic(text, Person, allow_multiple=True)
    >>> print(people)
    [Person(name='Alice', age=25), Person(name='Bob', age=30)]
"""

import dataclasses
import logging
import re
import types
import typing
from collections.abc import Sequence

import pydantic

T = typing.TypeVar("T", bound=pydantic.BaseModel)

# type aliases
XMLString = str

logger = logging.getLogger(__name__)


class RequiredXMLTagNotFoundError(Exception):
    pass


@dataclasses.dataclass
class XMLTagMatch:
    """Result of parsing an XML tag from text.

    Attributes:
        content: The content between the opening and closing tags, if found.
        end_pos: The position after the closing tag in the text, if found.
    """

    content: str | None
    end_pos: int | None

    def is_found(self) -> bool:
        """Check if the tag was found in the text."""
        return self.content is not None and self.end_pos is not None


def try_parse_xml_tag(tag: str, text: str) -> XMLTagMatch:
    """Extract content between XML-style tags in a text string, including multiline content.

    Returns the first match if multiple are present. If not found, returns a match
    with content=None, end_pos=None.
    """
    if tag.startswith("<") or tag.endswith(">"):
        raise ValueError(
            f"Tag `{tag}` should not contain angle brackets; "
            f"pass the bare tag name (e.g. 'my_tag' instead of '<my_tag>')"
        )

    # DOTALL so that '.' matches newlines
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        return XMLTagMatch(content=None, end_pos=None)

    return XMLTagMatch(
        content=match.group(1).strip(),
        end_pos=match.end(),
    )


def parse_xml_tag(tag: str, text: str) -> XMLTagMatch:
    """Extract content between XML tags, raising if not found."""
    match = try_parse_xml_tag(tag, text)
    if not match.is_found():
        raise RequiredXMLTagNotFoundError(f"No XML tag found for `{tag}` in text:\n\n{text}")
    return match


def pydantic_to_xml(instance: pydantic.BaseModel) -> XMLString:
    """Convert a Pydantic model to an XML string.

    Handles:
      - Primitive lists, each item wrapped in <item>.
      - Lists of Pydantic models, each flattened into a collection of tags.
    """
    json_dict = instance.model_dump()

    def _dict_to_xml_tags(d: dict[str, typing.Any]) -> str:
        return "".join(f"<{k}>{v}</{k}>" for k, v in d.items())

    xml_parts = []
    for key, value in json_dict.items():
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Lists of dicts/models
                inner_tags = "\n".join(_dict_to_xml_tags(item) for item in value)
                xml_parts.append(f"<{key}>\n{inner_tags}\n</{key}>")
            else:
                # Lists of primitives
                inner_tags = "\n".join(f"<item>{item}</item>" for item in value)
                xml_parts.append(f"<{key}>\n{inner_tags}\n</{key}>")
        else:
            xml_parts.append(f"<{key}>{value}</{key}>")

    return "\n".join(xml_parts)


def _str_to_bool(value: str) -> bool:
    """Permissive string->bool conversion."""
    match value.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise ValueError(f"Invalid boolean value: {value}")


def _is_optional_type(field_type: type | types.UnionType) -> bool:
    """True if `field_type` is effectively Union[X, None]."""
    origin = typing.get_origin(field_type)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(field_type)
        return len(args) == 2 and type(None) in args
    return False


@typing.overload
def parse_pydantic_from_xml_tags(
    pydantic_model_type: type[T],
    text_containing_xml_tags: str,
    *,
    allow_multiple: typing.Literal[True],
) -> Sequence[T]: ...


@typing.overload
def parse_pydantic_from_xml_tags(
    pydantic_model_type: type[T],
    text_containing_xml_tags: str,
    *,
    allow_multiple: typing.Literal[False] = False,
) -> T: ...


def parse_pydantic_from_xml_tags(
    pydantic_model_type: type[T],
    text_containing_xml_tags: str,
    *,
    allow_multiple: bool = False,
) -> T | Sequence[T]:
    """Parse one or more Pydantic models from XML-style tags in a text string.

    Parses XML-style tags in text and converts them to Pydantic model instances. Can handle both single
    and multiple model parsing based on the allow_multiple flag.

    Args:
        pydantic_model_type: The Pydantic model class to parse into.
        text_containing_xml_tags: String containing XML-style tags to parse.
        allow_multiple: If True, attempts to parse multiple non-overlapping instances.

    Returns:
        If allow_multiple=True, returns a list of parsed model instances. Returns empty list if no valid instances.
        If allow_multiple=False, returns a single parsed model instance.

    Raises:
        RequiredXMLTagNotFoundError: If a required field's XML tag is not found.
        ValueError: If XML tags are malformed or field values are invalid.

    Example:
        >>> class Person(pydantic.BaseModel):
        ...     name: str
        ...     age: int
        >>> xml = "<name>Alice</name><age>30</age>"
        >>> person = parse_pydantic_from_xml_tags(Person, xml)
        >>> person.name
        'Alice'
    """
    normalized_text = text_containing_xml_tags

    def _parse_list_field(
        field_name: str, element_type: type, content: str, is_required: bool
    ) -> tuple[list[typing.Any], int]:
        """Parses XML tags for a list field and returns the parsed values along with the ending position.

        Args:
            field_name: Name of the field/XML tag to parse
            element_type: Type of elements in the list
            content: String containing XML tags to parse
            is_required: Whether this field is required

        Returns:
            tuple containing:
                - list of parsed values of type element_type
                - integer position where parsing ended in original text

        Raises:
            RequiredXMLTagNotFoundError: If field is required but XML tag not found

        Example:
            >>> _parse_list_field("items", str, "<items>a</items><items>b</items>", True)
            (['a', 'b'], 32)
        """
        field_match = try_parse_xml_tag(field_name, content)
        if not field_match.is_found():
            if is_required:
                raise RequiredXMLTagNotFoundError(f"Required list field <{field_name}> not found.")
            return ([], 0)  # optional & not found => empty

        assert field_match.content is not None
        assert field_match.end_pos is not None
        sub_content = field_match.content  # Get sub_content before using it

        # Figure out how to parse each element
        parsed_items: list[typing.Any] = []

        # If `element_type` looks like a model, try scanning for its fields.
        if hasattr(element_type, "model_fields"):
            # This is a Pydantic model
            element_fields = list(element_type.model_fields.keys())

            # Find all instances of each field
            field_matches: dict[str, list[tuple[str, int]]] = {}
            for field in element_fields:
                pos = 0
                matches = []
                while pos < len(sub_content):
                    match = try_parse_xml_tag(field, sub_content[pos:])
                    if not match.is_found():
                        break
                    assert match.content is not None
                    assert match.end_pos is not None
                    matches.append((match.content, pos + match.end_pos))
                    pos += match.end_pos
                field_matches[field] = matches

            # Validate we have same number of matches for each field
            match_counts = {f: len(m) for f, m in field_matches.items()}
            if len(set(match_counts.values())) > 1:
                mismatched = [f"{f}({c})" for f, c in match_counts.items()]
                raise ValueError(f"Mismatched number of fields in <{field_name}>: {', '.join(mismatched)}")

            # Group fields into instances
            num_instances = match_counts[element_fields[0]]
            parsed_items = []
            for i in range(num_instances):
                instance_dict = {
                    field: matches[i][0]  # Take content from (content, pos) tuple
                    for field, matches in field_matches.items()
                }
                parsed_items.append(element_type(**instance_dict))

            return parsed_items, field_match.end_pos

        else:
            # Probably a primitive list => look for <item> sub-tags
            sub_tag_pos = 0
            sub_content = field_match.content

            while True:
                item_match = try_parse_xml_tag("item", sub_content[sub_tag_pos:])
                if not item_match.is_found():
                    break
                assert item_match.end_pos is not None
                sub_tag_pos += item_match.end_pos

                # Attempt cast to element_type
                if item_match.content is not None:
                    parsed_items.append(element_type(item_match.content))
                else:
                    parsed_items.append(None)

            if not parsed_items and is_required:
                raise RequiredXMLTagNotFoundError(f"Required list <{field_name}> not found or empty.")

        # absolute end position in parent text: field_match.end_pos is relative to `content`
        return parsed_items, field_match.end_pos

        # end _parse_list_field

    def _parse_scalar_field(
        field_name: str, field_info: pydantic.fields.FieldInfo, content: str
    ) -> tuple[typing.Any, int]:
        """Parse a non-list field from XML content and return its value and position.

        Parses a field that is not a list type from XML content. Handles type conversion
        for basic types like bool, int, str etc. as well as optional fields.

        Args:
            field_name: Name of the XML tag to parse
            field_info: Pydantic field info containing type annotation
            content: XML content string to parse from

        Returns:
            tuple containing:
                - The parsed value converted to the appropriate type
                - The end position of the tag in the original text

        Raises:
            RequiredXMLTagNotFoundError: If a required tag is missing or empty

        Examples:
            >>> _parse_scalar_field("name", field_info, "<name>Alice</name>")
            ('Alice', 17)
            >>> _parse_scalar_field("age", field_info, "<age>25</age>")
            (25, 11)
            >>> _parse_scalar_field("active", field_info, "<active>true</active>")
            (True, 19)
        """
        if field_info.annotation is None:
            raise ValueError(f"Field {field_name} has no type annotation.")

        is_optional = _is_optional_type(field_info.annotation)
        match_val = try_parse_xml_tag(field_name, content)

        if not match_val.is_found():
            if is_optional:
                return None, 0
            raise RequiredXMLTagNotFoundError(f"Field <{field_name}> not found.")

        end_pos = match_val.end_pos or 0
        raw_val = match_val.content

        if raw_val is None:
            if is_optional:
                return None, end_pos
            raise RequiredXMLTagNotFoundError(f"Field <{field_name}> is required but empty.")

        # Add additional check for whitespace-only content
        if not raw_val.strip() and not is_optional:
            raise RequiredXMLTagNotFoundError(f"Field <{field_name}> is required but empty.")

        # Type conversions:
        if field_info.annotation is bool:
            return _str_to_bool(raw_val), end_pos
        elif is_optional:
            # optional X => figure out X
            # get_args gives e.g. (str, NoneType) or (MyModel, NoneType)
            union_args = typing.get_args(field_info.annotation)
            # find the non-None arg
            actual_type = union_args[0] if union_args[1] is type(None) else union_args[1]
            return actual_type(raw_val), end_pos
        else:
            # e.g. str, int, float, etc.
            return field_info.annotation(raw_val), end_pos

    def parse_single_instance(start_pos: int = 0) -> tuple[T, int]:
        """Parse a single pydantic model instance from the normalized XML text starting at the given position.

        Args:
            start_pos: Starting position in normalized_text to begin parsing from. Defaults to 0.

        Returns:
            tuple containing:
                - The parsed pydantic model instance
                - The ending position after parsing this instance

        Example:
            >>> model, end_pos = parse_single_instance(0)
            >>> print(model)
            MyModel(name='Alice', age=25)
            >>> print(end_pos)
            47
        """
        cursor = 0  # Track position within sub_text
        sub_text = normalized_text[start_pos:]
        parsed_data = {}

        for field_name, field_info in pydantic_model_type.model_fields.items():
            annotation = field_info.annotation
            if annotation is None:
                raise ValueError(f"Field {field_name} has no type annotation.")
            is_list_field = typing.get_origin(annotation) is list

            # Only search in unprocessed portion of text
            local_text = sub_text[cursor:]

            if is_list_field:
                element_type = typing.get_args(annotation)[0]
                is_required = not _is_optional_type(annotation)
                val, local_end = _parse_list_field(field_name, element_type, local_text, is_required)
                parsed_data[field_name] = val
                # Advance cursor if tag was found
                if local_end:
                    cursor += local_end
            else:
                val, local_end = _parse_scalar_field(field_name, field_info, local_text)
                parsed_data[field_name] = val
                if local_end:
                    cursor += local_end

        return pydantic_model_type(**parsed_data), (start_pos + cursor)

    # ------------------
    # Single instance path
    # ------------------
    if not allow_multiple:
        instance, _ = parse_single_instance(0)
        return instance

    # ------------------
    # Multiple instances path
    # ------------------
    instances: list[T] = []
    current_pos = 0

    while current_pos < len(normalized_text):
        try:
            inst, new_pos = parse_single_instance(current_pos)
            if new_pos <= current_pos:
                # means we didn't advance -> likely no more occurrences
                break
            instances.append(inst)
            current_pos = new_pos
        except RequiredXMLTagNotFoundError:
            # No more matching instances
            break

    return instances
