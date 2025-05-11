from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class FilterJSON(GraphNode):
    """
    Filter JSON array based on a key-value condition.
    json, filter, array

    Use cases:
    - Filter arrays of objects
    - Search JSON data
    """

    array: list[dict] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Array of JSON objects to filter"
    )
    key: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Key to filter on"
    )
    value: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Value to match"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.FilterJSON"


class GetJSONPathBool(GraphNode):
    """
    Extract a boolean value from a JSON path
    json, path, extract, boolean
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathBool"


class GetJSONPathDict(GraphNode):
    """
    Extract a dictionary value from a JSON path
    json, path, extract, object
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathDict"


class GetJSONPathFloat(GraphNode):
    """
    Extract a float value from a JSON path
    json, path, extract, number
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: float | GraphNode | tuple[GraphNode, str] = Field(
        default=0.0, description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathFloat"


class GetJSONPathInt(GraphNode):
    """
    Extract an integer value from a JSON path
    json, path, extract, number
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathInt"


class GetJSONPathList(GraphNode):
    """
    Extract a list value from a JSON path
    json, path, extract, array
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: list | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathList"


class GetJSONPathStr(GraphNode):
    """
    Extract a string value from a JSON path
    json, path, extract, string
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON object to extract from"
    )
    path: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Path to the desired value (dot notation)"
    )
    default: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Default value to return if path is not found"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.GetJSONPathStr"


class JSONTemplate(GraphNode):
    """
    Template JSON strings with variable substitution.
    json, template, substitute, variables

    Example:
    template: '{"name": "$user", "age": $age}'
    values: {"user": "John", "age": 30}
    result: '{"name": "John", "age": 30}'

    Use cases:
    - Create dynamic JSON payloads
    - Generate JSON with variable data
    - Build API request templates
    """

    template: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="JSON template string with $variable placeholders"
    )
    values: dict[str, Any] | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="Dictionary of values to substitute into the template"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.JSONTemplate"


class ParseDict(GraphNode):
    """
    Parse a JSON string into a Python dictionary.
    json, parse, decode, dictionary

    Use cases:
    - Convert JSON API responses to Python dictionaries
    - Process JSON configuration files
    - Parse object-like JSON data
    """

    json_string: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="JSON string to parse into a dictionary"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.ParseDict"


class ParseList(GraphNode):
    """
    Parse a JSON string into a Python list.
    json, parse, decode, array, list

    Use cases:
    - Convert JSON array responses to Python lists
    - Process JSON data collections
    - Parse array-like JSON data
    """

    json_string: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="JSON string to parse into a list"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.ParseList"


class StringifyJSON(GraphNode):
    """
    Convert a Python object to a JSON string.
    json, stringify, encode

    Use cases:
    - Prepare data for API requests
    - Save data in JSON format
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="Data to convert to JSON"
    )
    indent: int | GraphNode | tuple[GraphNode, str] = Field(
        default=2, description="Number of spaces for indentation"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.StringifyJSON"


class ValidateJSON(GraphNode):
    """
    Validate JSON data against a schema.
    json, validate, schema

    Use cases:
    - Ensure API payloads match specifications
    - Validate configuration files
    """

    data: Any | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="JSON data to validate"
    )
    schema: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="JSON schema for validation"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.json.ValidateJSON"
