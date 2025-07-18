# src/nodetool/nodes/nodetool/json.py

import json
from typing import Any
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FolderRef


class ParseDict(BaseNode):
    """
    Parse a JSON string into a Python dictionary.
    json, parse, decode, dictionary

    Use cases:
    - Convert JSON API responses to Python dictionaries
    - Process JSON configuration files
    - Parse object-like JSON data
    """

    json_string: str = Field(
        default="", description="JSON string to parse into a dictionary"
    )

    async def process(self, context: ProcessingContext) -> dict:
        result = json.loads(self.json_string)
        if not isinstance(result, dict):
            raise ValueError("JSON string must represent an object/dictionary")
        return result


class ParseList(BaseNode):
    """
    Parse a JSON string into a Python list.
    json, parse, decode, array, list

    Use cases:
    - Convert JSON array responses to Python lists
    - Process JSON data collections
    - Parse array-like JSON data
    """

    json_string: str = Field(default="", description="JSON string to parse into a list")

    async def process(self, context: ProcessingContext) -> list:
        result = json.loads(self.json_string)
        if not isinstance(result, list):
            raise ValueError("JSON string must represent an array/list")
        return result


class StringifyJSON(BaseNode):
    """
    Convert a Python object to a JSON string.
    json, stringify, encode

    Use cases:
    - Prepare data for API requests
    - Save data in JSON format
    """

    data: Any = Field(default={}, description="Data to convert to JSON")
    indent: int = Field(default=2, description="Number of spaces for indentation")

    async def process(self, context: ProcessingContext) -> str:
        return json.dumps(self.data, indent=self.indent)


class BaseGetJSONPath(BaseNode):
    """
    Base class for extracting typed data from a JSON object using a path expression.
    json, path, extract

    Examples for an object {"a": {"b": {"c": 1}}}
    "a.b.c" -> 1
    "a.b" -> {"c": 1}
    "a" -> {"b": {"c": 1}}

    Use cases:
    - Navigate complex JSON structures
    - Extract specific values from nested JSON with type safety
    """

    data: Any = Field(default=None, description="JSON object to extract from")
    path: str = Field(
        default="", description="Path to the desired value (dot notation)"
    )

    @classmethod
    def is_visible(cls):
        return cls is not BaseGetJSONPath

    async def _extract_value(self) -> Any:
        try:
            result = self.data
            for key in self.path.split("."):
                if isinstance(result, dict):
                    result = result.get(key)
                elif isinstance(result, list) and key.isdigit():
                    result = result[int(key)] if int(key) < len(result) else None
                else:
                    return None
            return result
        except (KeyError, IndexError, TypeError):
            return None


class GetJSONPathStr(BaseGetJSONPath):
    """
    Extract a string value from a JSON path
    json, path, extract, string
    """

    default: str = Field(
        default="", description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> str:
        result = await self._extract_value()
        return str(result) if result is not None else self.default


class GetJSONPathInt(BaseGetJSONPath):
    """
    Extract an integer value from a JSON path
    json, path, extract, number
    """

    default: int = Field(
        default=0, description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> int:
        result = await self._extract_value()
        return int(result) if result is not None else self.default


class GetJSONPathFloat(BaseGetJSONPath):
    """
    Extract a float value from a JSON path
    json, path, extract, number
    """

    default: float = Field(
        default=0.0, description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> float:
        result = await self._extract_value()
        return float(result) if result is not None else self.default


class GetJSONPathBool(BaseGetJSONPath):
    """
    Extract a boolean value from a JSON path
    json, path, extract, boolean
    """

    default: bool = Field(
        default=False, description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> bool:
        result = await self._extract_value()
        return bool(result) if result is not None else self.default


class GetJSONPathList(BaseGetJSONPath):
    """
    Extract a list value from a JSON path
    json, path, extract, array
    """

    default: list = Field(
        default=[], description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> list:
        result = await self._extract_value()
        return list(result) if result is not None else self.default


class GetJSONPathDict(BaseGetJSONPath):
    """
    Extract a dictionary value from a JSON path
    json, path, extract, object
    """

    default: dict = Field(
        default={}, description="Default value to return if path is not found"
    )

    async def process(self, context: ProcessingContext) -> dict:
        result = await self._extract_value()
        return dict(result) if result is not None else self.default


class ValidateJSON(BaseNode):
    """
    Validate JSON data against a schema.
    json, validate, schema

    Use cases:
    - Ensure API payloads match specifications
    - Validate configuration files
    """

    data: Any = Field(default=None, description="JSON data to validate")
    json_schema: dict = Field(default={}, description="JSON schema for validation")

    async def process(self, context: ProcessingContext) -> bool:
        from jsonschema import validate, ValidationError

        try:
            validate(instance=self.data, schema=self.json_schema)
            return True
        except ValidationError:
            return False


class FilterJSON(BaseNode):
    """
    Filter JSON array based on a key-value condition.
    json, filter, array

    Use cases:
    - Filter arrays of objects
    - Search JSON data
    """

    array: list[dict] = Field(default=[], description="Array of JSON objects to filter")
    key: str = Field(default="", description="Key to filter on")
    value: Any = Field(default=None, description="Value to match")

    async def process(self, context: ProcessingContext) -> list[dict]:
        return [item for item in self.array if item.get(self.key) == self.value]


class JSONTemplate(BaseNode):
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

    @classmethod
    def get_title(cls):
        return "JSON Template"

    template: str = Field(
        default="", description="JSON template string with $variable placeholders"
    )
    values: dict[str, Any] = Field(
        default={},
        description="Dictionary of values to substitute into the template",
    )

    async def process(self, context: ProcessingContext) -> dict:
        result = self.template
        for key, value in self.values.items():
            placeholder = "$" + key
            result = result.replace(placeholder, str(value))

        try:
            res = json.loads(result)
            assert isinstance(res, dict), f"Resulting JSON must be a dictionary: {res}"
            return res
        except json.JSONDecodeError as e:
            raise ValueError(f"Resulting JSON is invalid: {e} \n {result}")


class LoadJSONAssets(BaseNode):
    """
    Load JSON files from an asset folder.
    load, json, file, import
    """

    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to load the JSON files from."
    )

    @classmethod
    def get_title(cls):
        return "Load JSON Folder"

    @classmethod
    def return_type(cls):
        return {
            "json": dict,
            "name": str,
        }

    async def gen_process(self, context: ProcessingContext):
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets = await context.list_assets(
            parent_id=parent_id, content_type="json"
        )

        for asset in list_assets.assets:
            yield "name", asset.name
            content = await context.download_asset(asset.id)
            try:
                json_data = json.load(content)
                yield "json", json_data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {asset.name}: {e}")
