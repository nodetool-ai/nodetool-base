# src/nodetool/nodes/nodetool/json.py

import json
from typing import Any, AsyncGenerator, TypedDict
from pydantic import Field
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FolderRef


class ParseDict(BaseNode):
    """
    Parse JSON string into Python dictionary object.

    Decodes a JSON string representing an object/dictionary. Raises ValueError
    if the JSON string represents a different type (array, string, etc.).

    Parameters:
    - json_string (required): Valid JSON string representing an object

    Returns: Python dictionary

    Raises: ValueError if JSON is invalid or not an object type

    Typical usage: Parse JSON responses from APIs, decode JSON configuration strings,
    or convert JSON object strings to dictionaries. Precede with HTTP GET or text
    extraction nodes. Follow with dictionary access, JSON path, or validation nodes.

    json, parse, decode, dictionary
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
    Parse JSON string into Python list object.

    Decodes a JSON string representing an array/list. Raises ValueError if the
    JSON string represents a different type (object, string, etc.).

    Parameters:
    - json_string (required): Valid JSON string representing an array

    Returns: Python list

    Raises: ValueError if JSON is invalid or not an array type

    Typical usage: Parse JSON array responses from APIs, decode JSON list strings,
    or convert JSON arrays to lists. Follow with list iteration (ForEach), filtering,
    or transformation nodes.

    json, parse, decode, array, list
    """

    json_string: str = Field(default="", description="JSON string to parse into a list")

    async def process(self, context: ProcessingContext) -> list:
        result = json.loads(self.json_string)
        if not isinstance(result, list):
            raise ValueError("JSON string must represent an array/list")
        return result


class StringifyJSON(BaseNode):
    """
    Convert Python object to formatted JSON string.

    Serializes any JSON-compatible Python object (dict, list, primitives) to a
    JSON string with configurable indentation for readability.

    Parameters:
    - data (optional, default={}): Python object to serialize (must be JSON-compatible)
    - indent (optional, default=2): Number of spaces for indentation (0 for compact)

    Returns: JSON-formatted string

    Typical usage: Prepare data for API POST requests, save configuration as JSON text,
    or format data for display. Precede with dictionary construction or data transformation
    nodes. Follow with HTTP POST, file save, or text output nodes.

    json, stringify, encode
    """

    data: Any = Field(default={}, description="Data to convert to JSON")
    indent: int = Field(default=2, description="Number of spaces for indentation")

    async def process(self, context: ProcessingContext) -> str:
        return json.dumps(self.data, indent=self.indent)


class BaseGetJSONPath(BaseNode):
    """
    Base class for extracting typed values from JSON using dot notation paths.

    Navigates nested JSON structures using dot-separated keys. Supports dictionary
    navigation and array indexing by numeric strings. Not directly visible in UI;
    extended by type-specific subclasses.

    Path examples for {"a": {"b": {"c": 1}}}:
    - "a.b.c" extracts 1
    - "a.b" extracts {"c": 1}
    - "a" extracts {"b": {"c": 1}}

    Array example for {"list": [10, 20, 30]}:
    - "list.0" extracts 10

    json, path, extract
    """

    data: Any = Field(default={}, description="JSON object to extract from")
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
    Extract string value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to string. Returns
    default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the value (e.g., "user.name")
    - default (optional, default=""): Value returned if path not found

    Returns: String value at path, or default

    Typical usage: Extract text fields from API responses, get configuration values,
    or navigate nested JSON structures. Follow with text processing or string manipulation.

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
    Extract integer value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to integer. Returns
    default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the value
    - default (optional, default=0): Value returned if path not found

    Returns: Integer value at path, or default

    Typical usage: Extract numeric IDs, counts, or integer fields from API responses.
    Follow with arithmetic, comparison, or conditional logic nodes.

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
    Extract float value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to floating point number.
    Returns default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the value
    - default (optional, default=0.0): Value returned if path not found

    Returns: Float value at path, or default

    Typical usage: Extract prices, percentages, measurements, or decimal fields from
    API responses. Follow with math operations or numeric formatting nodes.

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
    Extract boolean value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to boolean. Returns
    default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the value
    - default (optional, default=False): Value returned if path not found

    Returns: Boolean value at path, or default

    Typical usage: Extract flags, status indicators, or boolean fields from API
    responses. Follow with conditional logic (If node) or boolean operations.

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
    Extract list/array value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to list. Returns
    default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the array value
    - default (optional, default=[]): Value returned if path not found

    Returns: List value at path, or default

    Typical usage: Extract arrays from nested JSON structures, get lists of items
    from API responses. Follow with ForEach for iteration, filtering, or list operations.

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
    Extract dictionary/object value from JSON object using dot notation path.

    Navigates to the specified path and converts the value to dictionary. Returns
    default if path not found or value is None.

    Parameters:
    - data (required): JSON object (dict or list) to extract from
    - path (required): Dot-separated path to the object value
    - default (optional, default={}): Value returned if path not found

    Returns: Dictionary value at path, or default

    Typical usage: Extract nested objects from JSON structures, get configuration
    sections, or isolate sub-structures. Follow with further JSON path extraction
    or dictionary manipulation nodes.

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
    Validate JSON data against JSON Schema specification.

    Checks if the provided data conforms to the given JSON Schema. Returns true
    for valid data, false for invalid. Uses jsonschema library for validation.

    Parameters:
    - data (required): JSON data to validate (dict, list, or primitive)
    - json_schema (required): JSON Schema definition as dictionary

    Returns: Boolean - true if valid, false if invalid

    Typical usage: Validate API responses before processing, verify configuration
    format, or ensure data quality. Follow with conditional logic (If node) to handle
    validation results.

    json, validate, schema
    """

    data: Any = Field(default={}, description="JSON data to validate")
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
    Filter array of JSON objects by matching key-value condition.

    Returns only objects where the specified key exactly matches the given value.
    Objects without the key are excluded.

    Parameters:
    - array (required): List of dictionaries to filter
    - key (required): Dictionary key to check
    - value (required): Value to match against

    Returns: Filtered list of dictionaries

    Typical usage: Filter API results, select specific items from data collections,
    or search JSON arrays. Precede with JSON parsing or API response nodes. Follow
    with iteration, transformation, or further filtering.

    json, filter, array
    """

    array: list[dict] = Field(default=[], description="Array of JSON objects to filter")
    key: str = Field(default="", description="Key to filter on")
    value: Any = Field(default={}, description="Value to match")

    async def process(self, context: ProcessingContext) -> list[dict]:
        return [item for item in self.array if item.get(self.key) == self.value]


class JSONTemplate(BaseNode):
    """
    Generate JSON dictionary from template with variable substitution.

    Replaces $variable placeholders in the template string with provided values,
    then parses the result as JSON. Raises error if result is not valid JSON or
    not a dictionary.

    Template example:
    - template: '{"name": "$user", "age": $age}'
    - values: {"user": "John", "age": 30}
    - result: {"name": "John", "age": 30}

    Parameters:
    - template (required): JSON template string with $variable placeholders
    - values (optional, default={}): Dictionary mapping variable names to values

    Returns: Dictionary parsed from substituted template

    Raises: ValueError if resulting JSON is invalid or not a dictionary

    Typical usage: Create dynamic API payloads, generate parameterized JSON, or build
    structured data from variables. Follow with JSON POST or validation nodes.

    json, template, substitute, variables
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
    Load all JSON files from an asset folder and emit each as a stream item.

    Scans the specified asset folder for JSON files, parses each file, and emits
    a stream of JSON objects with their filenames. Streaming output allows downstream
    nodes to process each file individually.

    Parameters:
    - folder (required): FolderRef pointing to the asset folder containing JSON files

    Yields: Dictionary with "json" (parsed JSON data) and "name" (filename string)
    for each JSON file

    Raises: ValueError if folder is empty or JSON files contain invalid JSON

    Typical usage: Batch process JSON configuration files, load multiple datasets,
    or iterate through JSON documents. Follow with ForEach or Collect nodes to
    process or aggregate the stream.

    load, json, file, import
    """

    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to load the JSON files from."
    )

    @classmethod
    def get_title(cls):
        return "Load JSON Folder"

    class OutputType(TypedDict):
        json: dict
        name: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets, _ = await context.list_assets(
            parent_id=parent_id, content_type="json"
        )

        for asset in list_assets:
            content = await context.download_asset(asset.id)
            try:
                json_data = json.load(content)
                yield {"json": json_data, "name": asset.name}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {asset.name}: {e}")
