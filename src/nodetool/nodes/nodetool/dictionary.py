from enum import Enum
import json
from nodetool.config.logging_config import get_logger
from typing import Any, AsyncGenerator, ClassVar, TypedDict
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from pydantic import Field
import os
import csv
import datetime

from nodetool.config.environment import Environment

logger = get_logger(__name__)


class GetValue(BaseNode):
    """
    Retrieve value from dictionary by key, with optional default.

    Looks up the specified key in the dictionary. Returns the associated value
    if the key exists, or the default value if the key is not found.

    Parameters:
    - dictionary (required): Dictionary to query
    - key (required): Key to look up
    - default (optional, default=()): Value returned when key not found

    Returns: Value at key, or default

    Typical usage: Access configuration values, extract fields from parsed JSON,
    or safely read dictionary properties. Follow with type conversion, validation,
    or conditional logic nodes.

    dictionary, get, value, key
    """

    _layout: ClassVar[str] = "small"

    dictionary: dict[(str, Any)] = Field(default={})
    key: str = Field(default="")
    default: Any = Field(default=())

    async def process(self, context: ProcessingContext) -> Any:
        return self.dictionary.get(self.key, self.default)


class Update(BaseNode):
    """
    Merge new key-value pairs into dictionary, modifying in place.

    Adds new keys or updates existing keys with values from new_pairs.
    Original dictionary is modified; keys in new_pairs overwrite existing keys.

    Parameters:
    - dictionary (required): Dictionary to update
    - new_pairs (required): Dictionary of keys and values to merge in

    Returns: Updated dictionary

    Typical usage: Extend configurations, merge user input with defaults, or add
    computed values to data structures. Follow with validation or output nodes.

    dictionary, add, update
    """

    _layout: ClassVar[str] = "small"
    _ = "new_pairs"

    dictionary: dict[(str, Any)] = Field(default={})
    new_pairs: dict[(str, Any)] = Field(default={})

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        self.dictionary.update(self.new_pairs)
        return self.dictionary


class Remove(BaseNode):
    """
    Delete key-value pair from dictionary by key.

    Removes the specified key and its value from the dictionary if the key exists.
    Dictionary is modified in place. No error if key doesn't exist.

    Parameters:
    - dictionary (required): Dictionary to modify
    - key (required): Key to remove

    Returns: Modified dictionary

    Typical usage: Clean sensitive fields, remove temporary data, or exclude
    specific properties before processing. Follow with validation or output nodes.

    dictionary, remove, delete
    """

    _layout: ClassVar[str] = "small"

    dictionary: dict[(str, Any)] = Field(default={})
    key: str = Field(default="")

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        if self.key in self.dictionary:
            del self.dictionary[self.key]
        return self.dictionary


class ParseJSON(BaseNode):
    """
    Parse JSON string into dictionary object.

    Decodes JSON string and validates it represents an object/dictionary.
    Raises error if JSON is invalid or not a dictionary type.

    Parameters:
    - json_string (required): Valid JSON string representing object

    Returns: Python dictionary

    Raises: ValueError if JSON invalid or not dictionary type

    Typical usage: Parse API responses, load JSON config strings, or deserialize
    stored data. For lists use ParseList instead. Follow with GetValue or JSON
    path extraction nodes.

    json, parse, dictionary
    """

    _layout: ClassVar[str] = "small"

    json_string: str = Field(default="")

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        res = json.loads(self.json_string)
        if not isinstance(res, dict):
            raise ValueError("Input JSON is not a dictionary")
        return res


class Zip(BaseNode):
    """
    Create dictionary from parallel lists of keys and values.

    Pairs corresponding elements from keys and values lists into dictionary.
    If lists are different lengths, pairs up to the shorter list length.

    Parameters:
    - keys (required): List of keys
    - values (required): List of corresponding values

    Returns: Dictionary mapping keys to values

    Typical usage: Convert columnar data to key-value structure, create lookups
    from separate lists, or transform paired data. Follow with dictionary operations
    or output nodes.

    dictionary, create, zip
    """

    _layout: ClassVar[str] = "small"

    keys: list[Any] = Field(default=[])
    values: list[Any] = Field(default=[])

    async def process(self, context: ProcessingContext) -> dict[Any, Any]:
        return dict(zip(self.keys, self.values, strict=False))


class Combine(BaseNode):
    """
    Merge two dictionaries with second dictionary values taking precedence.

    Creates new dictionary containing all keys from both inputs. When keys overlap,
    dict_b values override dict_a values. Original dictionaries unchanged.

    Parameters:
    - dict_a (required): First dictionary (lower priority)
    - dict_b (required): Second dictionary (higher priority)

    Returns: New merged dictionary

    Typical usage: Apply overrides to defaults, merge configuration layers, or
    combine partial data structures. Follow with validation or output nodes.

    dictionary, merge, update, +, add, concatenate
    """

    _layout: ClassVar[str] = "small"

    dict_a: dict[(str, Any)] = Field(default={})
    dict_b: dict[(str, Any)] = Field(default={})

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        return {**self.dict_a, **self.dict_b}


class Filter(BaseNode):
    """
    Create new dictionary containing only specified keys from input.

    Selects subset of keys from input dictionary, creating new dictionary with
    only those keys and their values. Keys not in input are silently skipped.

    Parameters:
    - dictionary (required): Dictionary to filter
    - keys (required): List of keys to keep

    Returns: New dictionary with only specified keys

    Typical usage: Extract relevant fields from large objects, remove unwanted
    properties, or project specific columns. Follow with validation or output nodes.

    dictionary, filter, select
    """
    - Implement data access controls
    - Prepare specific data subsets for processing
    """

    dictionary: dict[(str, Any)] = Field(default={})
    keys: list[str] = Field(default=[])

    async def process(self, context: ProcessingContext) -> dict[(str, Any)]:
        return {
            key: self.dictionary[key] for key in self.keys if key in self.dictionary
        }


class ReduceDictionaries(BaseNode):
    """
    Reduces a list of dictionaries into one dictionary based on a specified key field.
    dictionary, reduce, aggregate

    Use cases:
    - Aggregate data by a specific field
    - Create summary dictionaries from list of records
    - Combine multiple data points into a single structure
    """

    class ConflictResolution(str, Enum):
        FIRST = "first"
        LAST = "last"
        ERROR = "error"

    dictionaries: list[dict[str, Any]] = Field(
        default=[],
        description="List of dictionaries to be reduced",
    )
    key_field: str = Field(
        default="",
        description="The field to use as the key in the resulting dictionary",
    )
    value_field: str = Field(
        default="",
        description="Optional field to use as the value. If not specified, the entire dictionary (minus the key field) will be used as the value.",
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.FIRST,
        description="How to handle conflicts when the same key appears multiple times",
    )

    async def process(self, context: ProcessingContext) -> dict[Any, Any]:
        result = {}
        for d in self.dictionaries:
            if self.key_field not in d:
                raise ValueError(
                    f"Key field '{self.key_field}' not found in dictionary"
                )

            key = d[self.key_field]

            if self.value_field:
                if self.value_field not in d:
                    raise ValueError(
                        f"Value field '{self.value_field}' not found in dictionary"
                    )
                value = d[self.value_field]
            else:
                value = {k: v for k, v in d.items() if k != self.key_field}

            if key in result:
                if self.conflict_resolution == self.ConflictResolution.FIRST:
                    continue
                elif self.conflict_resolution == self.ConflictResolution.LAST:
                    result[key] = value
                else:  # ConflictResolution.ERROR
                    raise ValueError(f"Duplicate key found: {key}")
            else:
                result[key] = value

        return result


class MakeDictionary(BaseNode):
    """
    Creates a simple dictionary with up to three key-value pairs.
    dictionary, create, simple

    Use cases:
    - Create configuration entries
    - Initialize simple data structures
    - Build basic key-value mappings
    """

    _layout: ClassVar[str] = "small"
    _is_dynamic: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        logger.debug("Dynamic properties: %s", self._dynamic_properties)
        return self._dynamic_properties.copy()


class ArgMax(BaseNode):
    """
    Returns the label associated with the highest value in a dictionary.
    dictionary, maximum, label, argmax

    Use cases:
    - Get the most likely class from classification probabilities
    - Find the category with highest score
    - Identify the winner in a voting/ranking system
    """

    _layout: ClassVar[str] = "small"

    scores: dict[str, float] = Field(
        default={},
        description="Dictionary mapping labels to their corresponding scores/values",
    )

    async def process(self, context: ProcessingContext) -> str:
        if not self.scores:
            raise ValueError("Input dictionary cannot be empty")
        return max(self.scores.items(), key=lambda x: x[1])[0]


class LoadCSVFile(BaseNode):
    """
    Read a CSV file from disk.
    files, csv, read, input, load, file
    """

    path: str = Field(default="", description="Path to the CSV file to read")

    async def process(self, context: ProcessingContext) -> list[dict]:
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.path:
            raise ValueError("path cannot be empty")
        expanded_path = os.path.expanduser(self.path)
        with open(expanded_path, "r") as f:
            reader = csv.DictReader(f)
            return [row for row in reader]


class SaveCSVFile(BaseNode):
    """
    Write a list of dictionaries to a CSV file.
    files, csv, write, output, save, file

    The filename can include time and date variables:
    %Y - Year, %m - Month, %d - Day
    %H - Hour, %M - Minute, %S - Second
    """

    data: list[dict] = Field(
        default=[], description="list of dictionaries to write to CSV"
    )
    folder: str = Field(default="", description="Folder where the file will be saved")
    filename: str = Field(
        default="",
        description="Name of the CSV file to save. Supports strftime format codes.",
    )

    async def process(self, context: ProcessingContext):
        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.data:
            raise ValueError("'data' field cannot be empty")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        if not self.filename:
            raise ValueError("filename cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        filename = datetime.datetime.now().strftime(self.filename)
        expanded_path = os.path.join(expanded_folder, filename)
        with open(expanded_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)


class FilterDictByValue(BaseNode):
    """
    Filters a stream of dictionaries based on their values using various criteria.
    filter, dictionary, values, stream

    Use cases:
    - Filter dictionaries by value content
    - Filter dictionaries by value type
    - Filter dictionaries by value patterns
    """

    class FilterType(str, Enum):
        CONTAINS = "contains"
        STARTS_WITH = "starts_with"
        ENDS_WITH = "ends_with"
        EQUALS = "equals"
        TYPE_IS = "type_is"
        LENGTH_GREATER = "length_greater"
        LENGTH_LESS = "length_less"
        EXACT_LENGTH = "exact_length"

    value: dict = Field(default={}, description="Input dictionary stream")
    key: str = Field(default="", description="The dictionary key to check")
    filter_type: FilterType = Field(
        default=FilterType.CONTAINS, description="The type of filter to apply"
    )
    criteria: str = Field(
        default="",
        description="The filtering criteria (text to match, type name, or length as string)",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: dict

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        current_key = self.key
        current_filter_type = self.filter_type
        current_criteria = self.criteria

        async for handle, item in self.iter_any_input():
            if handle == "key":
                current_key = item
                continue
            elif handle == "filter_type":
                current_filter_type = item
                continue
            elif handle == "criteria":
                current_criteria = item
                continue
            elif handle == "value":
                d = item
                if not isinstance(d, dict):
                    continue
                
                if current_key not in d:
                    continue

                val = d[current_key]
                value_str = str(val)
                length_criteria = 0
                
                if current_filter_type in [
                    self.FilterType.LENGTH_GREATER,
                    self.FilterType.LENGTH_LESS,
                    self.FilterType.EXACT_LENGTH,
                ]:
                    try:
                        length_criteria = int(current_criteria)
                    except ValueError:
                        continue

                matched = False
                if current_filter_type == self.FilterType.CONTAINS:
                    if current_criteria in value_str:
                        matched = True
                elif current_filter_type == self.FilterType.STARTS_WITH:
                    if value_str.startswith(current_criteria):
                        matched = True
                elif current_filter_type == self.FilterType.ENDS_WITH:
                    if value_str.endswith(current_criteria):
                        matched = True
                elif current_filter_type == self.FilterType.EQUALS:
                    if value_str == current_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.TYPE_IS:
                    if type(val).__name__ == current_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.LENGTH_GREATER:
                    if hasattr(val, "__len__") and len(val) > length_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.LENGTH_LESS:
                    if hasattr(val, "__len__") and len(val) < length_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.EXACT_LENGTH:
                    if hasattr(val, "__len__") and len(val) == length_criteria:
                        matched = True
                
                if matched:
                    yield {"output": d}


class FilterDictByRange(BaseNode):
    """
    Filters a stream of dictionaries based on a numeric range for a specified key.
    filter, dictionary, range, between, stream

    Use cases:
    - Filter records based on numeric ranges (e.g., price range, age range)
    - Find entries with values within specified bounds
    """

    value: dict = Field(default={}, description="Input dictionary stream")
    key: str = Field(
        default="", description="The dictionary key to check for the range"
    )
    min_value: float = Field(
        default=0, description="The minimum value (inclusive) of the range"
    )
    max_value: float = Field(
        default=0, description="The maximum value (inclusive) of the range"
    )
    inclusive: bool = Field(
        default=True,
        description="If True, includes the min and max values in the results",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: dict

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        current_key = self.key
        current_min = self.min_value
        current_max = self.max_value
        current_inclusive = self.inclusive

        async for handle, item in self.iter_any_input():
            if handle == "key":
                current_key = item
                continue
            elif handle == "min_value":
                current_min = item
                continue
            elif handle == "max_value":
                current_max = item
                continue
            elif handle == "inclusive":
                current_inclusive = item
                continue
            elif handle == "value":
                d = item
                if not isinstance(d, dict):
                    continue
                
                if current_key not in d:
                    continue

                val = d[current_key]
                if not isinstance(val, (int, float)):
                    continue

                matched = False
                if current_inclusive:
                    if current_min <= val <= current_max:
                        matched = True
                else:
                    if current_min < val < current_max:
                        matched = True
                
                if matched:
                    yield {"output": d}


class FilterDictByNumber(BaseNode):
    """
    Filters a stream of dictionaries based on numeric values for a specified key.
    filter, dictionary, numbers, numeric, stream

    Use cases:
    - Filter dictionaries by numeric comparisons (greater than, less than, equal to)
    - Filter records with even/odd numeric values
    """

    class FilterDictNumberType(str, Enum):
        GREATER_THAN = "greater_than"
        LESS_THAN = "less_than"
        EQUAL_TO = "equal_to"
        EVEN = "even"
        ODD = "odd"
        POSITIVE = "positive"
        NEGATIVE = "negative"

    value: dict = Field(default={}, description="Input dictionary stream")
    key: str = Field(default="", description="The dictionary key to check")
    filter_type: FilterDictNumberType = Field(default=FilterDictNumberType.GREATER_THAN)
    compare_value: float = Field(default=0, description="Comparison value")

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: dict

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        current_key = self.key
        current_filter_type = self.filter_type
        current_compare_value = self.compare_value

        async for handle, item in self.iter_any_input():
            if handle == "key":
                current_key = item
                continue
            elif handle == "filter_type":
                current_filter_type = item
                continue
            elif handle == "compare_value":
                current_compare_value = item
                continue
            elif handle == "value":
                d = item
                if not isinstance(d, dict):
                    continue
                
                if current_key not in d:
                    continue
                
                num = d[current_key]
                if not isinstance(num, (int, float)):
                    continue

                matched = False
                if current_filter_type == self.FilterDictNumberType.GREATER_THAN:
                    if current_compare_value is not None and num > current_compare_value:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.LESS_THAN:
                    if current_compare_value is not None and num < current_compare_value:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.EQUAL_TO:
                    if current_compare_value is not None and num == current_compare_value:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.EVEN:
                    if isinstance(num, int) and num % 2 == 0:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.ODD:
                    if isinstance(num, int) and num % 2 != 0:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.POSITIVE:
                    if num > 0:
                        matched = True
                elif current_filter_type == self.FilterDictNumberType.NEGATIVE:
                    if num < 0:
                        matched = True
                
                if matched:
                    yield {"output": d}


class FilterDictRegex(BaseNode):
    """
    Filters a stream of dictionaries using regular expressions on specified keys.
    filter, regex, dictionary, pattern, stream

    Use cases:
    - Filter dictionaries with values matching complex patterns
    - Search for dictionaries containing emails, dates, or specific formats
    """

    value: dict = Field(default={}, description="Input dictionary stream")
    key: str = Field(default="", description="The dictionary key to check")
    pattern: str = Field(default="", description="The regex pattern")
    full_match: bool = Field(default=False, description="Full match or partial")

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: dict

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        import re
        
        current_key = self.key
        current_pattern = self.pattern
        current_full_match = self.full_match
        regex = None

        try:
             regex = re.compile(current_pattern)
        except re.error:
             pass

        async for handle, item in self.iter_any_input():
            if handle == "key":
                current_key = item
                continue
            elif handle == "pattern":
                current_pattern = item
                try:
                    regex = re.compile(current_pattern)
                except re.error:
                    regex = None
                continue
            elif handle == "full_match":
                current_full_match = item
                continue
            elif handle == "value":
                d = item
                if not isinstance(d, dict):
                    continue
                
                if current_key not in d:
                    continue
                
                if regex is None:
                    try:
                        regex = re.compile(current_pattern)
                    except re.error:
                        continue
                
                val = str(d[current_key])
                matched = False
                if current_full_match:
                    if regex.fullmatch(val):
                        matched = True
                else:
                    if regex.search(val):
                        matched = True
                
                if matched:
                    yield {"output": d}


class FilterDictByQuery(BaseNode):
    """
    Filter a stream of dictionary objects based on a pandas query condition.
    filter, query, condition, dictionary, stream

    Basic Operators:
    - Comparison: >, <, >=, <=, ==, !=
    - Logical: and, or, not
    - Membership: in, not in

    Use cases:
    - Filter dictionary objects based on complex criteria
    - Extract subset of data meeting specific conditions
    """

    value: dict = Field(default={}, description="Input dictionary stream")
    condition: str = Field(
        default="",
        description="The filtering condition using pandas query syntax.",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: dict

    async def gen_process(self, context: ProcessingContext) -> AsyncGenerator[OutputType, None]:
        import pandas as pd
        
        current_condition = self.condition

        async for handle, item in self.iter_any_input():
            if handle == "condition":
                current_condition = item
                continue
            elif handle == "value":
                d = item
                if not isinstance(d, dict):
                    continue
                
                if current_condition:
                    try:
                        df = pd.DataFrame([d])
                        filtered_df = df.query(current_condition)
                        if not filtered_df.empty:
                            yield {"output": d}
                    except Exception:
                        pass # Query failure or other error, skip
                else:
                    yield {"output": d}
