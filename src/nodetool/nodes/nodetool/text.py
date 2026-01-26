from datetime import datetime
from io import BytesIO
import json
import os
import string
import unicodedata

from typing import Any, AsyncGenerator, ClassVar, TypedDict
from nodetool.workflows.io import NodeInputs, NodeOutputs
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ASRModel, FolderRef, AudioRef, Provider, EmbeddingModel, NPArray
from nodetool.workflows.base_node import BaseNode
import re
from jsonpath_ng import parse
from nodetool.metadata.types import TextRef
from enum import Enum
import html2text
from nodetool.io.uri_utils import create_file_uri
from nodetool.workflows.types import SaveUpdate
import aiofiles


class ToString(BaseNode):
    """
    Converts any input value to its string representation.
    text, string, convert, repr, str, cast
    """

    class Mode(str, Enum):
        STR = "str"
        REPR = "repr"

    value: Any = Field(default=(), title="Value")
    mode: Mode = Field(
        default=Mode.STR,
        description="Conversion mode: use `str(value)` or `repr(value)`.",
    )

    @classmethod
    def get_title(cls):
        return "To String"

    async def process(self, context: ProcessingContext) -> str:
        if self.mode == self.Mode.REPR:
            return repr(self.value)
        return str(self.value)


class AutomaticSpeechRecognition(BaseNode):
    """
    Transcribe audio to text using automatic speech recognition models.
    audio, speech, recognition, transcription, ASR, whisper
    """

    class OutputType(TypedDict):
        text: str

    _expose_as_tool: ClassVar[bool] = True

    model: ASRModel = Field(
        default=ASRModel(provider=Provider.FalAI, id="openai/whisper-large-v3")
    )
    audio: AudioRef = Field(default=AudioRef(), description="The audio to transcribe")

    async def process(self, context: ProcessingContext) -> OutputType:
        provider = await context.get_provider(self.model.provider)
        audio_bytes = await context.asset_to_bytes(self.audio)
        text = await provider.automatic_speech_recognition(
            audio_bytes,
            model=self.model.id,
            context=context,
        )
        return {"text": text}


class Concat(BaseNode):
    """
    Concatenates two text inputs into a single output.
    text, combine, add, +, concatenate, merge, join, append
    """

    a: str = Field(default="")
    b: str = Field(default="")

    @classmethod
    def get_title(cls):
        return "Concatenate Text"

    async def process(self, context: ProcessingContext) -> str:
        return self.a + self.b


class Join(BaseNode):
    """
    Joins a list of strings into a single string using a specified separator.
    text, join, combine, +, add, concatenate, merge
    """

    strings: list[Any] = Field(default=[])
    separator: str = Field(default="")

    @classmethod
    def get_title(cls):
        return "Join"

    async def process(self, context: ProcessingContext) -> str:
        if len(self.strings) == 0:
            return ""
        return self.separator.join([str(s) for s in self.strings])


class Collect(BaseNode):
    """
    Collects a stream of text inputs into a single concatenated string.
    text, collect, list, stream, aggregate
    """

    input_item: str = Field(default="")
    separator: str = Field(default="")

    @classmethod
    def get_title(cls):
        return "Collect"

    class OutputType(TypedDict):
        output: str

    async def run(
        self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs
    ) -> None:
        collected_items = []
        async for input_item in inputs.stream("input_item"):
            collected_items.append(input_item)
        await outputs.emit("output", self.separator.join(collected_items))


class FormatText(BaseNode):
    """
    Replaces placeholders in a string with dynamic inputs using Jinja2 templating.
    text, template, formatting

    This node is dynamic and can be used to format text with dynamic properties.

    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}"
    """

    _is_dynamic: ClassVar[bool] = True

    template: str = Field(
        default="",
        description="""
    Example: Hello, {{ name }} or {{ title|truncate(20) }}

    Available filters: truncate, upper, lower, title, trim, replace, default, first, last, length, sort, join
""",
    )

    @classmethod
    def get_title(cls):
        return "Format Text"

    async def process(self, context: ProcessingContext) -> str:
        from jinja2 import Environment, BaseLoader

        try:
            # Create Jinja2 environment
            env = Environment(loader=BaseLoader())

            # Convert template variables to lowercase
            template_str = self.template
            for var in re.findall(r"{{\s*([^|}]+)", template_str):
                template_str = template_str.replace(var, var.lower())

            template = env.from_string(template_str)

            # Convert all dynamic property keys to lowercase for consistency
            lowercase_properties = {
                k.lower(): v for k, v in self._dynamic_properties.items()
            }

            # Render template with properties
            return template.render(**lowercase_properties)
        except Exception as e:
            raise ValueError(f"Template error: {str(e)}")


class Template(BaseNode):
    """
    Uses Jinja2 templating to format strings with variables and filters. This node is dynamic and can be used to format text with dynamic inputs.
    text, template, formatting, format, combine, concatenate, +, add, variable, replace, filter

    Use cases:
    - Generating personalized messages with dynamic content
    - Creating parameterized queries or commands
    - Formatting and filtering text output based on variable inputs

    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}"

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
    """

    string: str = Field(
        default="",
        description="""
    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}"

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
""",
    )
    values: Any = Field(
        default={},
        description="""
        The values to replace in the string.
        - If a string, it will be used as the format string.
        - If a list, it will be used as the format arguments.
        - If a dictionary, it will be used as the template variables.
        - If an object, it will be converted to a dictionary using the object's __dict__ method.
        """,
    )
    _is_dynamic: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> str:
        import re

        template_values = {}
        if isinstance(self.values, dict):
            template_values.update(self.values)
        template_values.update(self._dynamic_properties)

        result = self.string
        for key, value in template_values.items():
            # Replace patterns like {{ key }} with the provided value
            pattern = r"{{\s*" + re.escape(str(key)) + r"\s*}}"
            result = re.sub(pattern, str(value), result)
        return result


class Replace(BaseNode):
    """
    Replaces a substring in a text with another substring.
    text, replace, substitute

    Use cases:
    - Correcting or updating specific text patterns
    - Sanitizing or normalizing text data
    - Implementing simple text transformations
    """

    text: str = Field(title="Text", default="")
    old: str = Field(title="Old", default="")
    new: str = Field(title="New", default="")

    @classmethod
    def get_title(cls):
        return "Replace Text"

    async def process(self, context: ProcessingContext) -> str:
        return self.text.replace(self.old, self.new)


class SaveTextFile(BaseNode):
    """
    Saves input text to a file in the assets folder.
    text, save, file
    """

    text: str = Field(title="Text", default="")
    folder: str = Field(default="", description="Path to the output folder.")
    name: str = Field(
        title="Name",
        default="%Y-%m-%d-%H-%M-%S.txt",
        description="""
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    async def process(self, context: ProcessingContext) -> TextRef:
        filename = datetime.now().strftime(self.name)
        file_data = self.text.encode("utf-8")
        if not self.folder:
            raise ValueError("folder cannot be empty")
        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")
        filename = datetime.now().strftime(self.name)
        expanded_path = os.path.join(expanded_folder, filename)

        async with aiofiles.open(expanded_path, "wb") as f:
            await f.write(file_data)

        result = TextRef(uri=create_file_uri(expanded_path), data=file_data)

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(node_id=self.id, name=filename, value=result, output_type="text")
        )

        return result


class SaveText(BaseNode):
    """
    Saves input text to a file in the assets folder.
    text, save, file

    Use cases:
    - Persisting processed text results
    - Creating text files for downstream nodes or external use
    - Archiving text data within the workflow
    """

    text: str = Field(title="Text", default="")
    folder: FolderRef = Field(
        default=FolderRef(), description="Name of the output folder."
    )
    name: str = Field(
        title="Name",
        default="%Y-%m-%d-%H-%M-%S.txt",
        description="""
        Name of the output file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
        """,
    )

    def required_inputs(self):
        return ["text"]

    @classmethod
    def get_title(cls):
        return "Save Text"

    async def process(self, context: ProcessingContext) -> TextRef:
        filename = datetime.now().strftime(self.name)
        file = BytesIO(self.text.encode("utf-8"))
        parent_id = self.folder.asset_id if self.folder.is_set() else None
        asset = await context.create_asset(filename, "text/plain", file, parent_id)
        asset_uri = await context.get_asset_url(asset.id)
        result = TextRef(uri=asset_uri or "", asset_id=asset.id)

        # Emit SaveUpdate event
        context.post_message(
            SaveUpdate(node_id=self.id, name=filename, value=result, output_type="text")
        )

        return result


class LoadTextFolder(BaseNode):
    """
    Load all text files from a folder, optionally including subfolders.
    text, load, folder, files
    """

    folder: str = Field(default="", description="Folder to scan for text files")
    include_subdirectories: bool = Field(
        default=False, description="Include text files in subfolders"
    )
    extensions: list[str] = Field(
        default=[".txt", ".csv", ".json", ".xml", ".md", ".html", ".pdf"],
        description="Text file extensions to include",
    )
    pattern: str = Field(default="", description="Pattern to match text files")

    @classmethod
    def get_title(cls):
        return "Load Text Folder"

    class OutputType(TypedDict):
        text: str
        path: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        from nodetool.config.environment import Environment
        import fnmatch

        if Environment.is_production():
            raise ValueError("This node is not available in production")
        if not self.folder:
            raise ValueError("folder cannot be empty")

        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.isdir(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")

        allowed_exts = {ext.lower() for ext in self.extensions}

        def iter_files(base_folder: str):
            if self.include_subdirectories:
                for root, _, files in os.walk(base_folder):
                    for f in files:
                        yield os.path.join(root, f)
            else:
                for f in os.listdir(base_folder):
                    yield os.path.join(base_folder, f)

        for path in iter_files(expanded_folder):
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in allowed_exts:
                continue

            if self.pattern and not fnmatch.fnmatch(path, self.pattern):
                continue

            async with aiofiles.open(path, "r") as f:
                text_data = await f.read()

            yield {"path": path, "text": text_data}

class Split(BaseNode):
    """
    Separates text into a list of strings based on a specified delimiter.
    text, split, tokenize
    """

    text: str = Field(title="Text", default="")
    delimiter: str = ","

    @classmethod
    def get_title(cls):
        return "Split Text"

    async def process(self, context: ProcessingContext) -> list[str]:
        return self.text.split(self.delimiter)


class Extract(BaseNode):
    """
    Extracts a substring from input text.
    text, extract, substring
    """

    text: str = Field(title="Text", default="")
    start: int = Field(title="Start", default=0)
    end: int = Field(title="End", default=0)

    @classmethod
    def get_title(cls):
        return "Extract Text"

    async def process(self, context: ProcessingContext) -> str:
        return self.text[self.start : self.end]


class Chunk(BaseNode):
    """
    Splits text into chunks of specified word length.
    text, chunk, split
    """

    text: str = Field(title="Text", default="")
    length: int = Field(title="Length", default=100, le=1000, ge=1)
    overlap: int = Field(title="Overlap", default=0)
    separator: str = Field(title="Separator", default=" ")

    @classmethod
    def get_title(cls):
        return "Split Text into Chunks"

    async def process(self, context: ProcessingContext) -> list[str]:
        text = self.text.split(sep=self.separator)
        chunks = [
            text[i : i + self.length]
            for i in range(0, len(text), self.length - self.overlap)
        ]
        return [" ".join(chunk) for chunk in chunks]


class ExtractRegex(BaseNode):
    """
    Extracts substrings matching regex groups from text.
    text, regex, extract

    Use cases:
    - Extracting structured data (e.g., dates, emails) from unstructured text
    - Parsing specific patterns in log files or documents
    - Isolating relevant information from complex text formats
    """

    text: str = Field(title="Text", default="")
    regex: str = Field(title="Regex", default="")
    dotall: bool = Field(title="Dotall", default=False)
    ignorecase: bool = Field(title="Ignorecase", default=False)
    multiline: bool = Field(title="Multiline", default=False)

    @classmethod
    def get_title(cls):
        return "Extract Regex Groups"

    async def process(self, context: ProcessingContext) -> list[str]:
        options = 0
        if self.dotall:
            options |= re.DOTALL
        if self.ignorecase:
            options |= re.IGNORECASE
        if self.multiline:
            options |= re.MULTILINE
        match = re.search(self.regex, self.text, options)
        if match is None:
            return []
        return list(match.groups())


class FindAllRegex(BaseNode):
    """
    Finds all regex matches in text as separate substrings.
    text, regex, find
    """

    text: str = Field(title="Text", default="")
    regex: str = Field(title="Regex", default="")
    dotall: bool = Field(title="Dotall", default=False)
    ignorecase: bool = Field(title="Ignorecase", default=False)
    multiline: bool = Field(title="Multiline", default=False)

    @classmethod
    def get_title(cls):
        return "Find All Regex Matches"

    async def process(self, context: ProcessingContext) -> list[str]:
        options = 0
        if self.dotall:
            options |= re.DOTALL
        if self.ignorecase:
            options |= re.IGNORECASE
        if self.multiline:
            options |= re.MULTILINE
        matches = re.findall(self.regex, self.text, options)
        return list(matches)


class ParseJSON(BaseNode):
    """
    Parses a JSON string into a Python object.
    json, parse, convert
    """

    text: str = Field(title="JSON string", default="")

    @classmethod
    def get_title(cls):
        return "Parse JSON String"

    async def process(self, context: ProcessingContext) -> Any:
        return json.loads(self.text)


class ExtractJSON(BaseNode):
    """
    Extracts data from JSON using JSONPath expressions.
    json, extract, jsonpath
    """

    text: str = Field(title="JSON Text", default="")
    json_path: str = Field(title="JSONPath Expression", default="$.*")
    find_all: bool = Field(title="Find All", default=False)

    @classmethod
    def get_title(cls):
        return "Extract JSON"

    async def process(self, context: ProcessingContext) -> Any:
        parsed_json = json.loads(self.text)
        jsonpath_expr = parse(self.json_path)
        if self.find_all:
            return [match.value for match in jsonpath_expr.find(parsed_json)]
        else:
            return jsonpath_expr.find(parsed_json)[0].value


class RegexMatch(BaseNode):
    """
    Find all matches of a regex pattern in text.
    regex, search, pattern, match

    Use cases:
    - Extract specific patterns from text
    - Validate text against patterns
    - Find all occurrences of a pattern
    """

    text: str = Field(default="", description="Text to search in")
    pattern: str = Field(default="", description="Regular expression pattern")
    group: int = Field(
        default=0, description="Capture group to extract (0 for full match)"
    )

    @classmethod
    def get_title(cls):
        return "Find Regex Matches"

    async def process(self, context: ProcessingContext) -> list[str]:
        if self.group is None:
            return re.findall(self.pattern, self.text)
        matches = re.finditer(self.pattern, self.text)
        return [match.group(self.group) for match in matches]


class RegexReplace(BaseNode):
    """
    Replace text matching a regex pattern.
    regex, replace, substitute

    Use cases:
    - Clean or standardize text
    - Remove unwanted patterns
    - Transform text formats
    """

    text: str = Field(default="", description="Text to perform replacements on")
    pattern: str = Field(default="", description="Regular expression pattern")
    replacement: str = Field(default="", description="Replacement text")
    count: int = Field(default=0, description="Maximum replacements (0 for unlimited)")

    @classmethod
    def get_title(cls):
        return "Replace with Regex"

    async def process(self, context: ProcessingContext) -> str:
        return re.sub(self.pattern, self.replacement, self.text, count=self.count)


class RegexSplit(BaseNode):
    """
    Split text using a regex pattern as delimiter.
    regex, split, tokenize

    Use cases:
    - Parse structured text
    - Extract fields from formatted strings
    - Tokenize text
    """

    text: str = Field(default="", description="Text to split")
    pattern: str = Field(
        default="", description="Regular expression pattern to split on"
    )
    maxsplit: int = Field(
        default=0, description="Maximum number of splits (0 for unlimited)"
    )

    @classmethod
    def get_title(cls):
        return "Split with Regex"

    async def process(self, context: ProcessingContext) -> list[str]:
        return re.split(self.pattern, self.text, maxsplit=self.maxsplit)


class RegexValidate(BaseNode):
    """
    Check if text matches a regex pattern.
    regex, validate, check

    Use cases:
    - Validate input formats (email, phone, etc)
    - Check text structure
    - Filter text based on patterns
    """

    text: str = Field(default="", description="Text to validate")
    pattern: str = Field(default="", description="Regular expression pattern")

    @classmethod
    def get_title(cls):
        return "Validate with Regex"

    async def process(self, context: ProcessingContext) -> bool:
        return bool(re.match(self.pattern, self.text))


class Compare(BaseNode):
    """
    Compares two text values and reports ordering.
    text, compare, equality, sort, equals, =

    Use cases:
    - Checking if two strings are identical before branching
    - Determining lexical order for sorting or deduplication
    - Normalizing casing/spacing before compares
    """

    text_a: str = Field(title="First Text", default="")
    text_b: str = Field(title="Second Text", default="")
    case_sensitive: bool = Field(
        title="Case Sensitive", default=True, description="Compare without lowercasing"
    )
    trim_whitespace: bool = Field(
        title="Trim Whitespace",
        default=False,
        description="Strip leading/trailing whitespace before comparing",
    )

    class ComparisonResult(str, Enum):
        LESS = "less"
        EQUAL = "equal"
        GREATER = "greater"

    @classmethod
    def get_title(cls):
        return "Compare Text"

    async def process(self, context: ProcessingContext) -> str:
        def normalize(value: str) -> str:
            result = value.strip() if self.trim_whitespace else value
            return result if self.case_sensitive else result.lower()

        left = normalize(self.text_a)
        right = normalize(self.text_b)

        if left < right:
            return self.ComparisonResult.LESS.value
        if left > right:
            return self.ComparisonResult.GREATER.value
        return self.ComparisonResult.EQUAL.value


class Equals(BaseNode):
    """
    Checks if two text inputs are equal.
    text, compare, equals, match, =
    """

    text_a: str = Field(title="First Text", default="")
    text_b: str = Field(title="Second Text", default="")
    case_sensitive: bool = Field(
        title="Case Sensitive",
        default=True,
        description="Disable lowercasing before compare",
    )
    trim_whitespace: bool = Field(
        title="Trim Whitespace",
        default=False,
        description="Strip leading/trailing whitespace prior to comparison",
    )

    @classmethod
    def get_title(cls):
        return "Equals"

    async def process(self, context: ProcessingContext) -> bool:
        def normalize(value: str) -> str:
            result = value.strip() if self.trim_whitespace else value
            return result if self.case_sensitive else result.lower()

        return normalize(self.text_a) == normalize(self.text_b)


class ToUppercase(BaseNode):
    """
    Converts text to uppercase.
    text, transform, uppercase, format
    """

    text: str = Field(title="Text", default="")

    @classmethod
    def get_title(cls):
        return "To Uppercase"

    async def process(self, context: ProcessingContext) -> str:
        return self.text.upper()


class ToLowercase(BaseNode):
    """
    Converts text to lowercase.
    text, transform, lowercase, format
    """

    text: str = Field(title="Text", default="")

    @classmethod
    def get_title(cls):
        return "To Lowercase"

    async def process(self, context: ProcessingContext) -> str:
        return self.text.lower()


class ToTitlecase(BaseNode):
    """
    Converts text to title case.
    text, transform, titlecase, format

    Use cases:
    - Cleaning user provided titles before display
    - Normalizing headings in generated documents
    - Making list entries easier to scan
    """

    text: str = Field(title="Text", default="")

    @classmethod
    def get_title(cls):
        return "To Title Case"

    async def process(self, context: ProcessingContext) -> str:
        return self.text.title()


class CapitalizeText(BaseNode):
    """
    Capitalizes only the first character.
    text, transform, capitalize, format

    Use cases:
    - Formatting short labels or sentences
    - Cleaning up LLM output before UI rendering
    - Quickly fixing lowercase starts after concatenation
    """

    text: str = Field(title="Text", default="")

    @classmethod
    def get_title(cls):
        return "Capitalize Text"

    async def process(self, context: ProcessingContext) -> str:
        return self.text.capitalize()


class Slice(BaseNode):
    """
    Slices text using Python's slice notation (start:stop:step).
    text, slice, substring

    Use cases:
    - Extracting specific portions of text with flexible indexing
    - Reversing text using negative step
    - Taking every nth character with step parameter

    Examples:
    - start=0, stop=5: first 5 characters
    - start=-5: last 5 characters
    - step=2: every second character
    - step=-1: reverse the text
    """

    text: str = Field(title="Text", default="")
    start: int = Field(title="Start Index", default=0)
    stop: int = Field(title="Stop Index", default=0)
    step: int = Field(title="Step", default=1)

    @classmethod
    def get_title(cls):
        return "Slice Text"

    async def process(self, context: ProcessingContext) -> str:
        return self.text[self.start : self.stop : self.step]


class StartsWith(BaseNode):
    """
    Checks if text starts with a specified prefix.
    text, check, prefix, compare, validate, substring, string

    Use cases:
    - Validating string prefixes
    - Filtering text based on starting content
    - Checking file name patterns
    """

    text: str = Field(title="Text", default="")
    prefix: str = Field(title="Prefix", default="")

    @classmethod
    def get_title(cls):
        return "Starts With"

    async def process(self, context: ProcessingContext) -> bool:
        return self.text.startswith(self.prefix)


class EndsWith(BaseNode):
    """
    Checks if text ends with a specified suffix.
    text, check, suffix, compare, validate, substring, string

    Use cases:
    - Validating file extensions
    - Checking string endings
    - Filtering text based on ending content
    """

    text: str = Field(title="Text", default="")
    suffix: str = Field(title="Suffix", default="")

    @classmethod
    def get_title(cls):
        return "Ends With"

    async def process(self, context: ProcessingContext) -> bool:
        return self.text.endswith(self.suffix)


class Contains(BaseNode):
    """
    Checks if text contains a specified substring.
    text, compare, validate, substring, string
    """

    class MatchMode(str, Enum):
        ANY = "any"
        ALL = "all"
        NONE = "none"

    text: str = Field(title="Text", default="")
    substring: str = Field(title="Substring", default="")
    search_values: list[str] = Field(
        default=[],
        description="Optional list of additional substrings to check",
    )
    case_sensitive: bool = Field(title="Case Sensitive", default=True)
    match_mode: MatchMode = Field(
        title="Match Mode",
        default=MatchMode.ANY,
        description="ANY requires one match, ALL needs every value, NONE ensures none",
    )

    @classmethod
    def get_title(cls):
        return "Contains Text"

    async def process(self, context: ProcessingContext) -> bool:
        targets = self.search_values or [self.substring]
        if not targets:
            return False

        haystack = self.text if self.case_sensitive else self.text.lower()
        needles = (
            targets if self.case_sensitive else [needle.lower() for needle in targets]
        )

        if self.match_mode == self.MatchMode.ALL:
            return all(needle in haystack for needle in needles)
        if self.match_mode == self.MatchMode.NONE:
            return all(needle not in haystack for needle in needles)
        return any(needle in haystack for needle in needles)


class TrimWhitespace(BaseNode):
    """
    Trims whitespace from the start and/or end of text.
    text, whitespace, clean, remove
    """

    text: str = Field(title="Text", default="")
    trim_start: bool = Field(title="Trim Start", default=True)
    trim_end: bool = Field(title="Trim End", default=True)

    @classmethod
    def get_title(cls):
        return "Trim Whitespace"

    async def process(self, context: ProcessingContext) -> str:
        if self.trim_start and self.trim_end:
            return self.text.strip()
        if self.trim_start:
            return self.text.lstrip()
        if self.trim_end:
            return self.text.rstrip()
        return self.text


class CollapseWhitespace(BaseNode):
    """
    Collapses consecutive whitespace into single separators.
    text, whitespace, normalize, clean, remove
    """

    text: str = Field(title="Text", default="")
    preserve_newlines: bool = Field(
        title="Preserve Newlines",
        default=False,
        description="Keep newline characters instead of replacing them",
    )
    replacement: str = Field(
        title="Replacement",
        default=" ",
        description="String used to replace whitespace runs",
    )
    trim_edges: bool = Field(
        title="Trim Edges",
        default=True,
        description="Strip whitespace before collapsing",
    )

    @classmethod
    def get_title(cls):
        return "Collapse Whitespace"

    async def process(self, context: ProcessingContext) -> str:
        value = self.text.strip() if self.trim_edges else self.text
        if self.preserve_newlines:
            return re.sub(r"[^\S\r\n]+", self.replacement, value)
        return re.sub(r"\s+", self.replacement, value)


class IsEmpty(BaseNode):
    """
    Checks if text is empty or contains only whitespace.
    text, check, empty, compare, validate, whitespace, string
    """

    text: str = Field(title="Text", default="")
    trim_whitespace: bool = Field(title="Trim Whitespace", default=True)

    @classmethod
    def get_title(cls):
        return "Is Empty"

    async def process(self, context: ProcessingContext) -> bool:
        text = self.text
        if self.trim_whitespace:
            text = text.strip()
        return len(text) == 0


class RemovePunctuation(BaseNode):
    """
    Removes punctuation characters from text.
    text, cleanup, punctuation, normalize
    """

    text: str = Field(title="Text", default="")
    replacement: str = Field(
        title="Replacement",
        default="",
        description="String to insert where punctuation was removed",
    )
    punctuation: str = Field(
        title="Punctuation Characters",
        default=string.punctuation,
        description="Characters that should be removed or replaced",
    )

    @classmethod
    def get_title(cls):
        return "Remove Punctuation"

    async def process(self, context: ProcessingContext) -> str:
        translation = {ord(ch): self.replacement for ch in self.punctuation}
        return self.text.translate(translation)


class StripAccents(BaseNode):
    """
    Removes accent marks while keeping base characters.
    text, cleanup, accents, normalize
    """

    text: str = Field(title="Text", default="")
    preserve_non_ascii: bool = Field(
        title="Preserve Non-ASCII",
        default=True,
        description="Keep non-ASCII characters that are not accents",
    )

    @classmethod
    def get_title(cls):
        return "Strip Accents"

    async def process(self, context: ProcessingContext) -> str:
        normalized = unicodedata.normalize("NFKD", self.text)
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        if self.preserve_non_ascii:
            return stripped
        return stripped.encode("ascii", "ignore").decode("ascii")


class Slugify(BaseNode):
    """
    Converts text into a slug suitable for URLs or IDs.
    text, slug, normalize, id
    """

    text: str = Field(title="Text", default="")
    separator: str = Field(title="Separator", default="-")
    lowercase: bool = Field(title="Lowercase", default=True)
    allow_unicode: bool = Field(
        title="Allow Unicode",
        default=False,
        description="Keep unicode letters instead of converting to ASCII",
    )

    @classmethod
    def get_title(cls):
        return "Slugify"

    async def process(self, context: ProcessingContext) -> str:
        value = unicodedata.normalize("NFKD", self.text)
        if not self.allow_unicode:
            value = value.encode("ascii", "ignore").decode("ascii")
        value = re.sub(r"[^\w\s-]", "", value)
        value = re.sub(r"[\s_-]+", self.separator, value).strip(self.separator)
        if self.lowercase:
            value = value.lower()
        return value


class HasLength(BaseNode):
    """
    Checks if text length meets specified conditions.
    text, check, length, compare, validate, whitespace, string
    """

    text: str = Field(title="Text", default="")
    min_length: int = Field(title="Minimum Length", default=0)
    max_length: int = Field(title="Maximum Length", default=0)
    exact_length: int = Field(title="Exact Length", default=0)

    @classmethod
    def get_title(cls):
        return "Check Length"

    async def process(self, context: ProcessingContext) -> bool:
        length = len(self.text)

        if self.exact_length is not None:
            return length == self.exact_length

        if self.min_length is not None and length < self.min_length:
            return False

        if self.max_length is not None and length > self.max_length:
            return False

        return True


class TruncateText(BaseNode):
    """
    Truncates text to a maximum length.
    text, truncate, length, clip
    """

    text: str = Field(title="Text", default="")
    max_length: int = Field(title="Max Length", default=100, ge=0)
    ellipsis: str = Field(
        title="Ellipsis",
        default="",
        description="Optional suffix appended when truncation occurs",
    )

    @classmethod
    def get_title(cls):
        return "Truncate Text"

    async def process(self, context: ProcessingContext) -> str:
        if self.max_length <= 0:
            return self.ellipsis if self.ellipsis else ""
        if len(self.text) <= self.max_length:
            return self.text
        if self.ellipsis and len(self.ellipsis) < self.max_length:
            cut = self.max_length - len(self.ellipsis)
            return f"{self.text[:cut]}{self.ellipsis}"
        return self.text[: self.max_length]


class PadText(BaseNode):
    """
    Pads text to a target length.
    text, pad, length, format
    """

    class PadDirection(str, Enum):
        LEFT = "left"
        RIGHT = "right"
        BOTH = "both"

    text: str = Field(title="Text", default="")
    length: int = Field(title="Target Length", default=0, ge=0)
    pad_character: str = Field(
        title="Pad Character",
        default=" ",
        description="Single character to use for padding",
    )
    direction: PadDirection = Field(
        title="Direction",
        default=PadDirection.RIGHT,
        description="Where padding should be applied",
    )

    @classmethod
    def get_title(cls):
        return "Pad Text"

    async def process(self, context: ProcessingContext) -> str:
        if len(self.pad_character) != 1:
            raise ValueError("pad_character must be a single character")
        if self.length <= len(self.text):
            return self.text

        needed = self.length - len(self.text)
        if self.direction == self.PadDirection.LEFT:
            return self.pad_character * needed + self.text
        if self.direction == self.PadDirection.BOTH:
            left = needed // 2
            right = needed - left
            return f"{self.pad_character * left}{self.text}{self.pad_character * right}"
        return self.text + self.pad_character * needed


class Length(BaseNode):
    """
    Measures text length as characters, words, or lines.
    text, analyze, length, count
    """

    class Measure(str, Enum):
        CHARACTERS = "characters"
        WORDS = "words"
        LINES = "lines"

    text: str = Field(title="Text", default="")
    measure: Measure = Field(
        title="Measure",
        default=Measure.CHARACTERS,
        description="Choose whether to count characters, words, or lines",
    )
    trim_whitespace: bool = Field(
        title="Trim Whitespace",
        default=False,
        description="Strip whitespace before counting",
    )

    @classmethod
    def get_title(cls):
        return "Measure Length"

    async def process(self, context: ProcessingContext) -> int:
        value = self.text.strip() if self.trim_whitespace else self.text

        if self.measure == self.Measure.WORDS:
            return len([word for word in value.split() if word])
        if self.measure == self.Measure.LINES:
            if not value:
                return 0
            return len(
                [
                    line
                    for line in value.splitlines()
                    if line or not self.trim_whitespace
                ]
            )
        return len(value)


class IndexOf(BaseNode):
    """
    Finds the position of a substring in text.
    text, search, find, substring
    """

    text: str = Field(title="Text", default="")
    substring: str = Field(title="Substring", default="")
    case_sensitive: bool = Field(title="Case Sensitive", default=True)
    start_index: int = Field(
        title="Start Index",
        default=0,
        description="Index to begin the search from",
        ge=0,
    )
    end_index: int = Field(
        title="End Index",
        default=0,
        description="Optional exclusive end index for the search",
    )
    search_from_end: bool = Field(
        title="Search From End",
        default=False,
        description="Use the last occurrence instead of the first",
    )

    @classmethod
    def get_title(cls):
        return "Index Of"

    async def process(self, context: ProcessingContext) -> int:
        haystack = self.text
        needle = self.substring
        if not self.case_sensitive:
            haystack = haystack.lower()
            needle = needle.lower()

        end = self.end_index if self.end_index is not None else len(haystack)
        end = max(self.start_index, end)

        if self.search_from_end:
            return haystack.rfind(needle, self.start_index, end)
        return haystack.find(needle, self.start_index, end)


class SurroundWith(BaseNode):
    """
    Wraps text with the provided prefix and suffix.
    text, format, surround, decorate
    """

    text: str = Field(title="Text", default="")
    prefix: str = Field(title="Prefix", default="")
    suffix: str = Field(title="Suffix", default="")
    skip_if_wrapped: bool = Field(
        title="Skip If Already Wrapped",
        default=True,
        description="Do not add duplicates if the text is already wrapped",
    )

    @classmethod
    def get_title(cls):
        return "Surround With"

    async def process(self, context: ProcessingContext) -> str:
        if (
            self.skip_if_wrapped
            and self.text.startswith(self.prefix)
            and self.text.endswith(self.suffix)
        ):
            return self.text
        return f"{self.prefix}{self.text}{self.suffix}"


class CountTokens(BaseNode):
    """
    Counts the number of tokens in text using tiktoken.
    text, tokens, count, encoding
    """

    class TiktokenEncoding(str, Enum):
        """Available tiktoken encodings"""

        CL100K_BASE = "cl100k_base"  # GPT-4, GPT-3.5
        P50K_BASE = "p50k_base"  # GPT-3
        R50K_BASE = "r50k_base"  # GPT-2

    text: str = Field(title="Text", default="")
    encoding: TiktokenEncoding = Field(
        title="Encoding",
        default=TiktokenEncoding.CL100K_BASE,
        description="The tiktoken encoding to use for token counting",
    )

    @classmethod
    def get_title(cls):
        return "Count Tokens"

    async def process(self, context: ProcessingContext) -> int:
        import tiktoken

        encoding = tiktoken.get_encoding(self.encoding.value)
        return len(encoding.encode(self.text))


class HtmlToText(BaseNode):
    """
    Converts HTML content to plain text using html2text.
    html, convert, text, parse, extract
    """

    html: str = Field(title="HTML", default="", description="HTML content to convert")
    base_url: str = Field(
        title="Base URL",
        default="",
        description="Base URL for resolving relative links",
    )
    body_width: int = Field(
        title="Body Width", default=1000, description="Width for text wrapping"
    )
    ignore_images: bool = Field(
        title="Ignore Images", default=True, description="Whether to ignore image tags"
    )
    ignore_mailto_links: bool = Field(
        title="Ignore Mailto Links",
        default=True,
        description="Whether to ignore mailto links",
    )

    @classmethod
    def get_title(cls):
        return "HTML to Text"

    async def process(self, context: ProcessingContext) -> str:
        # Convert to plain text
        h = html2text.HTML2Text(baseurl=self.base_url, bodywidth=self.body_width)
        h.ignore_images = self.ignore_images
        h.ignore_mailto_links = self.ignore_mailto_links
        content = h.handle(self.html)
        return content


class LoadTextAssets(BaseNode):
    """
    Load text files from an asset folder.
    load, text, file, import
    """

    folder: FolderRef = Field(
        default=FolderRef(), description="The asset folder to load the text files from."
    )

    @classmethod
    def get_title(cls):
        return "Load Text Assets"

    class OutputType(TypedDict):
        text: TextRef
        name: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        if self.folder.is_empty():
            raise ValueError("Please select an asset folder.")

        parent_id = self.folder.asset_id
        list_assets, _ = await context.list_assets(
            parent_id=parent_id, content_type="text"
        )

        for asset in list_assets:
            yield {
                "name": asset.name,
                "text": TextRef(
                    type="text",
                    uri=await context.get_asset_url(asset.id),
                ),
            }


class FilterString(BaseNode):
    """
    Filters a stream of strings based on various criteria.
    filter, strings, text, stream
    """

    class FilterType(str, Enum):
        CONTAINS = "contains"
        STARTS_WITH = "starts_with"
        ENDS_WITH = "ends_with"
        LENGTH_GREATER = "length_greater"
        LENGTH_LESS = "length_less"
        EXACT_LENGTH = "exact_length"

    value: str = Field(default="", description="Input string stream")
    filter_type: FilterType = Field(
        default=FilterType.CONTAINS, description="The type of filter to apply"
    )
    criteria: str = Field(
        default="",
        description="The filtering criteria (text to match or length as string)",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        current_filter_type = self.filter_type
        current_criteria = self.criteria

        async for handle, item in self.iter_any_input():
            if handle == "filter_type":
                current_filter_type = item
                continue
            elif handle == "criteria":
                current_criteria = item
                continue
            elif handle == "value":
                val = item
                if not isinstance(val, str):
                    continue

                length_criteria = 0
                if current_filter_type in [
                    self.FilterType.LENGTH_GREATER,
                    self.FilterType.LENGTH_LESS,
                    self.FilterType.EXACT_LENGTH,
                ]:
                    try:
                        length_criteria = int(current_criteria)
                    except ValueError:
                        continue  # Skip if invalid criteria for length

                matched = False
                if current_filter_type == self.FilterType.CONTAINS:
                    if current_criteria in val:
                        matched = True
                elif current_filter_type == self.FilterType.STARTS_WITH:
                    if val.startswith(current_criteria):
                        matched = True
                elif current_filter_type == self.FilterType.ENDS_WITH:
                    if val.endswith(current_criteria):
                        matched = True
                elif current_filter_type == self.FilterType.LENGTH_GREATER:
                    if len(val) > length_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.LENGTH_LESS:
                    if len(val) < length_criteria:
                        matched = True
                elif current_filter_type == self.FilterType.EXACT_LENGTH:
                    if len(val) == length_criteria:
                        matched = True

                if matched:
                    yield {"output": val}


class FilterRegexString(BaseNode):
    """
    Filters a stream of strings using regular expressions.
    filter, regex, pattern, text, stream
    """

    value: str = Field(default="", description="Input string stream")
    pattern: str = Field(
        default="", description="The regular expression pattern to match against."
    )
    full_match: bool = Field(
        default=False,
        description="Whether to match the entire string or find pattern anywhere in string",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    class OutputType(TypedDict):
        output: str

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        import re

        current_pattern = self.pattern
        current_full_match = self.full_match
        regex = None

        try:
            regex = re.compile(current_pattern)
        except re.error:
            pass  # Handle invalid regex gracefully (maybe log or just don't match)

        async for handle, item in self.iter_any_input():
            if handle == "pattern":
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
                if regex is None:
                    try:
                        # Fallback try to compile if it wasn't valid before
                        regex = re.compile(current_pattern)
                    except re.error:
                        continue

                val = item
                if not isinstance(val, str):
                    continue

                matched = False
                if current_full_match:
                    if regex.fullmatch(val):
                        matched = True
                else:
                    if regex.search(val):
                        matched = True

                if matched:
                    yield {"output": val}


class Embedding(BaseNode):
    """
    Generate vector representations of text using any supported embedding provider.
    Automatically routes to the appropriate backend (OpenAI, Gemini, Mistral).
    embeddings, similarity, search, clustering, classification, vectors, semantic

    Uses embedding models to create dense vector representations of text.
    These vectors capture semantic meaning, enabling:
    - Semantic search
    - Text clustering
    - Document classification
    - Recommendation systems
    - Anomaly detection
    - Measuring text similarity and diversity
    """

    _expose_as_tool: ClassVar[bool] = True

    model: EmbeddingModel = Field(
        default=EmbeddingModel(
            provider=Provider.OpenAI,
            id="text-embedding-3-small",
            name="Text Embedding 3 Small",
        ),
        description="The embedding model to use",
    )
    input: str = Field(
        title="Input",
        default="",
        description="The text to embed",
    )
    chunk_size: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Size of text chunks for embedding (used when input exceeds model limits)",
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        """
        Generate embeddings for the input text using the specified model.

        Returns:
            NPArray: The embedding vector representation of the text
        """
        if not self.input:
            raise ValueError("Input text cannot be empty")

        if self.model.provider == Provider.Empty:
            raise ValueError("Please select an embedding model")

        # Chunk the input into smaller pieces if necessary
        chunks = [
            self.input[i : i + self.chunk_size]
            for i in range(0, len(self.input), self.chunk_size)
        ]

        if self.model.provider == Provider.OpenAI:
            return await self._process_openai(context, chunks)
        elif self.model.provider == Provider.Gemini:
            return await self._process_gemini(context)
        elif self.model.provider == Provider.Mistral:
            return await self._process_mistral(context, chunks)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.model.provider}")

    async def _process_openai(
        self, context: ProcessingContext, chunks: list[str]
    ) -> NPArray:
        """Process embedding using OpenAI provider."""
        import numpy as np
        from nodetool.providers.openai_prediction import run_openai
        from openai.types.create_embedding_response import CreateEmbeddingResponse

        response = await context.run_prediction(
            self.id,
            provider="openai",
            params={"input": chunks},
            model=self.model.id,
            run_prediction_function=run_openai,
        )

        res = CreateEmbeddingResponse(**response)
        all_embeddings = [i.embedding for i in res.data]
        avg = np.mean(all_embeddings, axis=0)
        return NPArray.from_numpy(avg)

    async def _process_gemini(self, context: ProcessingContext) -> NPArray:
        """Process embedding using Gemini provider."""
        import numpy as np
        from nodetool.providers.gemini_provider import GeminiProvider

        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]

        # Generate embedding using Gemini's embedding API
        response = await client.models.embed_content(
            model=self.model.id,
            contents=self.input,
        )

        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("No embedding generated from the input text")

        embedding_values = response.embeddings[0].values
        return NPArray.from_numpy(np.array(embedding_values))

    async def _process_mistral(
        self, context: ProcessingContext, chunks: list[str]
    ) -> NPArray:
        """Process embedding using Mistral provider."""
        import numpy as np

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        from mistralai import Mistral

        client = Mistral(api_key=api_key)

        response = await client.embeddings.create_async(
            model=self.model.id,
            inputs=chunks,
        )

        if not response or not response.data:
            raise ValueError("No embeddings received from Mistral API")

        all_embeddings = [item.embedding for item in response.data]
        avg_embedding = np.mean(all_embeddings, axis=0)
        return NPArray.from_numpy(avg_embedding)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["model", "input"]
