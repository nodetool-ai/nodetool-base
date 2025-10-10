from datetime import datetime
from io import BytesIO
import json
import os

from typing import Any, AsyncGenerator, ClassVar, TypedDict
from nodetool.providers import get_provider
from nodetool.workflows.io import NodeInputs, NodeOutputs
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ASRModel, AudioChunk, FolderRef, AudioRef, Provider
from nodetool.workflows.base_node import BaseNode
import re
from jsonpath_ng import parse
from nodetool.metadata.types import TextRef
from enum import Enum
import html2text
from nodetool.io.uri_utils import create_file_uri
from nodetool.workflows.types import SaveUpdate


class AutomaticSpeechRecognition(BaseNode):
    """
    Automatic speech recognition node.
    audio, speech, recognition
    """

    class OutputType(TypedDict):
        text: str

    _expose_as_tool: ClassVar[bool] = True

    model: ASRModel = Field(
        default=ASRModel(
            provider=Provider.FalAI, id="openai/whisper-large-v3"
        )
    )
    audio: AudioRef = Field(default=AudioRef(), description="The audio to transcribe")

    async def process(self, context: ProcessingContext) -> OutputType:
        provider = get_provider(self.model.provider)
        audio_bytes = await context.asset_to_bytes(self.audio)
        text = await provider.automatic_speech_recognition(
            audio_bytes, model=self.model.id, 
        )
        return {
            "text": text
        }

class Concat(BaseNode):
    """
    Concatenates two text inputs into a single output.
    text, concatenation, combine, +

    Use cases:
    - Joining outputs from multiple text processing nodes
    - Combining parts of sentences or paragraphs
    - Merging text data from different sources
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
    text, join, combine, +, add, concatenate

    Use cases:
    - Combining multiple text elements with a consistent delimiter
    - Creating comma-separated lists from individual items
    - Assembling formatted text from array elements
    """

    strings: list[str] = Field(default=[])
    separator: str = Field(default="")

    @classmethod
    def get_title(cls):
        return "Join"

    async def process(self, context: ProcessingContext) -> str:
        if len(self.strings) == 0:
            return ""
        return self.separator.join(self.strings)


class Collect(BaseNode):
    """
    Collects a stream of text inputs into a single string.
    text, collect, list, stream
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

    _is_dynamic: ClassVar[bool] = True

    template: str = Field(
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
        default_factory=dict,
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
        from jinja2 import Environment, BaseLoader

        try:
            # Create Jinja2 environment
            env = Environment(loader=BaseLoader())
            template = env.from_string(self.string)

            # Merge values from the values field and dynamic properties
            template_values = {}
            if isinstance(self.values, dict):
                template_values.update(self.values)
            template_values.update(self._dynamic_properties)

            # Render template
            return template.render(**template_values)

        except Exception as e:
            raise ValueError(f"Template error: {str(e)}")


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
        file = BytesIO(self.text.encode("utf-8"))
        if not self.folder:
            raise ValueError("folder cannot be empty")
        expanded_folder = os.path.expanduser(self.folder)
        if not os.path.exists(expanded_folder):
            raise ValueError(f"Folder does not exist: {expanded_folder}")
        filename = datetime.now().strftime(self.name)
        expanded_path = os.path.join(expanded_folder, filename)
        with open(expanded_path, "wb") as f:
            f.write(file.getvalue())
        result = TextRef(uri=create_file_uri(expanded_path), data=file.getvalue())

        # Emit SaveUpdate event
        context.post_message(SaveUpdate(
            node_id=self.id,
            name=filename,
            value=result,
            output_type="text"
        ))

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
        context.post_message(SaveUpdate(
            node_id=self.id,
            name=filename,
            value=result,
            output_type="text"
        ))

        return result


class Split(BaseNode):
    """
    Separates text into a list of strings based on a specified delimiter.
    text, split, tokenize

    Use cases:
    - Parsing CSV or similar delimited data
    - Breaking down sentences into words or phrases
    - Extracting specific elements from structured text
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

    Use cases:
    - Extracting specific portions of text for analysis
    - Trimming unwanted parts from text data
    - Focusing on relevant sections of longer documents
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

    Use cases:
    - Preparing text for processing by models with input length limits
    - Creating manageable text segments for parallel processing
    - Generating summaries of text sections
    """

    text: str = Field(title="Text", default="")
    length: int = Field(title="Length", default=100, le=1000, ge=1)
    overlap: int = Field(title="Overlap", default=0)
    separator: str | None = Field(title="Separator", default=None)

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

    Use cases:
    - Identifying all occurrences of a pattern in text
    - Extracting multiple instances of structured data
    - Analyzing frequency and distribution of specific text patterns
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

    Use cases:
    - Converting JSON API responses for further processing
    - Preparing structured data for analysis or storage
    - Extracting configuration or settings from JSON files
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

    Use cases:
    - Retrieving specific fields from complex JSON structures
    - Filtering and transforming JSON data for analysis
    - Extracting nested data from API responses or configurations
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
    group: int | None = Field(
        default=None, description="Capture group to extract (0 for full match)"
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
    start: int | None = Field(title="Start Index", default=None)
    stop: int | None = Field(title="Stop Index", default=None)
    step: int | None = Field(title="Step", default=None)

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
    text, check, contains, compare, validate, substring, string

    Use cases:
    - Searching for keywords in text
    - Filtering content based on presence of terms
    - Validating text content
    """

    text: str = Field(title="Text", default="")
    substring: str = Field(title="Substring", default="")
    case_sensitive: bool = Field(title="Case Sensitive", default=True)

    @classmethod
    def get_title(cls):
        return "Contains Text"

    async def process(self, context: ProcessingContext) -> bool:
        text = self.text
        if not self.case_sensitive:
            text = text.lower()
            substring = self.substring.lower()
        else:
            substring = self.substring
        return substring in text


class IsEmpty(BaseNode):
    """
    Checks if text is empty or contains only whitespace.
    text, check, empty, compare, validate, whitespace, string

    Use cases:
    - Validating required text fields
    - Filtering out empty content
    - Checking for meaningful input
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


class HasLength(BaseNode):
    """
    Checks if text length meets specified conditions.
    text, check, length, compare, validate, whitespace, string

    Use cases:
    - Validating input length requirements
    - Filtering text by length
    - Checking content size constraints
    """

    text: str = Field(title="Text", default="")
    min_length: int | None = Field(title="Minimum Length", default=None)
    max_length: int | None = Field(title="Maximum Length", default=None)
    exact_length: int | None = Field(title="Exact Length", default=None)

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


class CountTokens(BaseNode):
    """
    Counts the number of tokens in text using tiktoken.
    text, tokens, count, encoding

    Use cases:
    - Checking text length for LLM input limits
    - Estimating API costs
    - Managing token budgets in text processing
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

    Use cases:
    - Converting HTML documents to readable plain text
    - Extracting text content from web pages
    - Cleaning HTML markup from text data
    - Processing HTML emails or documents
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

    Use cases:
    - Loading multiple text files for batch processing
    - Importing text content from a directory
    - Processing collections of text documents
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
                    asset_id=asset.id,
                ),
            }
