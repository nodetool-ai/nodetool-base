import asyncio
import datetime
import os
from enum import Enum
from typing import Any, ClassVar, List, TypedDict
from urllib.parse import urljoin

import aiohttp
from pydantic import Field

from nodetool.metadata.types import (
    ColumnDef,
    DataframeRef,
    DocumentRef,
    ImageRef,
    RecordType,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)


class HTTPBaseNode(BaseNode):
    """Base class for HTTP request nodes.

    Provides common URL field and request configuration for HTTP operations.
    Not directly visible in the UI; extended by specific HTTP method nodes.

    http, network, request
    """

    url: str = Field(
        default="",
        description="The URL to make the request to.",
    )

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not HTTPBaseNode

    def get_request_kwargs(self) -> dict[str, Any]:
        return {}

    @classmethod
    def get_basic_fields(cls):
        return ["url"]


class GetRequest(HTTPBaseNode):
    """
    Perform an HTTP GET request and return the response body as text.

    Makes a synchronous HTTP GET request to the specified URL and decodes the response
    content using the detected or default UTF-8 encoding. Returns the full response body
    as a string.

    Parameters:
    - url (required): The complete URL to request, including protocol (http:// or https://)

    Returns: Response body decoded as text string

    Side effects: Network request to external URL

    Typical usage: Fetch HTML pages, API responses, or text-based resources. Follow with
    parsing nodes (BeautifulSoup, JSON) or text processing nodes to extract structured data.

    http, get, request, url
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "GET Request"

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return res.content.decode(res.encoding or "utf-8")


class PostRequest(HTTPBaseNode):
    """
    Send data to a server using HTTP POST and return the response body as text.

    Makes an HTTP POST request with the provided data payload. Response is decoded
    using the detected or default UTF-8 encoding.

    Parameters:
    - url (required): Target endpoint URL
    - data (optional, default=""): Payload to send in the request body as string

    Returns: Response body decoded as text string

    Side effects: Network request that may create or modify server-side resources

    Typical usage: Submit form data, create API resources, or trigger server-side actions.
    For structured JSON data, use JSONPostRequest instead. Follow with response parsing
    nodes if the server returns structured data.

    http, post, request, url, data
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "POST Request"

    data: str = Field(
        default="",
        description="The data to send in the POST request.",
    )

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_post(
            self.url, data=self.data, **self.get_request_kwargs()
        )
        return res.content.decode(res.encoding or "utf-8")


class PutRequest(HTTPBaseNode):
    """
    Update or replace resources using HTTP PUT and return the response body as text.

    Makes an HTTP PUT request with the provided data payload. Used for full resource
    replacement in REST APIs. Response is decoded as text.

    Parameters:
    - url (required): Target resource URL
    - data (optional, default=""): Replacement data to send in the request body

    Returns: Response body decoded as text string

    Side effects: Network request that replaces or creates server-side resources

    Typical usage: Update complete API resources, replace configuration, or modify
    server state. For partial updates use JSONPatchRequest. For JSON payloads use
    JSONPutRequest instead.

    http, put, request, url, data
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "PUT Request"

    data: str = Field(
        default="",
        description="The data to send in the PUT request.",
    )

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_put(
            self.url, data=self.data, **self.get_request_kwargs()
        )
        return res.content.decode(res.encoding or "utf-8")


class DeleteRequest(HTTPBaseNode):
    """
    Remove a resource using HTTP DELETE and return the response body as text.

    Makes an HTTP DELETE request to remove the resource at the specified URL.
    Response is decoded as text and may contain deletion confirmation or status.

    Parameters:
    - url (required): URL of the resource to delete

    Returns: Response body decoded as text string

    Side effects: Network request that removes server-side resources

    Typical usage: Delete API resources, cancel operations, or clear server-side state.
    Typically followed by conditional logic to verify successful deletion or handle errors.

    http, delete, request, url
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "DELETE Request"

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_delete(self.url, **self.get_request_kwargs())
        return res.content.decode(res.encoding or "utf-8")


class HeadRequest(HTTPBaseNode):
    """
    Retrieve HTTP headers without downloading the full response body.

    Makes an HTTP HEAD request to fetch only the response headers, useful for checking
    resource metadata, existence, or size without transferring the entire content.

    Parameters:
    - url (required): URL to check

    Returns: Dictionary of header name-value pairs

    Side effects: Network request (minimal data transfer)

    Typical usage: Verify resource existence before downloading, check content type or
    size, validate URLs, or check authentication. Often precedes conditional GET requests
    or used in URL validation workflows.

    http, head, request, url
    """

    @classmethod
    def get_title(cls):
        return "HEAD Request"

    async def process(self, context: ProcessingContext) -> dict[str, str]:
        res = await context.http_head(self.url, **self.get_request_kwargs())
        return dict(res.headers.items())


class FetchPage(BaseNode):
    """
    Fetch a web page using headless Chrome browser and return fully-rendered HTML.

    Uses Selenium with headless Chrome to load the page, execute JavaScript, and wait
    for the DOM to be ready. Returns the complete rendered HTML including content
    generated by client-side scripts.

    Parameters:
    - url (required): Web page URL to fetch
    - wait_time (optional, default=10): Maximum seconds to wait for page load

    Returns: Dictionary with "html" (rendered page source), "success" (boolean),
    and "error_message" (string or None)

    Side effects: Launches Chrome browser process, network request, executes JavaScript

    Typical usage: Scrape JavaScript-heavy websites (SPAs, React apps), capture dynamic
    content, or interact with web applications. Follow with BeautifulSoup nodes to parse
    the rendered HTML. Use GetRequest for static pages to avoid browser overhead.

    selenium, fetch, webpage, http
    """

    url: str = Field(
        default="",
        description="The URL to fetch the page from.",
    )
    wait_time: int = Field(
        default=10,
        description="Maximum time to wait for page load (in seconds).",
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        html: str
        success: bool
        error_message: str | None

    async def process(self, context: ProcessingContext) -> OutputType:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        html = ""

        try:
            driver.get(self.url)
            WebDriverWait(driver, self.wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            html = driver.page_source

            return {
                "html": html,
                "success": True,
                "error_message": None,
            }
        except Exception as e:
            return {
                "html": html,
                "success": False,
                "error_message": str(e),
            }
        finally:
            driver.quit()


class ImageDownloader(BaseNode):
    """
    Download multiple images concurrently from URLs and return as ImageRef list.

    Downloads images in parallel with configurable concurrency. Handles relative URLs
    when base_url is provided. Failed downloads are tracked separately.

    Parameters:
    - images (required): List of image URLs (absolute or relative)
    - base_url (optional, default=""): Base URL prepended to relative image URLs
    - max_concurrent_downloads (optional, default=10): Parallel download limit

    Returns: Dictionary with "images" (list of ImageRef objects for successful downloads)
    and "failed_urls" (list of strings for failed URLs)

    Side effects: Multiple concurrent network requests, image data stored in context

    Typical usage: Batch download images from web scraping results, prepare image
    datasets, or collect visual assets. Precede with web scraping or API nodes that
    extract image URLs. Follow with image processing or ML model nodes.

    image download, web scraping, data processing
    """

    images: list[str] = Field(
        default=[],
        description="List of image URLs to download.",
    )
    base_url: str = Field(
        default="",
        description="Base URL to prepend to relative image URLs.",
    )
    max_concurrent_downloads: int = Field(
        default=10,
        description="Maximum number of concurrent image downloads.",
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        images: list[ImageRef]
        failed_urls: list[str]

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        context: ProcessingContext,
    ) -> tuple[ImageRef | None, str | None]:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    image_ref = await context.image_from_bytes(content)
                    return image_ref, None
                else:
                    error_msg = f"Failed to download image from {url}. Status code: {response.status}"
                    logger.warning(error_msg)
                    return None, url
        except Exception as e:
            error_msg = f"Error downloading image from {url}: {str(e)}"
            logger.warning(error_msg)
            return None, url

    async def process(self, context: ProcessingContext) -> OutputType:
        images = []
        failed_urls = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for src in self.images:
                url = urljoin(self.base_url, src)
                task = self.download_image(session, url, context)
                tasks.append(task)

                if len(tasks) >= self.max_concurrent_downloads:
                    completed = await asyncio.gather(*tasks)
                    for img, failed_url in completed:
                        if img is not None:
                            images.append(img)
                        if failed_url is not None:
                            failed_urls.append(failed_url)
                    tasks = []

            if tasks:
                completed = await asyncio.gather(*tasks)
                for img, failed_url in completed:
                    if img is not None:
                        images.append(img)
                    if failed_url is not None:
                        failed_urls.append(failed_url)

        return {
            "images": images,
            "failed_urls": failed_urls,
        }


class GetRequestBinary(HTTPBaseNode):
    """
    Perform HTTP GET request and return response as raw binary bytes.

    Retrieves the resource without any text decoding, preserving the exact binary
    content. Use for any non-text resources.

    Parameters:
    - url (required): URL of the binary resource

    Returns: Raw bytes of the response content

    Side effects: Network request

    Typical usage: Download binary files, images, media, PDFs, or any non-text content.
    Follow with binary-to-image nodes, document processing nodes, or file save nodes.
    For text responses use GetRequest; for images use GetRequestDocument or ImageDownloader.

    http, get, request, url, binary, download
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "GET Binary"

    async def process(self, context: ProcessingContext) -> bytes:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return res.content


class GetRequestDocument(HTTPBaseNode):
    """
    Perform HTTP GET request and return response as DocumentRef.

    Downloads the resource and wraps the binary content in a DocumentRef for use
    with document processing nodes.

    Parameters:
    - url (required): URL of the document to download

    Returns: DocumentRef containing the document binary data

    Side effects: Network request

    Typical usage: Download PDFs, Word docs, Excel files, or other documents for
    processing. Follow with PDF extraction nodes (PDFPlumber, PyMuPDF), document
    conversion nodes (Pandoc, MarkItDown), or OCR nodes for image-based documents.

    http, get, request, url, document
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "GET Document"

    async def process(self, context: ProcessingContext) -> DocumentRef:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return DocumentRef(data=res.content)


class PostRequestBinary(HTTPBaseNode):
    """
    Send data via HTTP POST and return the response as raw binary bytes.

    Posts the provided data (string or binary) and returns the unencoded response.
    Useful for APIs that return binary content in response to uploads or transformations.

    Parameters:
    - url (required): Target endpoint URL
    - data (optional, default=""): Data to send (string or bytes)

    Returns: Raw bytes of the response content

    Side effects: Network request that may create or modify resources

    Typical usage: Post data to APIs that return binary responses (image processing APIs,
    file converters, media encoders). Follow with binary-to-image, document processing,
    or file save nodes.

    http, post, request, url, data, binary
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "POST Binary"

    data: str | bytes = Field(
        default="",
        description="The data to send in the POST request. Can be string or binary.",
    )

    async def process(self, context: ProcessingContext) -> bytes:
        res = await context.http_post(
            self.url, data=self.data, **self.get_request_kwargs()
        )
        return res.content


class DownloadDataframe(HTTPBaseNode):
    """
    Download tabular data from URL and return as structured DataframeRef.

    Fetches CSV, TSV, or JSON data and parses it into a dataframe with typed columns.
    Column definitions control structure, order, and type casting. Missing columns
    in source data are filled with None.

    Parameters:
    - url (required): URL of the data file
    - file_format (required, default="csv"): Format type (csv, tsv, or json)
    - columns (required): RecordType defining expected column names and data types
    - encoding (optional, default="utf-8"): Text encoding of the file
    - delimiter (optional, default=","): Column delimiter for CSV/TSV

    Returns: DataframeRef with columns and rows matching the definition

    Side effects: Network request

    Typical usage: Import datasets from web sources, fetch API data in tabular format,
    or download analytics exports. Precede with URL construction nodes. Follow with
    dataframe filtering, transformation, or visualization nodes. Empty columns definition
    returns empty dataframe.

    http, get, request, url, dataframe, csv, json, data
    """

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def get_title(cls):
        return "Download Dataframe"

    @classmethod
    def get_basic_fields(cls):
        return super().get_basic_fields() + ["columns", "file_format"]

    class FileFormat(str, Enum):
        CSV = "csv"
        JSON = "json"
        TSV = "tsv"

    file_format: FileFormat = Field(
        default=FileFormat.CSV,
        description="The format of the data file (csv, json, tsv).",
    )
    columns: RecordType = Field(
        default=RecordType(),
        description="The columns of the dataframe.",
    )
    encoding: str = Field(
        default="utf-8",
        description="The encoding of the text file.",
    )
    delimiter: str = Field(
        default=",",
        description="The delimiter for CSV/TSV files.",
    )

    @staticmethod
    def _cast_value(value: Any, target_type_str: str) -> Any:
        if value is None:
            return None

        # If the value is an empty string and the target is not string,
        # it often means 'missing' or 'null' in tabular data.
        # Casting "" to int/float will fail. Treat as None.
        if (
            isinstance(value, str)
            and value == ""
            and target_type_str not in ["string", "object"]
        ):
            return None

        try:
            if target_type_str == "int":
                return int(float(value))  # Use float(value) first to handle "1.0" -> 1
            elif target_type_str == "float":
                return float(value)
            elif target_type_str == "string":
                return str(value)
            elif target_type_str == "datetime":
                if isinstance(value, (datetime.datetime, datetime.date)):
                    return value
                if isinstance(
                    value, (int, float)
                ):  # handle timestamps (assume seconds since epoch)
                    try:
                        # OSError can be raised for out-of-range timestamps on some systems
                        return datetime.datetime.fromtimestamp(
                            value, tz=datetime.timezone.utc
                        )  # Assume UTC
                    except (ValueError, TypeError, OSError):
                        logger.warning(
                            "Could not parse numeric value '%s' as a timestamp for datetime. Returning original value.",
                            value,
                        )
                        return value

                if isinstance(value, str):
                    original_value_for_warning = value
                    try:
                        if value.endswith(
                            "Z"
                        ):  # Python's fromisoformat doesn't like 'Z' before 3.11
                            value = value[:-1] + "+00:00"
                        # Add more robust parsing or alternative formats if needed in the future
                        return datetime.datetime.fromisoformat(value)
                    except ValueError:
                        logger.warning(
                            "Could not parse datetime string '%s' to datetime object. Returning original string.",
                            original_value_for_warning,
                        )
                        return original_value_for_warning
                # If not a string, or already a datetime, or a convertible numeric, return as is or let it be handled by final catch-all
                return value
            elif target_type_str == "object":
                return value  # No casting needed
            else:  # Unknown target_type_str
                logger.warning(
                    "Unknown target data type '%s' for casting. Returning original value.",
                    target_type_str,
                )
                return value
        except (ValueError, TypeError) as e:
            # General catch for casting errors like int('abc') or float(None) if not caught earlier
            logger.warning(
                "Could not cast value '%s' (type: %s) to type '%s': %s. Returning None.",
                str(value)[:50],
                type(value).__name__,
                target_type_str,
                e,
            )
            return None

    async def process(self, context: ProcessingContext) -> DataframeRef:
        import csv
        import json
        import io
        from typing import Any  # For type hints in helper

        res = await context.http_get(self.url, **self.get_request_kwargs())
        content = res.content.decode(self.encoding)

        # If no columns are defined in self.columns.columns, return an empty dataframe.
        # This means the user expects a dataframe shaped by these definitions,
        # and an empty definition list results in an empty (no columns, no data) dataframe.
        if not self.columns.columns:
            return DataframeRef(columns=[], data=[])

        # Helper function to map parsed data to the defined columns
        def _map_data_to_defined_columns(
            parsed_cols_from_file: List[str],
            parsed_rows_from_file: List[List[Any]],
            target_column_definitions: List[ColumnDef],
        ) -> tuple[List[ColumnDef], List[List[Any]]]:

            file_col_name_to_idx_map: dict[str, int] = {
                name: i for i, name in enumerate(parsed_cols_from_file)
            }

            reordered_data_rows: List[List[Any]] = []

            for original_row in parsed_rows_from_file:
                new_row = [None] * len(target_column_definitions)
                for i, target_col_def in enumerate(target_column_definitions):
                    target_col_name = target_col_def.name
                    target_col_type = (
                        target_col_def.data_type
                    )  # Get the target data type

                    if target_col_name in file_col_name_to_idx_map:
                        original_col_idx = file_col_name_to_idx_map[target_col_name]
                        if original_col_idx < len(original_row):
                            raw_value = original_row[original_col_idx]
                            # Cast the value to the defined column type
                            casted_value = self._cast_value(raw_value, target_col_type)
                            new_row[i] = casted_value
                reordered_data_rows.append(new_row)

            return target_column_definitions, reordered_data_rows

        if self.file_format == self.FileFormat.CSV:
            reader = csv.reader(io.StringIO(content), delimiter=self.delimiter)
            data = list(reader)
            if data:  # Ensure there's at least a header row
                cols_from_file = data[0]
                rows_from_file = data[1:]
                mapped_definitions, mapped_rows = _map_data_to_defined_columns(
                    cols_from_file, rows_from_file, self.columns.columns
                )
                return DataframeRef(columns=mapped_definitions, data=mapped_rows)
            raise ValueError("No data found in CSV")

        elif self.file_format == self.FileFormat.TSV:
            reader = csv.reader(io.StringIO(content), delimiter="\\t")  # TSV delimiter
            data = list(reader)
            if data:  # Ensure there's at least a header row
                cols_from_file = data[0]
                rows_from_file = data[1:]
                mapped_definitions, mapped_rows = _map_data_to_defined_columns(
                    cols_from_file, rows_from_file, self.columns.columns
                )
                return DataframeRef(columns=mapped_definitions, data=mapped_rows)
            raise ValueError("No data found in TSV")

        elif self.file_format == self.FileFormat.JSON:
            json_data = json.loads(content)
            if isinstance(json_data, list) and json_data:
                cols_from_file: List[str] = []
                rows_from_file: List[List[Any]] = []

                if isinstance(json_data[0], dict):
                    # Assuming all dicts in the list have similar keys, use the first one for headers
                    cols_from_file = list(json_data[0].keys())
                    rows_from_file = [
                        [item.get(col, None) for col in cols_from_file]
                        for item in json_data
                    ]
                elif isinstance(json_data[0], list):
                    # Assuming the first list is the header row
                    cols_from_file = json_data[0]
                    rows_from_file = json_data[1:]
                else:
                    raise ValueError(
                        "JSON data is a list, but items are not dictionaries or lists."
                    )

                if (
                    not cols_from_file and not rows_from_file and not json_data[0]
                ):  # Handles cases like [[]] or [{}]
                    raise ValueError("JSON data parsed to empty columns and rows.")

                mapped_definitions, mapped_rows = _map_data_to_defined_columns(
                    cols_from_file, rows_from_file, self.columns.columns
                )
                return DataframeRef(columns=mapped_definitions, data=mapped_rows)
            raise ValueError("No data found or data is not a list of records in JSON")

        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")


class FilterValidURLs(HTTPBaseNode):
    """
    Filter URL list to only valid URLs by checking HTTP status with HEAD requests.

    Performs concurrent HEAD requests to each URL and returns only those with
    successful status codes (200-399). Invalid or unreachable URLs are filtered out.

    Parameters:
    - urls (required): List of URLs to validate
    - max_concurrent_requests (optional, default=10): Parallel request limit

    Returns: List of valid URLs (strings)

    Side effects: Multiple concurrent network requests

    Typical usage: Clean URL lists from web scraping, verify resource availability
    before downloading, or validate links before processing. Precede with URL extraction
    or list generation nodes. Follow with download nodes or URL processing workflows.

    url validation, http, head request
    """

    @classmethod
    def get_title(cls):
        return "Filter Valid URLs"

    urls: list[str] = Field(
        default=[],
        description="List of URLs to validate.",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent HEAD requests.",
    )

    async def check_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> tuple[str, bool]:
        try:
            async with session.head(
                url,
                allow_redirects=True,
                **self.get_request_kwargs(),
            ) as response:
                is_valid = 200 <= response.status < 400
                return url, is_valid
        except Exception:
            return url, False

    async def process(self, context: ProcessingContext) -> list[str]:
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in self.urls:
                task = self.check_url(session, url)
                tasks.append(task)

                if len(tasks) >= self.max_concurrent_requests:
                    completed = await asyncio.gather(*tasks)
                    results.extend(completed)
                    tasks = []

            if tasks:
                completed = await asyncio.gather(*tasks)
                results.extend(completed)

        valid_urls = [url for url, is_valid in results if is_valid]

        return valid_urls


class DownloadFiles(BaseNode):
    """
    Download multiple files concurrently from URLs and save to local folder.

    Downloads files in parallel with configurable concurrency and emits progress updates.
    Creates the output folder if it doesn't exist. Filenames are extracted from URLs
    or Content-Disposition headers.

    Parameters:
    - urls (required): List of file URLs to download
    - output_folder (required, default="downloads"): Local directory path for saved files
    - max_concurrent_downloads (optional, default=5): Parallel download limit

    Returns: Dictionary with "success" (list of saved file paths) and "failed"
    (list of URLs that failed to download)

    Side effects: Multiple concurrent network requests, local file system writes,
    directory creation, progress messages emitted during processing

    Typical usage: Batch download files, archive web content, or collect remote
    resources. Precede with URL collection or extraction nodes. Follow with file
    processing nodes or directory scanning workflows.

    download, files, urls, batch
    """

    urls: list[str] = Field(
        default=[],
        description="List of URLs to download.",
    )
    output_folder: str = Field(
        default="downloads",
        description="Local folder path where files will be saved.",
    )

    _expose_as_tool: ClassVar[bool] = True

    @classmethod
    def is_visible(cls) -> bool:
        return cls is not HTTPBaseNode

    def get_request_kwargs(self) -> dict[str, Any]:
        return {}

    max_concurrent_downloads: int = Field(
        default=5,
        description="Maximum number of concurrent downloads.",
    )

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> str:
        try:
            async with session.get(url, **self.get_request_kwargs()) as response:
                if response.status == 200:
                    # Extract filename from URL or Content-Disposition header
                    filename = response.headers.get("Content-Disposition")
                    if filename and "filename=" in filename:
                        filename = filename.split("filename=")[-1].strip("\"'")
                    else:
                        filename = url.split("/")[-1]
                        if not filename:
                            filename = "unnamed_file"

                    if not self.output_folder:
                        raise ValueError("output_folder cannot be empty")
                    expanded_folder = os.path.expanduser(self.output_folder)
                    os.makedirs(expanded_folder, exist_ok=True)

                    filepath = os.path.join(expanded_folder, filename)
                    content = await response.read()

                    with open(filepath, "wb") as f:
                        f.write(content)

                    return filepath
                else:
                    return ""
        except Exception:
            return ""

    class OutputType(TypedDict):
        success: list[str]
        failed: list[str]

    async def process(self, context: ProcessingContext) -> OutputType:
        successful = []
        failed = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            num_completed = 0
            for url in self.urls:
                task = self.download_file(session, url)
                tasks.append(task)

                if len(tasks) >= self.max_concurrent_downloads:
                    completed = await asyncio.gather(*tasks)
                    num_completed += len(completed)
                    context.post_message(
                        NodeProgress(
                            node_id=self.id,
                            progress=num_completed,
                            total=len(self.urls),
                        )
                    )
                    for filepath in completed:
                        if filepath:
                            successful.append(filepath)
                        else:
                            failed.append(url)
                    tasks = []

            if tasks:
                completed = await asyncio.gather(*tasks)
                num_completed += len(completed)
                context.post_message(
                    NodeProgress(
                        node_id=self.id,
                        progress=num_completed,
                        total=len(self.urls),
                    )
                )
                for filepath in completed:
                    if filepath:
                        successful.append(filepath)
                    else:
                        failed.append(url)

        return {
            "success": successful,
            "failed": failed,
        }


class JSONPostRequest(HTTPBaseNode):
    """
    Send JSON dictionary via HTTP POST and return parsed JSON response.

    Serializes the data dictionary to JSON, sends with Content-Type: application/json
    header, and parses the response JSON automatically.

    Parameters:
    - url (required): Target API endpoint URL
    - data (optional, default={}): Dictionary to serialize and send as JSON

    Returns: Parsed JSON response as dictionary

    Side effects: Network request that may create or modify resources

    Typical usage: Create API resources, submit structured data to REST endpoints, or
    interact with JSON APIs. Precede with dictionary construction or transformation nodes.
    Follow with JSON path extraction or response validation nodes.

    http, post, request, url, json, api
    """

    @classmethod
    def get_title(cls):
        return "POST JSON"

    data: dict = Field(
        default={},
        description="The JSON data to send in the POST request.",
    )

    async def process(self, context: ProcessingContext) -> dict:
        res = await context.http_post(
            self.url,
            json=self.data,
            headers={
                "Content-Type": "application/json",
            },
        )
        return res.json()


class JSONPutRequest(HTTPBaseNode):
    """
    Replace resource with JSON dictionary via HTTP PUT and return parsed JSON response.

    Serializes the data dictionary to JSON, sends with Content-Type: application/json
    header for full resource replacement, and parses the response JSON automatically.

    Parameters:
    - url (required): Target resource URL
    - data (optional, default={}): Dictionary to serialize and send as JSON

    Returns: Parsed JSON response as dictionary

    Side effects: Network request that replaces or creates server-side resources

    Typical usage: Update complete API resources, replace configuration, or modify
    server state with structured data. For partial updates use JSONPatchRequest.
    Follow with response validation or confirmation logic.

    http, put, request, url, json, api
    """

    @classmethod
    def get_title(cls):
        return "PUT JSON"

    data: dict = Field(
        default={},
        description="The JSON data to send in the PUT request.",
    )

    async def process(self, context: ProcessingContext) -> dict:
        headers = {"Content-Type": "application/json"}
        res = await context.http_put(
            self.url,
            json=self.data,
            headers=headers,
        )
        return res.json()


class JSONPatchRequest(HTTPBaseNode):
    """
    Partially update resource with JSON dictionary via HTTP PATCH and return parsed JSON response.

    Serializes the data dictionary to JSON, sends with Content-Type: application/json
    header for partial resource modification, and parses the response JSON automatically.
    More efficient than PUT for updating specific fields.

    Parameters:
    - url (required): Target resource URL
    - data (optional, default={}): Dictionary with fields to update

    Returns: Parsed JSON response as dictionary

    Side effects: Network request that modifies server-side resources

    Typical usage: Update specific fields of API resources without replacing entire
    objects, modify configuration values, or make efficient updates to large resources.
    For full replacement use JSONPutRequest. Follow with response validation nodes.

    http, patch, request, url, json, api
    """

    @classmethod
    def get_title(cls):
        return "PATCH JSON"

    data: dict = Field(
        default={},
        description="The JSON data to send in the PATCH request.",
    )

    async def process(self, context: ProcessingContext) -> dict:
        headers = {"Content-Type": "application/json"}
        res = await context.http_patch(
            self.url,
            json=self.data,
            headers=headers,
        )
        return res.json()


class JSONGetRequest(HTTPBaseNode):
    """
    Perform HTTP GET request and return parsed JSON response as dictionary.

    Sends request with Accept: application/json header and automatically parses
    the JSON response body. Raises error if response is not valid JSON.

    Parameters:
    - url (required): Target API endpoint URL

    Returns: Parsed JSON response as dictionary

    Side effects: Network request

    Typical usage: Fetch data from REST APIs, retrieve JSON configuration, or access
    JSON web services. Follow with JSON path extraction, validation, or transformation
    nodes to process the structured response.

    http, get, request, url, json, api
    """

    @classmethod
    def get_title(cls):
        return "GET JSON"

    async def process(self, context: ProcessingContext) -> dict:
        headers = {"Accept": "application/json"}
        res = await context.http_get(
            self.url,
            headers=headers,
        )
        return res.json()
