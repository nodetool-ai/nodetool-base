import asyncio
import datetime
import os
from enum import Enum
from typing import Any, List
from urllib.parse import urljoin

import aiohttp
from pydantic import Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from nodetool.metadata.types import (
    ColumnDef,
    DataframeRef,
    DocumentRef,
    FilePath,
    ImageRef,
    RecordType,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress


class HTTPBaseNode(BaseNode):
    """Base node for HTTP requests.

    http, network, request

    Use cases:
    - Share common fields for HTTP nodes
    - Add custom request parameters in subclasses
    - Control visibility of specific request types
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
    Perform an HTTP GET request to retrieve data from a specified URL.
    http, get, request, url

    Use cases:
    - Fetch web page content
    - Retrieve API data
    - Download files
    - Check website availability
    """

    @classmethod
    def get_title(cls):
        return "GET Request"

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return res.content.decode(res.encoding or "utf-8")


class PostRequest(HTTPBaseNode):
    """
    Send data to a server using an HTTP POST request.
    http, post, request, url, data

    Use cases:
    - Submit form data
    - Create new resources on an API
    - Upload files
    - Authenticate users
    """

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
    Update existing resources on a server using an HTTP PUT request.
    http, put, request, url, data

    Use cases:
    - Update user profiles
    - Modify existing API resources
    - Replace file contents
    - Set configuration values
    """

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
    Remove a resource from a server using an HTTP DELETE request.
    http, delete, request, url

    Use cases:
    - Delete user accounts
    - Remove API resources
    - Cancel subscriptions
    - Clear cache entries
    """

    @classmethod
    def get_title(cls):
        return "DELETE Request"

    async def process(self, context: ProcessingContext) -> str:
        res = await context.http_delete(self.url, **self.get_request_kwargs())
        return res.content.decode(res.encoding or "utf-8")


class HeadRequest(HTTPBaseNode):
    """
    Retrieve headers from a resource using an HTTP HEAD request.
    http, head, request, url

    Use cases:
    - Check resource existence
    - Get metadata without downloading content
    - Verify authentication or permissions
    """

    @classmethod
    def get_title(cls):
        return "HEAD Request"

    async def process(self, context: ProcessingContext) -> dict[str, str]:
        res = await context.http_head(self.url, **self.get_request_kwargs())
        return dict(res.headers.items())


class FetchPage(BaseNode):
    """
    Fetch a web page using Selenium and return its content.
    selenium, fetch, webpage, http

    Use cases:
    - Retrieve content from dynamic websites
    - Capture JavaScript-rendered content
    - Interact with web applications
    """

    url: str = Field(
        default="",
        description="The URL to fetch the page from.",
    )
    wait_time: int = Field(
        default=10,
        description="Maximum time to wait for page load (in seconds).",
    )

    @classmethod
    def return_type(cls):
        return {
            "html": str,
            "success": bool,
            "error_message": str | None,
        }

    async def process(self, context: ProcessingContext):
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
    Download images from list of URLs and return a list of ImageRefs.
    image download, web scraping, data processing

    Use cases:
    - Prepare image datasets for machine learning tasks
    - Archive images from web pages
    - Process and analyze images extracted from websites
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

    @classmethod
    def return_type(cls):
        return {
            "images": list[ImageRef],
            "failed_urls": list[str],
        }

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
                    print(error_msg)
                    return None, url
        except Exception as e:
            error_msg = f"Error downloading image from {url}: {str(e)}"
            print(error_msg)
            return None, url

    async def process(self, context: ProcessingContext):
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
    Perform an HTTP GET request and return raw binary data.
    http, get, request, url, binary, download

    Use cases:
    - Download binary files
    - Fetch images or media
    - Retrieve PDF documents
    - Download any non-text content
    """

    @classmethod
    def get_title(cls):
        return "GET Binary"

    async def process(self, context: ProcessingContext) -> bytes:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return res.content


class GetRequestDocument(HTTPBaseNode):
    """
    Perform an HTTP GET request and return a document
    http, get, request, url, document

    Use cases:
    - Download PDF documents
    - Retrieve Word documents
    - Fetch Excel files
    - Download any document format
    """

    @classmethod
    def get_title(cls):
        return "GET Document"

    async def process(self, context: ProcessingContext) -> DocumentRef:
        res = await context.http_get(self.url, **self.get_request_kwargs())
        return DocumentRef(data=res.content)


class PostRequestBinary(HTTPBaseNode):
    """
    Send data using an HTTP POST request and return raw binary data.
    http, post, request, url, data, binary

    Use cases:
    - Upload and receive binary files
    - Interact with binary APIs
    - Process image or media uploads
    - Handle binary file transformations
    """

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
    Download data from a URL and return as a dataframe.
    http, get, request, url, dataframe, csv, json, data

    Use cases:
    - Download CSV data and convert to dataframe
    - Fetch JSON data and convert to dataframe
    - Retrieve tabular data from APIs
    - Process data files from URLs
    """

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
                        print(
                            f"Warning: Could not parse numeric value '{value}' as a timestamp for datetime. Returning original value."
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
                        print(
                            f"Warning: Could not parse datetime string '{original_value_for_warning}' to datetime object. Returning original string."
                        )
                        return original_value_for_warning
                # If not a string, or already a datetime, or a convertible numeric, return as is or let it be handled by final catch-all
                return value
            elif target_type_str == "object":
                return value  # No casting needed
            else:  # Unknown target_type_str
                print(
                    f"Warning: Unknown target data type '{target_type_str}' for casting. Returning original value."
                )
                return value
        except (ValueError, TypeError) as e:
            # General catch for casting errors like int('abc') or float(None) if not caught earlier
            print(
                f"Warning: Could not cast value '{str(value)[:50]}' (type: {type(value).__name__}) to type '{target_type_str}': {e}. Returning None."
            )
            return None

    async def process(self, context: ProcessingContext) -> DataframeRef:
        import csv
        import json
        import io
        from nodetool.metadata.types import (
            ColumnDef,
        )  # Ensure ColumnDef is in scope for the helper
        from typing import List, Any  # For type hints in helper

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
    Filter a list of URLs by checking their validity using HEAD requests.
    url validation, http, head request

    Use cases:
    - Clean URL lists by removing broken links
    - Verify resource availability
    - Validate website URLs before processing
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
    Download files from a list of URLs into a local folder.
    download, files, urls, batch

    Use cases:
    - Batch download files from multiple URLs
    - Create local copies of remote resources
    - Archive web content
    - Download datasets
    """

    urls: list[str] = Field(
        default=[],
        description="List of URLs to download.",
    )
    output_folder: FilePath = Field(
        default=FilePath(path="downloads"),
        description="Local folder path where files will be saved.",
    )

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

                    expanded_path = os.path.expanduser(self.output_folder.path)
                    os.makedirs(os.path.dirname(expanded_path), exist_ok=True)

                    filepath = os.path.join(expanded_path, filename)
                    content = await response.read()

                    with open(filepath, "wb") as f:
                        f.write(content)

                    return filepath
                else:
                    return ""
        except Exception as e:
            return ""

    @classmethod
    def return_type(cls):
        return {
            "success": list[str],
            "failed": list[str],
        }

    async def process(self, context: ProcessingContext):
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
            "successful": successful,
            "failed": failed,
        }


class JSONPostRequest(HTTPBaseNode):
    """
    Send JSON data to a server using an HTTP POST request.
    http, post, request, url, json, api

    Use cases:
    - Send structured data to REST APIs
    - Create resources with JSON payloads
    - Interface with modern web services
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
    Update resources with JSON data using an HTTP PUT request.
    http, put, request, url, json, api

    Use cases:
    - Update existing API resources
    - Replace complete objects in REST APIs
    - Set configuration with JSON data
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
    Partially update resources with JSON data using an HTTP PATCH request.
    http, patch, request, url, json, api

    Use cases:
    - Partial updates to API resources
    - Modify specific fields without full replacement
    - Efficient updates for large objects
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
    Perform an HTTP GET request and parse the response as JSON.
    http, get, request, url, json, api

    Use cases:
    - Fetch data from REST APIs
    - Retrieve JSON-formatted responses
    - Interface with JSON web services
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
