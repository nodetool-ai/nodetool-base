from pydantic import Field
import typing
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode
import nodetool.nodes.lib.http


class DeleteRequest(GraphNode):
    """
    Remove a resource from a server using an HTTP DELETE request.
    http, delete, request, url

    Use cases:
    - Delete user accounts
    - Remove API resources
    - Cancel subscriptions
    - Clear cache entries
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.DeleteRequest"


class DownloadDataframe(GraphNode):
    """
    Download data from a URL and return as a dataframe.
    http, get, request, url, dataframe, csv, json, data

    Use cases:
    - Download CSV data and convert to dataframe
    - Fetch JSON data and convert to dataframe
    - Retrieve tabular data from APIs
    - Process data files from URLs
    """

    FileFormat: typing.ClassVar[type] = (
        nodetool.nodes.lib.http.DownloadDataframe.FileFormat
    )
    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    file_format: nodetool.nodes.lib.http.DownloadDataframe.FileFormat = Field(
        default=nodetool.nodes.lib.http.DownloadDataframe.FileFormat.CSV,
        description="The format of the data file (csv, json, tsv).",
    )
    columns: types.RecordType | GraphNode | tuple[GraphNode, str] = Field(
        default=types.RecordType(type="record_type", columns=[]),
        description="The columns of the dataframe.",
    )
    encoding: str | GraphNode | tuple[GraphNode, str] = Field(
        default="utf-8", description="The encoding of the text file."
    )
    delimiter: str | GraphNode | tuple[GraphNode, str] = Field(
        default=",", description="The delimiter for CSV/TSV files."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.DownloadDataframe"


class DownloadFiles(GraphNode):
    """
    Download files from a list of URLs into a local folder.
    download, files, urls, batch

    Use cases:
    - Batch download files from multiple URLs
    - Create local copies of remote resources
    - Archive web content
    - Download datasets
    """

    urls: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of URLs to download."
    )
    output_folder: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path="downloads"),
        description="Local folder path where files will be saved.",
    )
    max_concurrent_downloads: int | GraphNode | tuple[GraphNode, str] = Field(
        default=5, description="Maximum number of concurrent downloads."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.DownloadFiles"


class FetchPage(GraphNode):
    """
    Fetch a web page using Selenium and return its content.
    selenium, fetch, webpage, http

    Use cases:
    - Retrieve content from dynamic websites
    - Capture JavaScript-rendered content
    - Interact with web applications
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to fetch the page from."
    )
    wait_time: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum time to wait for page load (in seconds)."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.FetchPage"


class FilterValidURLs(GraphNode):
    """
    Filter a list of URLs by checking their validity using HEAD requests.
    url validation, http, head request

    Use cases:
    - Clean URL lists by removing broken links
    - Verify resource availability
    - Validate website URLs before processing
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    urls: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of URLs to validate."
    )
    max_concurrent_requests: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of concurrent HEAD requests."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.FilterValidURLs"


class GetRequest(GraphNode):
    """
    Perform an HTTP GET request to retrieve data from a specified URL.
    http, get, request, url

    Use cases:
    - Fetch web page content
    - Retrieve API data
    - Download files
    - Check website availability
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.GetRequest"


class GetRequestBinary(GraphNode):
    """
    Perform an HTTP GET request and return raw binary data.
    http, get, request, url, binary, download

    Use cases:
    - Download binary files
    - Fetch images or media
    - Retrieve PDF documents
    - Download any non-text content
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.GetRequestBinary"


class GetRequestDocument(GraphNode):
    """
    Perform an HTTP GET request and return a document
    http, get, request, url, document

    Use cases:
    - Download PDF documents
    - Retrieve Word documents
    - Fetch Excel files
    - Download any document format
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.GetRequestDocument"


class HeadRequest(GraphNode):
    """
    Retrieve headers from a resource using an HTTP HEAD request.
    http, head, request, url

    Use cases:
    - Check resource existence
    - Get metadata without downloading content
    - Verify authentication or permissions
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.HeadRequest"


class ImageDownloader(GraphNode):
    """
    Download images from list of URLs and return a list of ImageRefs.
    image download, web scraping, data processing

    Use cases:
    - Prepare image datasets for machine learning tasks
    - Archive images from web pages
    - Process and analyze images extracted from websites
    """

    images: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of image URLs to download."
    )
    base_url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Base URL to prepend to relative image URLs."
    )
    max_concurrent_downloads: int | GraphNode | tuple[GraphNode, str] = Field(
        default=10, description="Maximum number of concurrent image downloads."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.ImageDownloader"


class JSONGetRequest(GraphNode):
    """
    Perform an HTTP GET request and parse the response as JSON.
    http, get, request, url, json, api

    Use cases:
    - Fetch data from REST APIs
    - Retrieve JSON-formatted responses
    - Interface with JSON web services
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.JSONGetRequest"


class JSONPatchRequest(GraphNode):
    """
    Partially update resources with JSON data using an HTTP PATCH request.
    http, patch, request, url, json, api

    Use cases:
    - Partial updates to API resources
    - Modify specific fields without full replacement
    - Efficient updates for large objects
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The JSON data to send in the PATCH request."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.JSONPatchRequest"


class JSONPostRequest(GraphNode):
    """
    Send JSON data to a server using an HTTP POST request.
    http, post, request, url, json, api

    Use cases:
    - Send structured data to REST APIs
    - Create resources with JSON payloads
    - Interface with modern web services
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The JSON data to send in the POST request."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.JSONPostRequest"


class JSONPutRequest(GraphNode):
    """
    Update resources with JSON data using an HTTP PUT request.
    http, put, request, url, json, api

    Use cases:
    - Update existing API resources
    - Replace complete objects in REST APIs
    - Set configuration with JSON data
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: dict | GraphNode | tuple[GraphNode, str] = Field(
        default={}, description="The JSON data to send in the PUT request."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.JSONPutRequest"


class PostRequest(GraphNode):
    """
    Send data to a server using an HTTP POST request.
    http, post, request, url, data

    Use cases:
    - Submit form data
    - Create new resources on an API
    - Upload files
    - Authenticate users
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The data to send in the POST request."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.PostRequest"


class PostRequestBinary(GraphNode):
    """
    Send data using an HTTP POST request and return raw binary data.
    http, post, request, url, data, binary

    Use cases:
    - Upload and receive binary files
    - Interact with binary APIs
    - Process image or media uploads
    - Handle binary file transformations
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: str | bytes | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The data to send in the POST request. Can be string or binary.",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.PostRequestBinary"


class PutRequest(GraphNode):
    """
    Update existing resources on a server using an HTTP PUT request.
    http, put, request, url, data

    Use cases:
    - Update user profiles
    - Modify existing API resources
    - Replace file contents
    - Set configuration values
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The URL to make the request to."
    )
    data: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The data to send in the PUT request."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.http.PutRequest"
