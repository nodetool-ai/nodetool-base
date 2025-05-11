from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Browser(GraphNode):
    """
    Fetches content from a web page using a headless browser.
    browser, web, scraping, content, fetch

    Use cases:
    - Extract content from JavaScript-heavy websites
    - Retrieve text content from web pages
    - Get metadata from web pages
    - Save extracted content to files
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL to navigate to"
    )
    timeout: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20000, description="Timeout in milliseconds for page navigation"
    )
    use_readability: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Use Python's Readability for better content extraction",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.Browser"


import nodetool.nodes.nodetool.browser
import nodetool.nodes.nodetool.browser


class BrowserNavigation(GraphNode):
    """
    Navigates and interacts with web pages in a browser session.
    browser, navigation, interaction, click, extract

    Use cases:
    - Perform complex web interactions
    - Navigate through multi-step web processes
    - Extract content after interaction
    """

    Action: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.browser.BrowserNavigation.Action
    )
    ExtractType: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.browser.BrowserNavigation.ExtractType
    )
    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL to navigate to (required for 'goto' action)"
    )
    action: nodetool.nodes.nodetool.browser.BrowserNavigation.Action = Field(
        default=nodetool.nodes.nodetool.browser.BrowserNavigation.Action.GOTO,
        description="Navigation or extraction action to perform",
    )
    selector: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="CSS selector for the element to interact with or extract from",
    )
    timeout: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30000, description="Timeout in milliseconds for the action"
    )
    wait_for: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Optional selector to wait for after performing the action",
    )
    extract_type: nodetool.nodes.nodetool.browser.BrowserNavigation.ExtractType = Field(
        default=nodetool.nodes.nodetool.browser.BrowserNavigation.ExtractType.TEXT,
        description="Type of content to extract (for 'extract' action)",
    )
    attribute: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Attribute name to extract (when extract_type is 'attribute')",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.BrowserNavigation"


import nodetool.nodes.nodetool.browser


class BrowserUseNode(GraphNode):
    """
    Browser agent tool that uses browser_use under the hood.

    This module provides a tool for running browser-based agents using the browser_use library.
    The agent can perform complex web automation tasks like form filling, navigation, data extraction,
    and multi-step workflows using natural language instructions.

    Use cases:
    - Perform complex web automation tasks based on natural language.
    - Automate form filling and data entry.
    - Scrape data after complex navigation or interaction sequences.
    - Automate multi-step web workflows.
    """

    BrowserUseModel: typing.ClassVar[type] = (
        nodetool.nodes.nodetool.browser.BrowserUseModel
    )
    model: nodetool.nodes.nodetool.browser.BrowserUseModel = Field(
        default=nodetool.nodes.nodetool.browser.BrowserUseModel.GPT_4O,
        description="The model to use for the browser agent.",
    )
    task: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="Natural language description of the browser task to perform. Can include complex multi-step instructions like 'Compare prices between websites', 'Fill out forms', or 'Extract specific data'.",
    )
    timeout: int | GraphNode | tuple[GraphNode, str] = Field(
        default=300,
        description="Maximum time in seconds to allow for task completion. Complex tasks may require longer timeouts.",
    )
    use_remote_browser: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Use a remote browser instead of a local one"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.BrowserUse"


class DownloadFile(GraphNode):
    """
    Downloads a file from a URL and saves it to disk.
    download, file, web, save

    Use cases:
    - Download documents, images, or other files from the web
    - Save data for further processing
    - Retrieve file assets for analysis
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL of the file to download"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.DownloadFile"


class Screenshot(GraphNode):
    """
    Takes a screenshot of a web page or specific element.
    browser, screenshot, capture, image

    Use cases:
    - Capture visual representation of web pages
    - Document specific UI elements
    - Create visual records of web content
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL to navigate to before taking screenshot"
    )
    selector: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Optional CSS selector for capturing a specific element"
    )
    output_file: types.FilePath | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FilePath(type="file_path", path="screenshot.png"),
        description="Path to save the screenshot (relative to workspace)",
    )
    timeout: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30000, description="Timeout in milliseconds for page navigation"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.Screenshot"


class WebFetch(GraphNode):
    """
    Fetches HTML content from a URL and converts it to text.
    web, fetch, html, markdown, http

    Use cases:
    - Extract text content from web pages
    - Process web content for analysis
    - Save web content to files
    """

    url: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="URL to fetch content from"
    )
    selector: str | GraphNode | tuple[GraphNode, str] = Field(
        default="body", description="CSS selector to extract specific elements"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.browser.WebFetch"
