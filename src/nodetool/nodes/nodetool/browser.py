import traceback
import aiohttp
import html2text
import logging
from bs4 import BeautifulSoup
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import Field
from playwright.async_api import async_playwright
from readability import Document
import asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import FilePath
import os

logger = logging.getLogger(__name__)

# Browser Use
os.environ["ANONYMIZED_TELEMETRY"] = "false"


class Browser(BaseNode):
    """
    Fetches content from a web page using a headless browser.
    browser, web, scraping, content, fetch

    Use cases:
    - Extract content from JavaScript-heavy websites
    - Retrieve text content from web pages
    - Get metadata from web pages
    - Save extracted content to files
    """

    url: str = Field(default="", description="URL to navigate to")

    timeout: int = Field(
        default=20000, description="Timeout in milliseconds for page navigation"
    )

    use_readability: bool = Field(
        default=True,
        description="Use Python's Readability for better content extraction",
    )

    @classmethod
    def return_type(cls):
        return {
            "success": bool,
            "content": str,
            "metadata": Dict[str, Any],
        }

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.url:
            raise ValueError("URL is required")

        # Initialize browser
        playwright_instance = await async_playwright().start()
        browser_endpoint = Environment.get("BROWSER_URL")

        if browser_endpoint:
            browser = await playwright_instance.chromium.connect_over_cdp(
                browser_endpoint
            )
            # Create context with additional permissions and settings
            browser_context = await browser.new_context(
                bypass_csp=True,
            )
        else:
            # Launch browser with similar settings for local usage
            browser = await playwright_instance.chromium.launch(headless=True)
            browser_context = await browser.new_context(
                bypass_csp=True,
            )

        # Create page from the context instead of directly from browser
        page = await browser_context.new_page()

        try:
            # Navigate to the URL with more complete loading strategy
            await page.goto(self.url, wait_until="networkidle", timeout=self.timeout)

            # Extract metadata from the page
            metadata = await self._extract_metadata(page)

            result = {
                "success": True,
                "metadata": metadata,
            }

            # Extract content using Readability or plain HTML
            if self.use_readability:
                try:
                    logger.debug("Using Python Readability for URL: %s", self.url)

                    # Get the HTML content from the page
                    html_content = await page.content()
                    logger.debug("HTML content: %s", html_content)

                    doc = Document(html_content)

                    # Get the article content
                    article_content = doc.summary()

                    # Convert to plain text
                    h = html2text.HTML2Text(baseurl=self.url, bodywidth=1000)
                    h.ignore_images = True
                    h.ignore_mailto_links = True
                    content = h.handle(article_content)

                except Exception as e:
                    logger.warning("Exception using Python Readability: %s", e)
                    # Fallback to using regular HTML content on exception
                    content = html2text.html2text(await page.content())
            else:
                content = html2text.html2text(await page.content())

            result["content"] = content

            return result
        except Exception as e:
            logger.error("Error fetching page: %s", e)
            traceback.print_exc()
            raise ValueError(f"Error fetching page: {str(e)}")

        finally:
            # Always close the browser session
            await browser.close()
            await playwright_instance.stop()

    async def _extract_metadata(self, page):
        """
        Extract both Open Graph and standard metadata from a webpage using Playwright.
        """
        # Create a dictionary to store the metadata
        metadata = {
            "og": {},  # For Open Graph metadata
            "standard": {},  # For standard metadata
        }

        # List of Open Graph properties to extract
        og_properties = [
            "og:locale",
            "og:type",
            "og:title",
            "og:description",
            "og:url",
            "og:site_name",
            "og:image",
            "og:image:width",
            "og:image:height",
            "og:image:type",
        ]

        # List of standard meta properties to extract
        standard_properties = [
            "description",
            "keywords",
            "author",
            "viewport",
            "robots",
            "canonical",
            "generator",
        ]

        # Extract Open Graph metadata
        for prop in og_properties:
            # Use locator to find the meta tag with the specific property
            locator = page.locator(f'meta[property="{prop}"]')

            # Check if the element exists
            if await locator.count() > 0:
                # Extract the content attribute
                content = await locator.first.get_attribute("content")
                # Store in dictionary (remove 'og:' prefix for cleaner keys)
                metadata["og"][prop.replace("og:", "")] = content

        # Extract standard metadata
        for prop in standard_properties:
            # Use locator to find the meta tag with the specific name
            locator = page.locator(f'meta[name="{prop}"]')

            # Check if the element exists
            if await locator.count() > 0:
                # Extract the content attribute
                content = await locator.first.get_attribute("content")
                # Store in dictionary
                metadata["standard"][prop] = content

        # Also get title from the title tag
        title_locator = page.locator("title")
        if await title_locator.count() > 0:
            metadata["standard"]["title"] = await title_locator.first.inner_text()

        return metadata


class Screenshot(BaseNode):
    """
    Takes a screenshot of a web page or specific element.
    browser, screenshot, capture, image

    Use cases:
    - Capture visual representation of web pages
    - Document specific UI elements
    - Create visual records of web content
    """

    url: str = Field(
        default="", description="URL to navigate to before taking screenshot"
    )

    selector: str = Field(
        default="", description="Optional CSS selector for capturing a specific element"
    )

    output_file: FilePath = Field(
        default=FilePath(path="screenshot.png"),
        description="Path to save the screenshot (relative to workspace)",
    )

    timeout: int = Field(
        default=30000, description="Timeout in milliseconds for page navigation"
    )

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise Exception(
                "Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"
            )

        if not self.url:
            raise ValueError("URL is required")

        # Initialize browser
        playwright_instance = await async_playwright().start()
        browser_endpoint = Environment.get("BROWSER_URL")

        if browser_endpoint:
            browser = await playwright_instance.chromium.connect_over_cdp(
                browser_endpoint
            )
            browser_context = await browser.new_context(bypass_csp=True)
        else:
            browser = await playwright_instance.chromium.launch(headless=True)
            browser_context = await browser.new_context(bypass_csp=True)

        page = await browser_context.new_page()

        try:
            # Navigate to the URL
            await page.goto(
                self.url, wait_until="domcontentloaded", timeout=self.timeout
            )

            # Prepare the output path
            full_path = context.resolve_workspace_path(self.output_file.path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Take screenshot of specific element or full page
            if self.selector:
                element = await page.query_selector(self.selector)
                if element:
                    await element.screenshot(path=full_path)
                else:
                    raise ValueError(
                        f"No element found matching selector: {self.selector}"
                    )
            else:
                await page.screenshot(path=full_path)

            return {"success": True, "path": full_path, "url": self.url}

        except Exception as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")

        finally:
            await browser.close()
            await playwright_instance.stop()


class WebFetch(BaseNode):
    """
    Fetches HTML content from a URL and converts it to text.
    web, fetch, html, markdown, http

    Use cases:
    - Extract text content from web pages
    - Process web content for analysis
    - Save web content to files
    """

    url: str = Field(default="", description="URL to fetch content from")

    selector: str = Field(
        default="body", description="CSS selector to extract specific elements"
    )

    async def process(self, context: ProcessingContext) -> str:
        if not self.url:
            raise ValueError("URL is required")

        try:
            # Make HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"HTTP request failed with status {response.status}"
                        )

                    # Check content type
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                    ):
                        return await response.text()

                    html_content = await response.text()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract content based on selector
            if self.selector:
                elements = soup.select(self.selector)
                if not elements:
                    raise ValueError(
                        f"No elements found matching selector: {self.selector}"
                    )

                # Get HTML content of all matching elements
                extracted_html = "".join(str(element) for element in elements)
            else:
                # Default to body if no selector provided
                body = soup.body
                if body:
                    extracted_html = str(body)
                else:
                    raise ValueError("No body element found in the HTML")

            # Convert to Markdown
            content = html2text.html2text(extracted_html)

            return content

        except aiohttp.ClientError as e:
            raise ValueError(f"HTTP request error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error fetching and processing content: {str(e)}")


class DownloadFile(BaseNode):
    """
    Downloads a file from a URL and saves it to disk.
    download, file, web, save

    Use cases:
    - Download documents, images, or other files from the web
    - Save data for further processing
    - Retrieve file assets for analysis
    """

    url: str = Field(default="", description="URL of the file to download")

    async def process(self, context: ProcessingContext) -> bytes:
        if not self.url:
            raise ValueError("URL is required")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"HTTP request failed with status {response.status}"
                        )

                    return await response.read()

        except Exception as e:
            raise ValueError(f"Error in download process: {str(e)}")


class BrowserNavigation(BaseNode):
    """
    Navigates and interacts with web pages in a browser session.
    browser, navigation, interaction, click, extract

    Use cases:
    - Perform complex web interactions
    - Navigate through multi-step web processes
    - Extract content after interaction
    """

    class Action(str, Enum):
        CLICK = "click"
        GOTO = "goto"
        BACK = "back"
        FORWARD = "forward"
        RELOAD = "reload"
        EXTRACT = "extract"

    class ExtractType(str, Enum):
        TEXT = "text"
        HTML = "html"
        VALUE = "value"
        ATTRIBUTE = "attribute"

    url: str = Field(
        default="", description="URL to navigate to (required for 'goto' action)"
    )

    action: Action = Field(
        default=Action.GOTO, description="Navigation or extraction action to perform"
    )

    selector: str = Field(
        default="",
        description="CSS selector for the element to interact with or extract from",
    )

    timeout: int = Field(
        default=30000, description="Timeout in milliseconds for the action"
    )

    wait_for: str = Field(
        default="",
        description="Optional selector to wait for after performing the action",
    )

    extract_type: ExtractType = Field(
        default=ExtractType.TEXT,
        description="Type of content to extract (for 'extract' action)",
    )

    attribute: str = Field(
        default="",
        description="Attribute name to extract (when extract_type is 'attribute')",
    )

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise Exception(
                "Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"
            )

        # Initialize browser
        playwright_instance = await async_playwright().start()
        browser_endpoint = Environment.get("BROWSER_URL")

        try:
            if browser_endpoint:
                browser = await playwright_instance.chromium.connect_over_cdp(
                    browser_endpoint
                )
                browser_context = await browser.new_context(bypass_csp=True)
            else:
                browser = await playwright_instance.chromium.launch(headless=True)
                browser_context = await browser.new_context(bypass_csp=True)

            # Create page from the context
            page = await browser_context.new_page()

            result = {
                "success": True,
                "action": self.action,
            }

            # Perform the requested action
            if self.action == self.Action.CLICK:
                if not self.selector:
                    raise ValueError("Selector is required for click action")

                # Wait for the element to be visible and clickable
                element = await page.wait_for_selector(
                    self.selector, timeout=self.timeout
                )
                if element:
                    await element.click()
                    result["clicked_selector"] = self.selector
                else:
                    raise ValueError(f"Element not found: {self.selector}")

            elif self.action == self.Action.GOTO:
                if not self.url:
                    raise ValueError("URL is required for goto action")

                await page.goto(
                    self.url, timeout=self.timeout, wait_until="domcontentloaded"
                )
                result["navigated_to"] = self.url

            elif self.action == self.Action.BACK:
                await page.go_back(timeout=self.timeout, wait_until="domcontentloaded")

            elif self.action == self.Action.FORWARD:
                await page.go_forward(
                    timeout=self.timeout, wait_until="domcontentloaded"
                )

            elif self.action == self.Action.RELOAD:
                await page.reload(timeout=self.timeout, wait_until="domcontentloaded")

            elif self.action == self.Action.EXTRACT:
                if self.selector:
                    # Wait for the element if specified
                    element = await page.wait_for_selector(
                        self.selector, timeout=self.timeout
                    )
                    if not element:
                        raise ValueError(f"Element not found: {self.selector}")

                    if self.extract_type == self.ExtractType.TEXT:
                        content = await element.text_content()
                    elif self.extract_type == self.ExtractType.HTML:
                        content = await element.inner_html()
                    elif self.extract_type == self.ExtractType.VALUE:
                        content = await element.input_value()
                    elif self.extract_type == self.ExtractType.ATTRIBUTE:
                        if not self.attribute:
                            raise ValueError(
                                "Attribute name is required for attribute extraction"
                            )
                        content = await element.get_attribute(self.attribute)
                else:
                    # Extract from entire page if no selector
                    if self.extract_type == self.ExtractType.TEXT:
                        content = await page.text_content("body")
                    elif self.extract_type == self.ExtractType.HTML:
                        content = await page.content()
                    else:
                        raise ValueError(
                            f"Invalid extract_type '{self.extract_type}' for full page extraction"
                        )

                result["content"] = content
                result["extract_type"] = self.extract_type
                if self.selector:
                    result["selector"] = self.selector

            # Wait for additional element if specified
            if self.wait_for:
                await page.wait_for_selector(self.wait_for, timeout=self.timeout)
                result["waited_for"] = self.wait_for

            # Add current page information to result
            result.update({"current_url": page.url, "title": await page.title()})

            return result

        except Exception as e:
            raise ValueError(f"Navigation/extraction action failed: {str(e)}")

        finally:
            # Always close the browser session
            await browser.close()
            await playwright_instance.stop()


class BrowserUseModel(str, Enum):
    GPT_4O = "gpt-4o"
    ANTHROPIC_CLAUDE_3_5_SONNET = "claude-3-5-sonnet"


class BrowserUseNode(BaseNode):
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

    model: BrowserUseModel = Field(
        default=BrowserUseModel.GPT_4O,
        description="The model to use for the browser agent.",
    )

    task: str = Field(
        default="",
        description="Natural language description of the browser task to perform. Can include complex multi-step instructions like 'Compare prices between websites', 'Fill out forms', or 'Extract specific data'.",
    )
    timeout: int = Field(
        default=300,
        description="Maximum time in seconds to allow for task completion. Complex tasks may require longer timeouts.",
        ge=1,
        le=3600,
    )
    use_remote_browser: bool = Field(
        default=True,
        description="Use a remote browser instead of a local one",
    )

    @classmethod
    def display_name(cls):
        return "Browser Agent"  # Provide a user-friendly name

    @classmethod
    def category(cls):
        return "Browser"  # Categorize the node

    @classmethod
    def return_type(cls):
        return {
            "success": bool,
            "task": str,
            "result": Any,  # Result can be varied
            "error": Optional[str],  # Include optional error field
        }

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Execute a browser agent task.
        """
        from browser_use import Agent, Browser as BrowserUse, BrowserConfig

        if self.model == BrowserUseModel.GPT_4O:
            openai_api_key = Environment.get("OPENAI_API_KEY")
            if not openai_api_key:
                return {
                    "success": False,
                    "task": self.task,
                    "error": "OpenAI API key not found in environment variables (OPENAI_API_KEY).",
                }

            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
        elif self.model == BrowserUseModel.ANTHROPIC_CLAUDE_3_5_SONNET:
            anthropic_api_key = Environment.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                return {
                    "success": False,
                    "task": self.task,
                    "error": "Anthropic API key not found in environment variables (ANTHROPIC_API_KEY).",
                }
            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet",
                temperature=0,
                api_key=anthropic_api_key,
                timeout=self.timeout,
                stop=["\n\n"],
            )

        browser_instance = None
        try:
            if self.use_remote_browser:
                browser_endpoint = Environment.get("BROWSER_URL")
                if not browser_endpoint:
                    raise ValueError(
                        "BrightData scraping browser endpoint not found in environment variables (BROWSER_URL)."
                    )

                # Use BrightData CDP endpoint
                browser_instance = BrowserUse(
                    config=BrowserConfig(
                        headless=True,  # Usually required for CDP connection
                        wss_url=browser_endpoint,
                    )
                )
            else:
                # Use local Playwright browser
                browser_instance = BrowserUse(
                    config=BrowserConfig(
                        headless=True,  # Can be set to False for local debugging if needed
                    )
                )

            # Create and run the agent
            agent = Agent(
                task=self.task,
                llm=llm,
                browser=browser_instance,
            )

            try:
                result = await asyncio.wait_for(agent.run(), timeout=self.timeout)
                return {
                    "success": True,
                    "task": self.task,
                    "result": result,
                    "error": None,
                }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "task": self.task,
                    "error": f"Task timed out after {self.timeout} seconds",
                    "result": None,
                }
            except Exception as agent_run_e:
                # Catch specific errors from agent.run() if possible
                return {
                    "success": False,
                    "task": self.task,
                    "error": f"Browser agent execution failed: {str(agent_run_e)}",
                    "result": None,
                }

        except Exception as setup_e:
            # Catch errors during setup (browser init, agent init)
            return {
                "success": False,
                "task": self.task,
                "error": f"Browser agent setup failed: {str(setup_e)}",
                "result": None,
            }
        finally:
            # Ensure browser is closed if it was initialized
            if browser_instance and hasattr(browser_instance, "close"):
                try:
                    await browser_instance.close()
                except Exception as close_e:
                    # Log error during close but don't overwrite primary error
                    logger.warning("Error closing browser instance: %s", close_e)
