import asyncio
import os
import re
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional, ClassVar, TypedDict

import aiohttp
from pydantic import Field

from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger
from nodetool.config.environment import Environment

logger = get_logger(__name__)

# Browser Use
os.environ["ANONYMIZED_TELEMETRY"] = "false"

def sanitize_node_id(node_id: str) -> str:
    """
    Sanitize a node id for use as a directory name using regex.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(node_id))


def _write_script(
    context: ProcessingContext, name: str, code: str, node_id: str
) -> str:
    base_dir = context.resolve_workspace_path(
        os.path.join(".nt-browser", sanitize_node_id(node_id))
    )
    os.makedirs(base_dir, exist_ok=True)
    unique = (
        f"{int(__import__('time').time()*1000)}-{__import__('secrets').token_hex(4)}"
    )
    path = os.path.join(base_dir, f"{name}-{unique}.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


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

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")

    timeout: int = Field(
        default=20000, description="Timeout in milliseconds for page navigation"
    )

    class OutputType(TypedDict):
        success: bool
        content: str
        metadata: Dict[str, Any]

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override]
        """Return a conservative overall timeout for the browser task.

        Uses the configured page timeout (milliseconds) plus headroom to
        cover driver startup and teardown.

        Returns:
            float | None: Timeout in seconds.
        """
        try:
            return max(5.0, float(self.timeout) / 1000.0 + 20.0)
        except Exception:
            return 60.0


    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")

        from playwright.async_api import async_playwright

        async with async_playwright() as apw:
            browser = await apw.chromium.launch()
            ctx = await browser.new_context(bypass_csp=True)
            page = await ctx.new_page()
            await page.goto(self.url, wait_until="networkidle", timeout=self.timeout)
            html = await page.content()
            import trafilatura

            try:
                meta = trafilatura.extract_metadata(html).as_dict()
                meta.pop("body", None)
                meta.pop("commentsbody", None)
            except (AttributeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to extract metadata: {e}")
                meta = {}
            try:
                content = trafilatura.extract(html) or ""
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to extract content: {e}")
                content = ""
            await browser.close()
            return {"success": True, "metadata": meta, "content": content}



class Screenshot(BaseNode):
    """
    Takes a screenshot of a web page or specific element.
    browser, screenshot, capture, image

    Use cases:
    - Capture visual representation of web pages
    - Document specific UI elements
    - Create visual records of web content
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(
        default="", description="URL to navigate to before taking screenshot"
    )

    selector: str = Field(
        default="", description="Optional CSS selector for capturing a specific element"
    )

    output_file: str = Field(
        default="screenshot.png",
        description="Path to save the screenshot (relative to workspace)",
    )

    timeout: int = Field(
        default=30000, description="Timeout in milliseconds for page navigation"
    )


    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.url:
            raise ValueError("URL is required")

        if not self.output_file:
            raise ValueError("output_file cannot be empty")
        host_path = context.resolve_workspace_path(self.output_file)
        os.makedirs(os.path.dirname(host_path), exist_ok=True)

        from playwright.async_api import async_playwright

        async with async_playwright() as apw:
            browser = await apw.chromium.launch()
            ctx = await browser.new_context(bypass_csp=True)
            page = await ctx.new_page()
            await page.goto(
                self.url, wait_until="domcontentloaded", timeout=self.timeout
            )
            if self.selector:
                el = await page.wait_for_selector(self.selector, timeout=self.timeout)
                if not el:
                    raise ValueError(
                        f"No element found matching selector: {self.selector}"
                    )
                await el.screenshot(path=host_path)
            else:
                await page.screenshot(path=host_path)
            await browser.close()
            return {"success": True, "path": host_path, "url": self.url}

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override]
        """Return a conservative overall timeout for the screenshot task.

        Based on navigation timeout (milliseconds) plus headroom.

        Returns:
            float | None: Timeout in seconds.
        """
        try:
            return max(5.0, float(self.timeout) / 1000.0 + 20.0)
        except Exception:
            return 60.0



class WebFetch(BaseNode):
    """
    Fetches HTML content from a URL and converts it to text.
    web, fetch, html, markdown, http

    Use cases:
    - Extract text content from web pages
    - Process web content for analysis
    - Save web content to files
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to fetch content from")

    selector: str = Field(
        default="body", description="CSS selector to extract specific elements"
    )

    async def process(self, context: ProcessingContext) -> str:
        if not self.url:
            raise ValueError("URL is required")

        try:
            from bs4 import BeautifulSoup
            import html2text

            # Make HTTP request
            async with aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            ) as session:
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

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL of the file to download")

    async def process(self, context: ProcessingContext) -> bytes:
        if not self.url:
            raise ValueError("URL is required")

        try:
            async with aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            ) as session:
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
        if self.action == self.Action.GOTO and not self.url:
            raise ValueError("URL is required for goto action")

        from playwright.async_api import async_playwright

        async with async_playwright() as apw:
            browser = await apw.chromium.launch()
            ctx = await browser.new_context(bypass_csp=True)
            page = await ctx.new_page()

            # Perform action
            if self.action == self.Action.GOTO:
                await page.goto(
                    self.url, wait_until="domcontentloaded", timeout=self.timeout
                )
            elif self.action == self.Action.RELOAD:
                await page.goto(
                    self.url, wait_until="domcontentloaded", timeout=self.timeout
                )
                await page.reload(wait_until="domcontentloaded", timeout=self.timeout)
            elif self.action == self.Action.CLICK:
                await page.goto(
                    self.url, wait_until="domcontentloaded", timeout=self.timeout
                )
                el = await page.wait_for_selector(self.selector, timeout=self.timeout)
                if not el:
                    raise ValueError(f"Element not found: {self.selector}")
                await el.click()
            elif self.action == self.Action.EXTRACT:
                await page.goto(
                    self.url, wait_until="domcontentloaded", timeout=self.timeout
                )
            elif self.action == self.Action.BACK:
                await page.go_back(timeout=self.timeout, wait_until="domcontentloaded")
            elif self.action == self.Action.FORWARD:
                await page.go_forward(
                    timeout=self.timeout, wait_until="domcontentloaded"
                )

            if self.wait_for:
                await page.wait_for_selector(self.wait_for, timeout=self.timeout)

            extracted = None
            if self.action == self.Action.EXTRACT:
                if self.selector:
                    el = await page.wait_for_selector(
                        self.selector, timeout=self.timeout
                    )
                    if not el:
                        raise ValueError(f"Element not found: {self.selector}")
                    if self.extract_type == self.ExtractType.HTML:
                        extracted = await el.evaluate("e => e.outerHTML")
                    elif self.extract_type == self.ExtractType.VALUE:
                        extracted = await el.evaluate("e => e.value ?? ''")
                    elif self.extract_type == self.ExtractType.ATTRIBUTE:
                        attr = self.attribute or ""
                        extracted = await el.evaluate(
                            "(e, a) => e.getAttribute(a)", attr
                        )
                    else:
                        extracted = await el.evaluate(
                            "e => e.innerText ?? e.textContent ?? ''"
                        )
                else:
                    if self.extract_type == self.ExtractType.HTML:
                        extracted = await page.content()
                    else:
                        extracted = await page.evaluate(
                            "() => document.body ? document.body.innerText : ''"
                        )

            await browser.close()
            return {
                "success": True,
                "action": self.action.value,
                "extracted": extracted,
            }

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override]
        """Return an overall timeout covering navigation and interactions.

        Uses the configured action timeout (milliseconds) plus headroom.

        Returns:
            float | None: Timeout in seconds.
        """
        try:
            return max(5.0, float(self.timeout) / 1000.0 + 20.0)
        except Exception:
            return 60.0



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

    _expose_as_tool: ClassVar[bool] = True

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

    class OutputType(TypedDict):
        success: bool
        task: str
        result: Any  # Result can be varied
        error: Optional[str]  # Include optional error field

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Execute a browser agent task.
        """
        from browser_use import Agent, Browser as BrowserUse, BrowserConfig
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI

        if self.model == BrowserUseModel.GPT_4O:
            openai_api_key = Environment.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ApiKeyMissingError(
                    "OpenAI API key not configured in NodeTool settings."
                )

            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
        elif self.model == BrowserUseModel.ANTHROPIC_CLAUDE_3_5_SONNET:
            anthropic_api_key = Environment.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ApiKeyMissingError(
                    "Anthropic API key not configured in NodeTool settings."
                )

            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet",
                temperature=0,
                api_key=anthropic_api_key,
                timeout=self.timeout,
                stop=["\n\n"],
            )

        browser_instance = None
        try:
            browser_url = Environment.get("BROWSER_URL")
            if self.use_remote_browser and browser_url:
                # Use a remote browser endpoint
                browser_instance = BrowserUse(
                    config=BrowserConfig(
                        headless=True,  # Usually required for CDP connection
                        wss_url=browser_url,
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


class SpiderCrawl(BaseNode):
    """
    Crawls websites following links and emitting URLs with optional HTML content.
    spider, crawler, web scraping, links, sitemap

    Use cases:
    - Build sitemaps and discover website structure
    - Collect URLs for bulk processing
    - Find all pages on a website
    - Extract content from multiple pages
    - Feed agentic workflows with discovered pages
    - Analyze website content and structure
    """

    _expose_as_tool: ClassVar[bool] = True

    start_url: str = Field(
        default="",
        description="The starting URL to begin crawling from",
    )

    max_depth: int = Field(
        default=2,
        description="Maximum depth to crawl (0 = start page only, 1 = start + linked pages, etc.)",
        ge=0,
        le=10,
    )

    max_pages: int = Field(
        default=50,
        description="Maximum number of pages to crawl (safety limit)",
        ge=1,
        le=1000,
    )

    same_domain_only: bool = Field(
        default=True,
        description="Only follow links within the same domain as the start URL",
    )

    include_html: bool = Field(
        default=False,
        description="Include the HTML content of each page in the output (increases bandwidth)",
    )

    respect_robots_txt: bool = Field(
        default=True,
        description="Respect robots.txt rules (follows web crawler best practices)",
    )

    delay_ms: int = Field(
        default=1000,
        description="Delay in milliseconds between requests (politeness policy)",
        ge=0,
        le=10000,
    )

    timeout: int = Field(
        default=30000,
        description="Timeout in milliseconds for each page load",
        ge=1000,
        le=120000,
    )

    url_pattern: str = Field(
        default="",
        description="Optional regex pattern to filter URLs (only crawl matching URLs)",
    )

    exclude_pattern: str = Field(
        default="",
        description="Optional regex pattern to exclude URLs (skip matching URLs)",
    )

    class OutputType(TypedDict):
        url: str
        depth: int
        html: Optional[str]
        title: Optional[str]
        status_code: int

    def get_timeout_seconds(self) -> float | None:  # type: ignore[override] # Returns timeout for spider crawl task based on max pages
        """Return overall timeout for spider crawl.

        Based on max pages and delays to prevent long-running crawls.
        """
        estimated = (self.max_pages * (self.timeout / 1000.0 + self.delay_ms / 1000.0)) + 60.0
        return min(estimated, 600.0)  # Cap at 10 minutes

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        """Stream discovered URLs and their content as they are crawled."""
        from urllib.parse import urljoin, urlparse
        from playwright.async_api import async_playwright
        import asyncio

        if not self.start_url:
            raise ValueError("start_url is required")

        # Parse start URL to get domain
        start_parsed = urlparse(self.start_url)
        start_domain = f"{start_parsed.scheme}://{start_parsed.netloc}"

        # Track visited URLs and URLs to visit
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(self.start_url, 0)]  # (url, depth)
        pages_crawled = 0

        # Compile URL patterns if provided
        url_pattern_re = None
        if self.url_pattern:
            try:
                url_pattern_re = re.compile(self.url_pattern)
            except re.error as e:
                logger.warning("Invalid url_pattern regex: %s", e)

        exclude_pattern_re = None
        if self.exclude_pattern:
            try:
                exclude_pattern_re = re.compile(self.exclude_pattern)
            except re.error as e:
                logger.warning("Invalid exclude_pattern regex: %s", e)

        # Check robots.txt if enabled
        robot_parser = None
        if self.respect_robots_txt:
            try:
                from urllib.robotparser import RobotFileParser
                robot_parser = RobotFileParser()
                robots_url = urljoin(start_domain, "/robots.txt")
                robot_parser.set_url(robots_url)
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                robots_content = await resp.text()
                                robot_parser.parse(robots_content.splitlines())
                                logger.info("Loaded robots.txt from %s", robots_url)
                    except Exception as e:
                        logger.info("Could not load robots.txt: %s", e)
                        robot_parser = None
            except ImportError:
                logger.warning("urllib.robotparser not available, skipping robots.txt check")
                robot_parser = None

        async with async_playwright() as apw:
            browser = await apw.chromium.launch()
            ctx = await browser.new_context(
                bypass_csp=True,
                user_agent="Mozilla/5.0 (compatible; NodeToolSpider/1.0; +http://nodetool.ai/bot)"
            )
            page = await ctx.new_page()

            try:
                while to_visit and pages_crawled < self.max_pages:
                    current_url, depth = to_visit.pop(0)

                    # Skip if already visited
                    if current_url in visited:
                        continue

                    # Check depth limit
                    if depth > self.max_depth:
                        continue

                    # Check robots.txt
                    if robot_parser and not robot_parser.can_fetch("*", current_url):
                        logger.debug("Skipping %s (blocked by robots.txt)", current_url)
                        continue

                    # Check URL patterns
                    if url_pattern_re and not url_pattern_re.search(current_url):
                        continue
                    if exclude_pattern_re and exclude_pattern_re.search(current_url):
                        continue

                    visited.add(current_url)

                    try:
                        # Navigate to page
                        response = await page.goto(
                            current_url,
                            wait_until="domcontentloaded",
                            timeout=self.timeout
                        )

                        status_code = response.status if response else 0

                        # Extract page info
                        title = await page.title()
                        html_content = None
                        if self.include_html:
                            html_content = await page.content()

                        # Yield current page
                        yield {
                            "url": current_url,
                            "depth": depth,
                            "html": html_content,
                            "title": title,
                            "status_code": status_code,
                        }

                        pages_crawled += 1

                        # Extract links if not at max depth
                        if depth < self.max_depth:
                            # Extract all href attributes from anchor tags, filtering out non-http links
                            links = await page.evaluate(
                                """() => {
                                    return Array.from(document.querySelectorAll('a[href]'))
                                        .map(a => a.href)
                                        .filter(href => 
                                            href && 
                                            !href.startsWith('javascript:') && 
                                            !href.startsWith('mailto:') && 
                                            !href.startsWith('tel:')
                                        );
                                }"""
                            )

                            # Process discovered links
                            for link in links:
                                try:
                                    # Normalize URL
                                    normalized_link = urljoin(current_url, link)
                                    # Remove fragment
                                    normalized_link = normalized_link.split('#')[0]

                                    # Skip if already visited or queued
                                    if normalized_link in visited:
                                        continue

                                    # Check domain restriction
                                    if self.same_domain_only:
                                        link_parsed = urlparse(normalized_link)
                                        link_domain = f"{link_parsed.scheme}://{link_parsed.netloc}"
                                        if link_domain != start_domain:
                                            continue

                                    # Add to queue
                                    if not any(url == normalized_link for url, _ in to_visit):
                                        to_visit.append((normalized_link, depth + 1))

                                except Exception as e:
                                    logger.debug("Error processing link %s: %s", link, e)

                        # Politeness delay
                        if self.delay_ms > 0 and to_visit:
                            await asyncio.sleep(self.delay_ms / 1000.0)

                    except Exception as e:
                        logger.warning("Error crawling %s: %s", current_url, e)
                        # Still yield the URL with error info
                        yield {
                            "url": current_url,
                            "depth": depth,
                            "html": None,
                            "title": None,
                            "status_code": 0,
                        }

            finally:
                await browser.close()

        logger.info("Spider crawl completed: %d pages visited", pages_crawled)
