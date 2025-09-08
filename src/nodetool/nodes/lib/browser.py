import asyncio
import logging
import os
import re
import traceback
from enum import Enum
from typing import Any, Dict, Iterator, Optional
import json
import hashlib

import aiohttp
import docker
import html2text
import trafilatura
from bs4 import BeautifulSoup
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# ServerDockerRunner is used at runtime to start a Playwright WS server in Docker
from nodetool.metadata.types import FilePath
from nodetool.workflows.base_node import ApiKeyMissingError, BaseNode
from nodetool.workflows.types import Notification, LogUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)

# Browser Use
os.environ["ANONYMIZED_TELEMETRY"] = "false"

PLAYWRIGHT_BASE_IMAGE = "mcr.microsoft.com/playwright:v1.55.0-noble"
PLAYWRIGHT_DRIVER_PORT = 3000

# Dockerfile content for the custom Playwright driver image.
# Builds on the official Microsoft Playwright image and ensures the Node.js
# Playwright CLI is installed for running the remote driver server.
_PLAYWRIGHT_DOCKERFILE = """
# Base Microsoft Playwright image
FROM mcr.microsoft.com/playwright:v1.55.0-noble

# Become root to install global npm packages if requested
USER root

RUN npm install -y -g --unsafe-perm playwright

# Set defaults and expose the Playwright driver port
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    DRIVER_PORT=3000

EXPOSE 3000

# Drop back to the non-root user used by Playwright images
USER pwuser

# Default command runs the Playwright driver server
CMD ["bash", "-lc", "playwright run-server --host 0.0.0.0 --port ${DRIVER_PORT}"]
"""


def _playwright_image_tag() -> str:
    """Return a deterministic image tag derived from the Dockerfile content.

    Including a short hash of the Dockerfile content in the tag ensures that
    changes to this file produce a new local image and avoid stale caches.
    """
    h = hashlib.sha256(_PLAYWRIGHT_DOCKERFILE.encode("utf-8")).hexdigest()[:12]
    return f"nodetool/playwright-driver:v1.55.0-noble-{h}"


def _ensure_custom_playwright_image(context: ProcessingContext, node: BaseNode) -> str:
    """Ensure the custom Playwright image exists locally; build if missing.

    Returns the image tag to use with Docker. Building happens in a workspace
    subdirectory so that users can inspect artifacts under `.nt-browser/`.
    """
    tag = _playwright_image_tag()
    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:
        raise RuntimeError(
            "Docker daemon is not available. Please start Docker and try again."
        ) from e

    try:
        client.images.get(tag)
        return tag
    except Exception:
        pass

    build_dir = context.resolve_workspace_path(
        os.path.join(".nt-browser", "images", f"playwright-{tag.split('-')[-1]}")
    )
    os.makedirs(build_dir, exist_ok=True)
    dockerfile_path = os.path.join(build_dir, "Dockerfile")
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(_PLAYWRIGHT_DOCKERFILE)

    context.post_message(
        Notification(
            node_id=node.id,
            severity="info",
            content=f"Building custom Playwright image: {tag}",
        )
    )
    context.post_message(
        LogUpdate(
            node_id=node.id,
            node_name=node.get_title(),
            content=f"Building custom Playwright image: {tag}",
            severity="info",
        )
    )

    # Build the image; stream build output lines as progress notifications.
    image, logs = client.images.build(
        path=build_dir,
        dockerfile="Dockerfile",
        tag=tag,
        rm=True,
        pull=True,
    )
    for entry in logs:
        text = str(entry)
        context.post_message(
            Notification(
                node_id=node.id,
                severity="info",
                content=text,
            )
        )
        context.post_message(
            LogUpdate(
                node_id=node.id,
                node_name=node.get_title(),
                content=text,
                severity="info",
            )
        )

    context.post_message(
        Notification(
            node_id=node.id,
            severity="info",
            content=f"Custom Playwright image ready: {tag}",
        )
    )
    context.post_message(
        LogUpdate(
            node_id=node.id,
            node_name=node.get_title(),
            content=f"Custom Playwright image ready: {tag}",
            severity="info",
        )
    )
    return tag


def _container_workspace_path(context: ProcessingContext, host_path: str) -> str:
    """Translate an absolute host path to the container workspace path.

    The Docker runner mounts the workspace at /workspace. This helper maps
    a host path inside the workspace to the corresponding container path.
    """
    # Normalize and ensure we are within the workspace
    abs_host = os.path.abspath(host_path)
    ws = os.path.abspath(context.workspace_dir)
    if not abs_host.startswith(ws + os.sep) and abs_host != ws:
        raise ValueError(
            f"Path '{host_path}' is outside of workspace '{context.workspace_dir}'"
        )
    rel = os.path.relpath(abs_host, ws)
    # Map to container mount
    return f"/workspace/{rel}" if rel != "." else "/workspace"


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

    _expose_as_tool: bool = True

    url: str = Field(default="", description="URL to navigate to")

    timeout: int = Field(
        default=20000, description="Timeout in milliseconds for page navigation"
    )

    @classmethod
    def return_type(cls):
        return {
            "success": bool,
            "content": str,
            "metadata": Dict[str, Any],
        }

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

    _runner: StreamRunnerBase | None = None

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.url:
            raise ValueError("URL is required")

        # Start Playwright driver in Docker and connect via remote WebSocket
        server_cmd = (
            f"playwright run-server --host 0.0.0.0 --port {PLAYWRIGHT_DRIVER_PORT}"
        )
        from nodetool.code_runners.server_runner import ServerDockerRunner

        runner = ServerDockerRunner(
            image=_ensure_custom_playwright_image(context, self),
            container_port=PLAYWRIGHT_DRIVER_PORT,
            scheme="ws",
            timeout_seconds=max(5, int(self.timeout / 1000) + 10),
            endpoint_path="/?ws=1",
        )
        self._runner = runner

        endpoint: str | None = None
        async for slot, value in runner.stream(
            user_code=server_cmd,
            env_locals={},
            context=context,
            node=self,
        ):
            # Stream docker logs as workflow logs
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            elif slot == "endpoint":
                endpoint = value
                break

        if not endpoint:
            raise ValueError("Failed to start Playwright server (no endpoint)")

        # Point Playwright Python at the remote driver
        os.environ["PLAYWRIGHT_DRIVER_URL"] = str(endpoint)
        from playwright.async_api import async_playwright

        async with async_playwright() as apw:
            browser = await apw.chromium.launch()
            ctx = await browser.new_context(bypass_csp=True)
            page = await ctx.new_page()
            await page.goto(self.url, wait_until="networkidle", timeout=self.timeout)
            html = await page.content()
            try:
                meta = trafilatura.extract_metadata(html).as_dict()
                meta.pop("body", None)
                meta.pop("commentsbody", None)
            except Exception:
                meta = {}
            try:
                content = trafilatura.extract(html) or ""
            except Exception:
                content = ""
            await browser.close()
            # Proactively stop server container
            try:
                runner.stop()
            except Exception:
                pass
            return {"success": True, "metadata": meta, "content": content}

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop the Playwright driver container if still running.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        if self._runner:
            try:
                self._runner.stop()
            except Exception:
                pass


class Screenshot(BaseNode):
    """
    Takes a screenshot of a web page or specific element.
    browser, screenshot, capture, image

    Use cases:
    - Capture visual representation of web pages
    - Document specific UI elements
    - Create visual records of web content
    """

    _expose_as_tool: bool = True

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

    _runner: StreamRunnerBase | None = None

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if not self.url:
            raise ValueError("URL is required")

        host_path = context.resolve_workspace_path(self.output_file.path)
        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        container_path = _container_workspace_path(context, host_path)

        # Start Playwright driver in Docker and take a screenshot via remote driver
        server_cmd = (
            f"playwright run-server --host 0.0.0.0 --port {PLAYWRIGHT_DRIVER_PORT}"
        )
        from nodetool.code_runners.server_runner import ServerDockerRunner

        runner = ServerDockerRunner(
            image=_ensure_custom_playwright_image(context, self),
            container_port=PLAYWRIGHT_DRIVER_PORT,
            scheme="ws",
            timeout_seconds=max(5, int(self.timeout / 1000) + 10),
            endpoint_path="/?ws=1",
        )
        self._runner = runner

        endpoint: str | None = None
        async for slot, value in runner.stream(
            user_code=server_cmd,
            env_locals={},
            context=context,
            node=self,
        ):
            if slot == "stdout":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="info",
                    )
                )
            elif slot == "stderr":
                context.post_message(
                    LogUpdate(
                        node_id=self.id,
                        node_name=self.get_title(),
                        content=str(value).rstrip("\n"),
                        severity="error",
                    )
                )
            elif slot == "endpoint":
                endpoint = str(value)
                break

        if not endpoint:
            raise ValueError("Failed to start Playwright server (no endpoint)")

        # Configure Playwright client to use remote driver
        os.environ["PLAYWRIGHT_DRIVER_URL"] = str(endpoint)
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
                await el.screenshot(path=container_path)
            else:
                await page.screenshot(path=container_path)
            await browser.close()
            try:
                runner.stop()
            except Exception:
                pass
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

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop the Playwright driver container if still running.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        if self._runner:
            try:
                self._runner.stop()
            except Exception:
                pass


class WebFetch(BaseNode):
    """
    Fetches HTML content from a URL and converts it to text.
    web, fetch, html, markdown, http

    Use cases:
    - Extract text content from web pages
    - Process web content for analysis
    - Save web content to files
    """

    _expose_as_tool: bool = True

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

    _expose_as_tool: bool = True

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

    _runner: StreamRunnerBase | None = None

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        if self.action == self.Action.GOTO and not self.url:
            raise ValueError("URL is required for goto action")

        # Use ServerDockerRunner to start Playwright driver and connect via Python client
        server_cmd = (
            f"playwright run-server --host 0.0.0.0 --port {PLAYWRIGHT_DRIVER_PORT}"
        )
        from nodetool.code_runners.server_runner import ServerDockerRunner

        runner = ServerDockerRunner(
            image=_ensure_custom_playwright_image(context, self),
            container_port=PLAYWRIGHT_DRIVER_PORT,
            scheme="ws",
            timeout_seconds=max(5, int(self.timeout / 1000) + 10),
            endpoint_path="/?ws=1",
        )
        self._runner = runner

        endpoint: str | None = None
        async for slot, value in runner.stream(
            user_code=server_cmd,
            env_locals={},
            context=context,
            node=self,
        ):
            if slot == "endpoint":
                endpoint = str(value)
                break

        if not endpoint:
            raise ValueError("Failed to start Playwright server (no endpoint)")

        # Configure Playwright client to use remote driver
        os.environ["PLAYWRIGHT_DRIVER_URL"] = str(endpoint)
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

            try:
                await browser.close()
            finally:
                try:
                    runner.stop()
                except Exception:
                    pass
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

    async def finalize(self, context: ProcessingContext):  # type: ignore[override]
        """Stop the Playwright driver container if still running.

        Args:
            context: Processing context (unused).

        Returns:
            None
        """
        if self._runner:
            try:
                self._runner.stop()
            except Exception:
                pass


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

    _expose_as_tool: bool = True

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
