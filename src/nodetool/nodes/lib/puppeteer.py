"""
Puppeteer-style browser automation nodes.

This module provides nodes for browser automation using pyppeteer,
the Python port of Puppeteer. These nodes enable headless browser
automation for web scraping, testing, and interaction.

puppeteer, browser, automation, web, scraping
"""

import os
from enum import Enum
from typing import Any, ClassVar, TypedDict

import pyppeteer
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef


class WaitUntilOption(str, Enum):
    """Options for when to consider navigation complete."""

    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    NETWORKIDLE0 = "networkidle0"
    NETWORKIDLE2 = "networkidle2"


class PuppeteerGoto(BaseNode):
    """
    Navigate to a URL using Puppeteer-style browser automation.
    puppeteer, browser, navigation, url, web

    Use cases:
    - Navigate to web pages for scraping
    - Load pages for automated testing
    - Open URLs for content extraction
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(
        default=30000, description="Timeout in milliseconds for navigation"
    )

    class OutputType(TypedDict):
        success: bool
        url: str
        title: str
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            title = await page.title()
            content = await page.content()
            return {
                "success": True,
                "url": self.url,
                "title": title,
                "content": content,
            }
        finally:
            await browser.close()


class PuppeteerClick(BaseNode):
    """
    Click an element on a web page using Puppeteer.
    puppeteer, browser, click, interaction, automation

    Use cases:
    - Click buttons or links on web pages
    - Trigger UI interactions for testing
    - Navigate through multi-page flows
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to before clicking")
    selector: str = Field(
        default="", description="CSS selector for the element to click"
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    wait_after_click: int = Field(
        default=1000,
        description="Time to wait in milliseconds after clicking",
    )

    class OutputType(TypedDict):
        success: bool
        url: str
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)
            await page.click(self.selector)
            if self.wait_after_click > 0:
                await page.waitFor(self.wait_after_click)
            content = await page.content()
            current_url = page.url
            return {
                "success": True,
                "url": current_url,
                "content": content,
            }
        finally:
            await browser.close()


class PuppeteerType(BaseNode):
    """
    Type text into an input field using Puppeteer.
    puppeteer, browser, type, input, form, automation

    Use cases:
    - Fill in form fields
    - Enter search queries
    - Input text data for testing
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to before typing")
    selector: str = Field(
        default="", description="CSS selector for the input field"
    )
    text: str = Field(default="", description="Text to type into the field")
    clear_first: bool = Field(
        default=True,
        description="Clear the field before typing",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    delay: int = Field(
        default=0,
        description="Delay between key presses in milliseconds (simulates human typing)",
    )

    class OutputType(TypedDict):
        success: bool
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)
            if self.clear_first:
                await page.click(self.selector, clickCount=3)
            await page.type(self.selector, self.text, delay=self.delay)
            content = await page.content()
            return {
                "success": True,
                "content": content,
            }
        finally:
            await browser.close()


class PuppeteerScreenshot(BaseNode):
    """
    Take a screenshot of a web page using Puppeteer.
    puppeteer, browser, screenshot, capture, image

    Use cases:
    - Capture visual state of web pages
    - Create screenshots for documentation
    - Visual testing and comparison
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(
        default="", description="URL to navigate to before taking screenshot"
    )
    selector: str = Field(
        default="",
        description="Optional CSS selector to screenshot specific element",
    )
    full_page: bool = Field(
        default=False,
        description="Capture the full scrollable page instead of viewport",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    viewport_width: int = Field(
        default=1280, description="Viewport width in pixels"
    )
    viewport_height: int = Field(
        default=720, description="Viewport height in pixels"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if not self.url:
            raise ValueError("URL is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.setViewport(
                {"width": self.viewport_width, "height": self.viewport_height}
            )
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            if self.selector:
                await page.waitForSelector(self.selector, timeout=self.timeout)
                element = await page.querySelector(self.selector)
                if not element:
                    raise ValueError(f"Element not found: {self.selector}")
                screenshot_bytes = await element.screenshot()
            else:
                screenshot_bytes = await page.screenshot(fullPage=self.full_page)

            return await context.image_from_bytes(screenshot_bytes)
        finally:
            await browser.close()


class PuppeteerEvaluate(BaseNode):
    """
    Execute JavaScript code on a web page using Puppeteer.
    puppeteer, browser, javascript, evaluate, execute

    Use cases:
    - Extract data using custom JavaScript
    - Manipulate page state
    - Run complex queries on the DOM
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(
        default="", description="URL to navigate to before executing JavaScript"
    )
    script: str = Field(
        default="",
        description="JavaScript code to execute. Use 'return' to return values.",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        result: Any

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.script:
            raise ValueError("Script is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            result = await page.evaluate(self.script)
            return {
                "success": True,
                "result": result,
            }
        finally:
            await browser.close()


class PuppeteerWaitForSelector(BaseNode):
    """
    Wait for an element to appear on the page using Puppeteer.
    puppeteer, browser, wait, selector, element

    Use cases:
    - Wait for dynamic content to load
    - Ensure elements exist before interaction
    - Handle asynchronous page updates
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    selector: str = Field(
        default="", description="CSS selector for the element to wait for"
    )
    visible: bool = Field(
        default=False,
        description="Wait for element to be visible (not just in DOM)",
    )
    hidden: bool = Field(
        default=False, description="Wait for element to be hidden or removed"
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        found: bool

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            options = {"timeout": self.timeout}
            if self.visible:
                options["visible"] = True
            if self.hidden:
                options["hidden"] = True

            await page.waitForSelector(self.selector, options)
            return {
                "success": True,
                "found": True,
            }
        except Exception:
            return {
                "success": False,
                "found": False,
            }
        finally:
            await browser.close()


class PuppeteerExtractText(BaseNode):
    """
    Extract text content from elements on a web page using Puppeteer.
    puppeteer, browser, extract, text, scraping

    Use cases:
    - Scrape text content from web pages
    - Extract article content
    - Gather text data for analysis
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    selector: str = Field(
        default="body",
        description="CSS selector for elements to extract text from",
    )
    all_matches: bool = Field(
        default=False,
        description="Extract text from all matching elements (returns list)",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        text: str | list[str]

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)

            if self.all_matches:
                text = await page.evaluate(
                    f"""() => {{
                        const elements = document.querySelectorAll('{self.selector}');
                        return Array.from(elements).map(el => el.innerText || el.textContent || '');
                    }}"""
                )
            else:
                text = await page.evaluate(
                    f"""() => {{
                        const el = document.querySelector('{self.selector}');
                        return el ? (el.innerText || el.textContent || '') : '';
                    }}"""
                )

            return {
                "success": True,
                "text": text,
            }
        finally:
            await browser.close()


class PuppeteerExtractAttribute(BaseNode):
    """
    Extract attribute values from elements on a web page using Puppeteer.
    puppeteer, browser, extract, attribute, scraping

    Use cases:
    - Extract href links from anchor tags
    - Get src attributes from images
    - Scrape data attributes
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    selector: str = Field(
        default="", description="CSS selector for elements"
    )
    attribute: str = Field(
        default="href", description="Attribute name to extract"
    )
    all_matches: bool = Field(
        default=False,
        description="Extract from all matching elements (returns list)",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        value: str | list[str] | None

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)

            if self.all_matches:
                value = await page.evaluate(
                    f"""() => {{
                        const elements = document.querySelectorAll('{self.selector}');
                        return Array.from(elements).map(el => el.getAttribute('{self.attribute}'));
                    }}"""
                )
            else:
                value = await page.evaluate(
                    f"""() => {{
                        const el = document.querySelector('{self.selector}');
                        return el ? el.getAttribute('{self.attribute}') : null;
                    }}"""
                )

            return {
                "success": True,
                "value": value,
            }
        finally:
            await browser.close()


class PuppeteerFillForm(BaseNode):
    """
    Fill out a form on a web page using Puppeteer.
    puppeteer, browser, form, fill, input, automation

    Use cases:
    - Automate form submission
    - Fill login forms
    - Complete multi-field forms
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    fields: dict[str, str] = Field(
        default={},
        description="Dictionary mapping CSS selectors to values to fill",
    )
    submit_selector: str = Field(
        default="",
        description="Optional CSS selector for submit button to click after filling",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    delay: int = Field(
        default=0,
        description="Delay between key presses in milliseconds",
    )

    class OutputType(TypedDict):
        success: bool
        url: str
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.fields:
            raise ValueError("At least one field is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )

            for selector, value in self.fields.items():
                await page.waitForSelector(selector, timeout=self.timeout)
                await page.click(selector, clickCount=3)  # Select all
                await page.type(selector, value, delay=self.delay)

            if self.submit_selector:
                await page.waitForSelector(
                    self.submit_selector, timeout=self.timeout
                )
                await page.click(self.submit_selector)
                await page.waitForNavigation(
                    waitUntil=self.wait_until.value, timeout=self.timeout
                )

            content = await page.content()
            current_url = page.url
            return {
                "success": True,
                "url": current_url,
                "content": content,
            }
        finally:
            await browser.close()


class PuppeteerPDF(BaseNode):
    """
    Generate a PDF from a web page using Puppeteer.
    puppeteer, browser, pdf, export, document

    Use cases:
    - Convert web pages to PDF documents
    - Generate printable versions of pages
    - Archive web content as PDF
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    output_file: str = Field(
        default="output.pdf",
        description="Path to save the PDF (relative to workspace)",
    )
    format: str = Field(
        default="A4",
        description="Paper format (A4, Letter, Legal, etc.)",
    )
    print_background: bool = Field(
        default=True,
        description="Include background graphics",
    )
    landscape: bool = Field(
        default=False,
        description="Use landscape orientation",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        path: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.output_file:
            raise ValueError("Output file path is required")


        host_path = context.resolve_workspace_path(self.output_file)
        os.makedirs(os.path.dirname(host_path) or ".", exist_ok=True)

        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.pdf(
                path=host_path,
                format=self.format,
                printBackground=self.print_background,
                landscape=self.landscape,
            )
            return {
                "success": True,
                "path": host_path,
            }
        finally:
            await browser.close()


class PuppeteerSelect(BaseNode):
    """
    Select an option from a dropdown menu using Puppeteer.
    puppeteer, browser, select, dropdown, form

    Use cases:
    - Select options in dropdown menus
    - Choose values in form selects
    - Automate dropdown interactions
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    selector: str = Field(
        default="", description="CSS selector for the select element"
    )
    value: str = Field(
        default="",
        description="Value to select (matches option's value attribute)",
    )
    by_text: bool = Field(
        default=False,
        description="Select by visible text instead of value attribute",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")

    class OutputType(TypedDict):
        success: bool
        selected: list[str]

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")
        if not self.value:
            raise ValueError("Value is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)

            if self.by_text:
                # Select by visible text using JavaScript
                selected = await page.evaluate(
                    f"""() => {{
                        const select = document.querySelector('{self.selector}');
                        const options = select.options;
                        for (let i = 0; i < options.length; i++) {{
                            if (options[i].text === '{self.value}') {{
                                select.value = options[i].value;
                                select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                return [options[i].value];
                            }}
                        }}
                        return [];
                    }}"""
                )
            else:
                selected = await page.select(self.selector, self.value)

            return {
                "success": True,
                "selected": selected,
            }
        finally:
            await browser.close()


class PuppeteerHover(BaseNode):
    """
    Hover over an element on a web page using Puppeteer.
    puppeteer, browser, hover, mouse, interaction

    Use cases:
    - Trigger hover states and tooltips
    - Reveal dropdown menus
    - Test hover interactions
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    selector: str = Field(
        default="", description="CSS selector for the element to hover over"
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    wait_after_hover: int = Field(
        default=500,
        description="Time to wait after hovering (ms) to let animations complete",
    )

    class OutputType(TypedDict):
        success: bool
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")
        if not self.selector:
            raise ValueError("Selector is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )
            await page.waitForSelector(self.selector, timeout=self.timeout)
            await page.hover(self.selector)
            if self.wait_after_hover > 0:
                await page.waitFor(self.wait_after_hover)
            content = await page.content()
            return {
                "success": True,
                "content": content,
            }
        finally:
            await browser.close()


class PuppeteerScroll(BaseNode):
    """
    Scroll on a web page using Puppeteer.
    puppeteer, browser, scroll, navigation

    Use cases:
    - Scroll to load lazy content
    - Navigate to page sections
    - Trigger infinite scroll loading
    """

    _expose_as_tool: ClassVar[bool] = True

    url: str = Field(default="", description="URL to navigate to")
    x: int = Field(default=0, description="Horizontal scroll amount in pixels")
    y: int = Field(default=0, description="Vertical scroll amount in pixels")
    selector: str = Field(
        default="",
        description="Optional CSS selector to scroll to (overrides x/y)",
    )
    wait_until: WaitUntilOption = Field(
        default=WaitUntilOption.NETWORKIDLE2,
        description="When to consider navigation complete",
    )
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    wait_after_scroll: int = Field(
        default=1000,
        description="Time to wait after scrolling (ms) for content to load",
    )

    class OutputType(TypedDict):
        success: bool
        content: str

    async def process(self, context: ProcessingContext) -> OutputType:
        if not self.url:
            raise ValueError("URL is required")


        browser = await pyppeteer.launch(headless=True)
        try:
            page = await browser.newPage()
            await page.goto(
                self.url,
                waitUntil=self.wait_until.value,
                timeout=self.timeout,
            )

            if self.selector:
                await page.waitForSelector(self.selector, timeout=self.timeout)
                await page.evaluate(
                    f"""() => {{
                        const el = document.querySelector('{self.selector}');
                        if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}"""
                )
            else:
                await page.evaluate(f"window.scrollBy({self.x}, {self.y})")

            if self.wait_after_scroll > 0:
                await page.waitFor(self.wait_after_scroll)

            content = await page.content()
            return {
                "success": True,
                "content": content,
            }
        finally:
            await browser.close()
