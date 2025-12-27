"""
Tests for Puppeteer automation nodes.

These tests use mocking to avoid requiring a real browser or network.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.puppeteer import (
    PuppeteerGoto,
    PuppeteerClick,
    PuppeteerType,
    PuppeteerScreenshot,
    PuppeteerEvaluate,
    PuppeteerWaitForSelector,
    PuppeteerExtractText,
    PuppeteerExtractAttribute,
    PuppeteerFillForm,
    PuppeteerSelect,
    PuppeteerHover,
    PuppeteerScroll,
    WaitUntilOption,
)


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def create_mock_browser():
    """Create a mock pyppeteer browser with all necessary methods."""
    mock_page = AsyncMock()
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.content = AsyncMock(return_value="<html><body>Test Content</body></html>")
    mock_page.url = "https://example.com"
    mock_page.goto = AsyncMock()
    mock_page.waitForSelector = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.type = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="result")
    mock_page.screenshot = AsyncMock(return_value=b"fake_image_bytes")
    mock_page.querySelector = AsyncMock()
    mock_page.setViewport = AsyncMock()
    mock_page.hover = AsyncMock()
    mock_page.select = AsyncMock(return_value=["value1"])
    mock_page.waitFor = AsyncMock()
    mock_page.waitForNavigation = AsyncMock()
    mock_page.pdf = AsyncMock()

    mock_element = AsyncMock()
    mock_element.screenshot = AsyncMock(return_value=b"element_image_bytes")
    mock_page.querySelector.return_value = mock_element

    mock_browser = AsyncMock()
    mock_browser.newPage = AsyncMock(return_value=mock_page)
    mock_browser.close = AsyncMock()

    return mock_browser, mock_page


@pytest.mark.asyncio
async def test_puppeteer_goto(context: ProcessingContext):
    """Test PuppeteerGoto navigates to a URL and returns page info."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerGoto(url="https://example.com")
        result = await node.process(context)

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        assert "Test Content" in result["content"]
        mock_page.goto.assert_called_once()
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_goto_requires_url(context: ProcessingContext):
    """Test that PuppeteerGoto raises error when URL is missing."""
    node = PuppeteerGoto(url="")
    with pytest.raises(ValueError, match="URL is required"):
        await node.process(context)


@pytest.mark.asyncio
async def test_puppeteer_click(context: ProcessingContext):
    """Test PuppeteerClick clicks an element on the page."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerClick(
            url="https://example.com",
            selector="#submit-btn",
            wait_after_click=0,
        )
        result = await node.process(context)

        assert result["success"] is True
        mock_page.click.assert_called_with("#submit-btn")
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_type(context: ProcessingContext):
    """Test PuppeteerType types text into an input field."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerType(
            url="https://example.com",
            selector="#search-input",
            text="hello world",
            clear_first=True,
        )
        result = await node.process(context)

        assert result["success"] is True
        mock_page.type.assert_called_with("#search-input", "hello world", delay=0)
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_screenshot(context: ProcessingContext):
    """Test PuppeteerScreenshot captures page screenshot."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerScreenshot(url="https://example.com")
        result = await node.process(context)

        # Result should be an ImageRef
        assert result is not None
        mock_page.screenshot.assert_called_once()
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_evaluate(context: ProcessingContext):
    """Test PuppeteerEvaluate executes JavaScript on page."""
    mock_browser, mock_page = create_mock_browser()
    mock_page.evaluate = AsyncMock(return_value={"data": "test"})

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerEvaluate(
            url="https://example.com",
            script="return document.title",
        )
        result = await node.process(context)

        assert result["success"] is True
        assert result["result"] == {"data": "test"}
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_wait_for_selector_found(context: ProcessingContext):
    """Test PuppeteerWaitForSelector when element is found."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerWaitForSelector(
            url="https://example.com",
            selector=".loading-complete",
        )
        result = await node.process(context)

        assert result["success"] is True
        assert result["found"] is True
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_wait_for_selector_not_found(context: ProcessingContext):
    """Test PuppeteerWaitForSelector when element is not found."""
    mock_browser, mock_page = create_mock_browser()
    mock_page.waitForSelector = AsyncMock(side_effect=Exception("Timeout"))

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerWaitForSelector(
            url="https://example.com",
            selector=".nonexistent",
        )
        result = await node.process(context)

        assert result["success"] is False
        assert result["found"] is False
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_extract_text(context: ProcessingContext):
    """Test PuppeteerExtractText extracts text content."""
    mock_browser, mock_page = create_mock_browser()
    mock_page.evaluate = AsyncMock(return_value="Extracted text content")

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerExtractText(
            url="https://example.com",
            selector=".article-content",
        )
        result = await node.process(context)

        assert result["success"] is True
        assert result["text"] == "Extracted text content"
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_extract_text_all_matches(context: ProcessingContext):
    """Test PuppeteerExtractText extracts text from all matches."""
    mock_browser, mock_page = create_mock_browser()
    mock_page.evaluate = AsyncMock(return_value=["Item 1", "Item 2", "Item 3"])

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerExtractText(
            url="https://example.com",
            selector=".list-item",
            all_matches=True,
        )
        result = await node.process(context)

        assert result["success"] is True
        assert result["text"] == ["Item 1", "Item 2", "Item 3"]


@pytest.mark.asyncio
async def test_puppeteer_extract_attribute(context: ProcessingContext):
    """Test PuppeteerExtractAttribute extracts attribute values."""
    mock_browser, mock_page = create_mock_browser()
    mock_page.evaluate = AsyncMock(return_value="https://example.com/link")

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerExtractAttribute(
            url="https://example.com",
            selector="a.main-link",
            attribute="href",
        )
        result = await node.process(context)

        assert result["success"] is True
        assert result["value"] == "https://example.com/link"


@pytest.mark.asyncio
async def test_puppeteer_fill_form(context: ProcessingContext):
    """Test PuppeteerFillForm fills multiple form fields."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerFillForm(
            url="https://example.com/form",
            fields={
                "#username": "testuser",
                "#password": "testpass",
            },
        )
        result = await node.process(context)

        assert result["success"] is True
        assert mock_page.type.call_count == 2
        mock_browser.close.assert_called_once()


@pytest.mark.asyncio
async def test_puppeteer_select(context: ProcessingContext):
    """Test PuppeteerSelect selects dropdown option."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerSelect(
            url="https://example.com",
            selector="#country-select",
            value="us",
        )
        result = await node.process(context)

        assert result["success"] is True
        mock_page.select.assert_called_with("#country-select", "us")


@pytest.mark.asyncio
async def test_puppeteer_hover(context: ProcessingContext):
    """Test PuppeteerHover hovers over an element."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerHover(
            url="https://example.com",
            selector=".dropdown-trigger",
            wait_after_hover=0,
        )
        result = await node.process(context)

        assert result["success"] is True
        mock_page.hover.assert_called_with(".dropdown-trigger")


@pytest.mark.asyncio
async def test_puppeteer_scroll(context: ProcessingContext):
    """Test PuppeteerScroll scrolls the page."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerScroll(
            url="https://example.com",
            y=500,
            wait_after_scroll=0,
        )
        result = await node.process(context)

        assert result["success"] is True
        mock_page.evaluate.assert_called()


@pytest.mark.asyncio
async def test_puppeteer_scroll_to_selector(context: ProcessingContext):
    """Test PuppeteerScroll scrolls to a specific element."""
    mock_browser, mock_page = create_mock_browser()

    with patch("nodetool.nodes.lib.puppeteer.pyppeteer") as mock_pyppeteer:
        mock_pyppeteer.launch = AsyncMock(return_value=mock_browser)

        node = PuppeteerScroll(
            url="https://example.com",
            selector="#footer",
            wait_after_scroll=0,
        )
        result = await node.process(context)

        assert result["success"] is True


@pytest.mark.asyncio
async def test_wait_until_options():
    """Test that all WaitUntilOption values are valid."""
    assert WaitUntilOption.LOAD.value == "load"
    assert WaitUntilOption.DOMCONTENTLOADED.value == "domcontentloaded"
    assert WaitUntilOption.NETWORKIDLE0.value == "networkidle0"
    assert WaitUntilOption.NETWORKIDLE2.value == "networkidle2"
