import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.lib.pandoc import (
    ConvertText,
    ConvertFile,
    InputFormat,
    OutputFormat,
)
from nodetool.metadata.types import FilePath


@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="test")


def _make_mock_pypandoc():
    """Return a MagicMock that stands in for the pypandoc module."""
    mock = MagicMock()
    mock.convert_text.return_value = ""
    mock.convert_file.return_value = ""
    return mock


@pytest.mark.asyncio
async def test_convert_text_markdown_to_plain(context: ProcessingContext):
    """Test ConvertText converts Markdown to plain text via mocked pypandoc."""
    expected = "Hello, world!"
    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_text.return_value = expected

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertText(
            content="Hello, world!",
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
        )
        result = await node.process(context)

    assert result == expected


@pytest.mark.asyncio
async def test_convert_text_with_extra_args(context: ProcessingContext):
    """Test ConvertText passes extra_args to pypandoc."""
    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_text.return_value = "result"

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertText(
            content="# Title",
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
            extra_args=["--wrap=none"],
        )
        await node.process(context)

    mock_pypandoc.convert_text.assert_called_once_with(
        "# Title",
        OutputFormat.PLAIN.value,
        format=InputFormat.MARKDOWN.value,
        extra_args=["--wrap=none"],
    )


@pytest.mark.asyncio
async def test_convert_text_rst_to_plain(context: ProcessingContext):
    """Test ConvertText with RST as input format."""
    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_text.return_value = "Hello\n"

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertText(
            content="Hello\n=====\n",
            input_format=InputFormat.RST,
            output_format=OutputFormat.PLAIN,
        )
        result = await node.process(context)

    assert result == "Hello\n"


@pytest.mark.asyncio
async def test_convert_text_html_input(context: ProcessingContext):
    """Test ConvertText with HTML as input format."""
    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_text.return_value = "paragraph text"

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertText(
            content="<p>paragraph text</p>",
            input_format=InputFormat.HTML,
            output_format=OutputFormat.PLAIN,
        )
        result = await node.process(context)

    assert result == "paragraph text"


@pytest.mark.asyncio
async def test_convert_file_raises_when_path_empty(context: ProcessingContext):
    """Test ConvertFile raises AssertionError when input_path is empty."""
    mock_pypandoc = _make_mock_pypandoc()

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertFile(
            input_path=FilePath(path=""),
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
        )
        with pytest.raises((AssertionError, ValueError)):
            await node.process(context)


@pytest.mark.asyncio
async def test_convert_file_raises_when_file_not_found(
    context: ProcessingContext, tmp_path
):
    """Test ConvertFile raises ValueError when the file does not exist."""
    missing_path = str(tmp_path / "nonexistent.md")
    mock_pypandoc = _make_mock_pypandoc()

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertFile(
            input_path=FilePath(path=missing_path),
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
        )
        with pytest.raises(ValueError, match="Input file not found"):
            await node.process(context)


@pytest.mark.asyncio
async def test_convert_file_success(context: ProcessingContext, tmp_path):
    """Test ConvertFile successfully converts an existing file."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n\nThis is a test.")
    expected_output = "Hello\n\nThis is a test.\n"

    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_file.return_value = expected_output

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertFile(
            input_path=FilePath(path=str(md_file)),
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
        )
        result = await node.process(context)

    assert result == expected_output
    mock_pypandoc.convert_file.assert_called_once_with(
        str(md_file),
        OutputFormat.PLAIN.value,
        format=InputFormat.MARKDOWN.value,
        extra_args=[],
    )


@pytest.mark.asyncio
async def test_convert_file_with_extra_args(context: ProcessingContext, tmp_path):
    """Test ConvertFile passes extra_args to pypandoc."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello")

    mock_pypandoc = _make_mock_pypandoc()
    mock_pypandoc.convert_file.return_value = "Hello\n"

    with patch.dict(sys.modules, {"pypandoc": mock_pypandoc}):
        node = ConvertFile(
            input_path=FilePath(path=str(md_file)),
            input_format=InputFormat.MARKDOWN,
            output_format=OutputFormat.PLAIN,
            extra_args=["--toc"],
        )
        await node.process(context)

    mock_pypandoc.convert_file.assert_called_once_with(
        str(md_file),
        OutputFormat.PLAIN.value,
        format=InputFormat.MARKDOWN.value,
        extra_args=["--toc"],
    )


def test_input_format_enum_values():
    """Test that key InputFormat enum values exist."""
    assert InputFormat.MARKDOWN.value == "markdown"
    assert InputFormat.HTML.value == "html"
    assert InputFormat.RST.value == "rst"
    assert InputFormat.DOCX.value == "docx"
    assert InputFormat.CSV.value == "csv"


def test_output_format_enum_values():
    """Test that key OutputFormat enum values exist."""
    assert OutputFormat.PLAIN.value == "plain"
    assert OutputFormat.DOCX.value == "docx"
    assert OutputFormat.PDF.value == "pdf"
