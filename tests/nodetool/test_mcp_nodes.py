"""
Tests for MCP (Model Context Protocol) nodes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.nodes.lib.mcp import (
    MCPStdioClient,
    MCPSSEClient,
    MCPStreamableHTTPClient,
    MCPCallTool,
    MCPReadResource,
    MCPGetPrompt,
    TransportType,
)


class MockTool:
    """Mock MCP Tool."""

    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description
        self.inputSchema = {"type": "object", "properties": {}}


class MockResource:
    """Mock MCP Resource."""

    def __init__(self, name: str, uri: str, description: str | None = None):
        self.name = name
        self.uri = uri
        self.description = description
        self.mimeType = "text/plain"


class MockPrompt:
    """Mock MCP Prompt."""

    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description
        self.arguments = None


class MockPromptArgument:
    """Mock MCP Prompt Argument."""

    def __init__(self, name: str, required: bool = False):
        self.name = name
        self.description = f"Description for {name}"
        self.required = required


class MockServerInfo:
    """Mock MCP Server Info."""

    def __init__(self, name: str = "test-server", version: str = "1.0.0"):
        self.name = name
        self.version = version


class MockInitializeResult:
    """Mock MCP Initialize Result."""

    def __init__(self):
        self.serverInfo = MockServerInfo()


class MockListToolsResult:
    """Mock MCP List Tools Result."""

    def __init__(self, tools: list | None = None):
        self.tools = tools or []


class MockListResourcesResult:
    """Mock MCP List Resources Result."""

    def __init__(self, resources: list | None = None):
        self.resources = resources or []


class MockListPromptsResult:
    """Mock MCP List Prompts Result."""

    def __init__(self, prompts: list | None = None):
        self.prompts = prompts or []


class MockTextContent:
    """Mock MCP Text Content."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text

    def model_dump(self) -> dict:
        return {"type": self.type, "text": self.text}


class MockCallToolResult:
    """Mock MCP Call Tool Result."""

    def __init__(self, content: list | None = None, is_error: bool = False):
        self.content = content or [MockTextContent("result")]
        self.isError = is_error


class MockResourceContent:
    """Mock MCP Resource Content."""

    def __init__(self, uri: str, text: str | None = None):
        self.uri = uri
        self.mimeType = "text/plain"
        self.text = text
        self.blob = None


class MockReadResourceResult:
    """Mock MCP Read Resource Result."""

    def __init__(self, contents: list | None = None):
        self.contents = contents or []


class MockPromptMessage:
    """Mock MCP Prompt Message."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class MockGetPromptResult:
    """Mock MCP Get Prompt Result."""

    def __init__(self, messages: list | None = None, description: str | None = None):
        self.messages = messages or []
        self.description = description


class MockClientSession:
    """Mock MCP Client Session."""

    def __init__(self, *args, **kwargs):
        self.initialize = AsyncMock(return_value=MockInitializeResult())
        self.list_tools = AsyncMock(
            return_value=MockListToolsResult(
                [MockTool("test-tool", "A test tool")]
            )
        )
        self.list_resources = AsyncMock(
            return_value=MockListResourcesResult(
                [MockResource("test-resource", "file:///test", "A test resource")]
            )
        )
        self.list_prompts = AsyncMock(
            return_value=MockListPromptsResult(
                [MockPrompt("test-prompt", "A test prompt")]
            )
        )
        self.call_tool = AsyncMock(
            return_value=MockCallToolResult([MockTextContent("tool result")])
        )
        self.read_resource = AsyncMock(
            return_value=MockReadResourceResult(
                [MockResourceContent("file:///test", "resource content")]
            )
        )
        self.get_prompt = AsyncMock(
            return_value=MockGetPromptResult(
                [MockPromptMessage("user", "Hello")],
                "Test prompt description",
            )
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_context():
    """Create a mock ProcessingContext."""
    ctx = MagicMock()
    return ctx


class MockAsyncContextManager:
    """A generic async context manager for mocking."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_stdio_client():
    """Create a mock for stdio_client."""
    mock_read = MagicMock()
    mock_write = MagicMock()

    def mock_stdio(*args, **kwargs):
        return MockAsyncContextManager((mock_read, mock_write))

    return mock_stdio


@pytest.fixture
def mock_sse_client():
    """Create a mock for sse_client."""
    mock_read = MagicMock()
    mock_write = MagicMock()

    def mock_sse(*args, **kwargs):
        return MockAsyncContextManager((mock_read, mock_write))

    return mock_sse


@pytest.fixture
def mock_http_client():
    """Create a mock for streamable_http_client."""
    mock_read = MagicMock()
    mock_write = MagicMock()
    mock_get_session = MagicMock(return_value=None)

    def mock_http(*args, **kwargs):
        return MockAsyncContextManager((mock_read, mock_write, mock_get_session))

    return mock_http


# Test node titles and basic fields
def test_mcp_stdio_client_title():
    """Test MCPStdioClient has correct title."""
    assert MCPStdioClient.get_title() == "MCP Stdio Client"


def test_mcp_sse_client_title():
    """Test MCPSSEClient has correct title."""
    assert MCPSSEClient.get_title() == "MCP SSE Client"


def test_mcp_http_client_title():
    """Test MCPStreamableHTTPClient has correct title."""
    assert MCPStreamableHTTPClient.get_title() == "MCP HTTP Client"


def test_mcp_call_tool_title():
    """Test MCPCallTool has correct title."""
    assert MCPCallTool.get_title() == "MCP Call Tool"


def test_mcp_read_resource_title():
    """Test MCPReadResource has correct title."""
    assert MCPReadResource.get_title() == "MCP Read Resource"


def test_mcp_get_prompt_title():
    """Test MCPGetPrompt has correct title."""
    assert MCPGetPrompt.get_title() == "MCP Get Prompt"


def test_mcp_stdio_client_basic_fields():
    """Test MCPStdioClient has correct basic fields."""
    assert MCPStdioClient.get_basic_fields() == ["command", "args"]


def test_mcp_sse_client_basic_fields():
    """Test MCPSSEClient has correct basic fields."""
    assert MCPSSEClient.get_basic_fields() == ["url"]


def test_mcp_http_client_basic_fields():
    """Test MCPStreamableHTTPClient has correct basic fields."""
    assert MCPStreamableHTTPClient.get_basic_fields() == ["url"]


def test_mcp_call_tool_basic_fields():
    """Test MCPCallTool has correct basic fields."""
    assert MCPCallTool.get_basic_fields() == ["transport", "tool_name", "tool_arguments"]


def test_mcp_read_resource_basic_fields():
    """Test MCPReadResource has correct basic fields."""
    assert MCPReadResource.get_basic_fields() == ["transport", "resource_uri"]


def test_mcp_get_prompt_basic_fields():
    """Test MCPGetPrompt has correct basic fields."""
    assert MCPGetPrompt.get_basic_fields() == ["transport", "prompt_name", "prompt_arguments"]


# Test transport type enum
def test_transport_type_values():
    """Test TransportType enum values."""
    assert TransportType.STDIO.value == "stdio"
    assert TransportType.SSE.value == "sse"
    assert TransportType.HTTP.value == "http"


# Test validation errors
@pytest.mark.asyncio
async def test_mcp_stdio_client_requires_command(mock_context):
    """Test MCPStdioClient raises error when command is empty."""
    node = MCPStdioClient(command="")
    with pytest.raises(ValueError, match="Command is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_sse_client_requires_url(mock_context):
    """Test MCPSSEClient raises error when URL is empty."""
    node = MCPSSEClient(url="")
    with pytest.raises(ValueError, match="URL is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_http_client_requires_url(mock_context):
    """Test MCPStreamableHTTPClient raises error when URL is empty."""
    node = MCPStreamableHTTPClient(url="")
    with pytest.raises(ValueError, match="URL is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_call_tool_requires_tool_name(mock_context):
    """Test MCPCallTool raises error when tool_name is empty."""
    node = MCPCallTool(tool_name="", command="test")
    with pytest.raises(ValueError, match="Tool name is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_read_resource_requires_uri(mock_context):
    """Test MCPReadResource raises error when resource_uri is empty."""
    node = MCPReadResource(resource_uri="", command="test")
    with pytest.raises(ValueError, match="Resource URI is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_get_prompt_requires_prompt_name(mock_context):
    """Test MCPGetPrompt raises error when prompt_name is empty."""
    node = MCPGetPrompt(prompt_name="", command="test")
    with pytest.raises(ValueError, match="Prompt name is required"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_call_tool_stdio_requires_command(mock_context):
    """Test MCPCallTool with stdio transport raises error when command is empty."""
    node = MCPCallTool(
        transport=TransportType.STDIO,
        tool_name="test-tool",
        command="",
    )
    with pytest.raises(ValueError, match="Command is required for stdio transport"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_call_tool_sse_requires_url(mock_context):
    """Test MCPCallTool with SSE transport raises error when URL is empty."""
    node = MCPCallTool(
        transport=TransportType.SSE,
        tool_name="test-tool",
        url="",
    )
    with pytest.raises(ValueError, match="URL is required for SSE transport"):
        await node.process(mock_context)


@pytest.mark.asyncio
async def test_mcp_call_tool_http_requires_url(mock_context):
    """Test MCPCallTool with HTTP transport raises error when URL is empty."""
    node = MCPCallTool(
        transport=TransportType.HTTP,
        tool_name="test-tool",
        url="",
    )
    with pytest.raises(ValueError, match="URL is required for HTTP transport"):
        await node.process(mock_context)


# Test with mocked MCP client
@pytest.mark.asyncio
async def test_mcp_stdio_client_process(mock_context, mock_stdio_client):
    """Test MCPStdioClient successfully connects and returns data."""
    with patch(
        "mcp.client.stdio.stdio_client", mock_stdio_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPStdioClient(command="test-command", args=["--arg1"])
        result = await node.process(mock_context)

        assert result["server_name"] == "test-server"
        assert result["server_version"] == "1.0.0"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test-tool"
        assert len(result["resources"]) == 1
        assert result["resources"][0]["name"] == "test-resource"
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "test-prompt"


@pytest.mark.asyncio
async def test_mcp_sse_client_process(mock_context, mock_sse_client):
    """Test MCPSSEClient successfully connects and returns data."""
    with patch(
        "mcp.client.sse.sse_client", mock_sse_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPSSEClient(url="http://localhost:8080/sse")
        result = await node.process(mock_context)

        assert result["server_name"] == "test-server"
        assert result["server_version"] == "1.0.0"
        assert len(result["tools"]) == 1
        assert len(result["resources"]) == 1
        assert len(result["prompts"]) == 1


@pytest.mark.asyncio
async def test_mcp_http_client_process(mock_context, mock_http_client):
    """Test MCPStreamableHTTPClient successfully connects and returns data."""
    with patch(
        "mcp.client.streamable_http.streamable_http_client", mock_http_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPStreamableHTTPClient(url="http://localhost:8080/mcp")
        result = await node.process(mock_context)

        assert result["server_name"] == "test-server"
        assert result["server_version"] == "1.0.0"
        assert len(result["tools"]) == 1
        assert len(result["resources"]) == 1
        assert len(result["prompts"]) == 1


@pytest.mark.asyncio
async def test_mcp_call_tool_stdio(mock_context, mock_stdio_client):
    """Test MCPCallTool with stdio transport."""
    with patch(
        "mcp.client.stdio.stdio_client", mock_stdio_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPCallTool(
            transport=TransportType.STDIO,
            command="test-command",
            tool_name="test-tool",
            tool_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["is_error"] is False
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_mcp_call_tool_sse(mock_context, mock_sse_client):
    """Test MCPCallTool with SSE transport."""
    with patch(
        "mcp.client.sse.sse_client", mock_sse_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPCallTool(
            transport=TransportType.SSE,
            url="http://localhost:8080/sse",
            tool_name="test-tool",
            tool_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["is_error"] is False
        assert len(result["content"]) == 1


@pytest.mark.asyncio
async def test_mcp_call_tool_http(mock_context, mock_http_client):
    """Test MCPCallTool with HTTP transport."""
    with patch(
        "mcp.client.streamable_http.streamable_http_client", mock_http_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPCallTool(
            transport=TransportType.HTTP,
            url="http://localhost:8080/mcp",
            tool_name="test-tool",
            tool_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["is_error"] is False
        assert len(result["content"]) == 1


@pytest.mark.asyncio
async def test_mcp_read_resource_stdio(mock_context, mock_stdio_client):
    """Test MCPReadResource with stdio transport."""
    with patch(
        "mcp.client.stdio.stdio_client", mock_stdio_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPReadResource(
            transport=TransportType.STDIO,
            command="test-command",
            resource_uri="file:///test/resource",
        )
        result = await node.process(mock_context)

        assert len(result["contents"]) == 1
        assert result["contents"][0]["uri"] == "file:///test"
        assert result["contents"][0]["text"] == "resource content"


@pytest.mark.asyncio
async def test_mcp_read_resource_sse(mock_context, mock_sse_client):
    """Test MCPReadResource with SSE transport."""
    with patch(
        "mcp.client.sse.sse_client", mock_sse_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPReadResource(
            transport=TransportType.SSE,
            url="http://localhost:8080/sse",
            resource_uri="file:///test/resource",
        )
        result = await node.process(mock_context)

        assert len(result["contents"]) == 1


@pytest.mark.asyncio
async def test_mcp_read_resource_http(mock_context, mock_http_client):
    """Test MCPReadResource with HTTP transport."""
    with patch(
        "mcp.client.streamable_http.streamable_http_client", mock_http_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPReadResource(
            transport=TransportType.HTTP,
            url="http://localhost:8080/mcp",
            resource_uri="file:///test/resource",
        )
        result = await node.process(mock_context)

        assert len(result["contents"]) == 1


@pytest.mark.asyncio
async def test_mcp_get_prompt_stdio(mock_context, mock_stdio_client):
    """Test MCPGetPrompt with stdio transport."""
    with patch(
        "mcp.client.stdio.stdio_client", mock_stdio_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPGetPrompt(
            transport=TransportType.STDIO,
            command="test-command",
            prompt_name="test-prompt",
            prompt_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["description"] == "Test prompt description"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_mcp_get_prompt_sse(mock_context, mock_sse_client):
    """Test MCPGetPrompt with SSE transport."""
    with patch(
        "mcp.client.sse.sse_client", mock_sse_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPGetPrompt(
            transport=TransportType.SSE,
            url="http://localhost:8080/sse",
            prompt_name="test-prompt",
            prompt_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["description"] == "Test prompt description"
        assert len(result["messages"]) == 1


@pytest.mark.asyncio
async def test_mcp_get_prompt_http(mock_context, mock_http_client):
    """Test MCPGetPrompt with HTTP transport."""
    with patch(
        "mcp.client.streamable_http.streamable_http_client", mock_http_client
    ), patch("mcp.ClientSession", MockClientSession):
        node = MCPGetPrompt(
            transport=TransportType.HTTP,
            url="http://localhost:8080/mcp",
            prompt_name="test-prompt",
            prompt_arguments={"arg1": "value1"},
        )
        result = await node.process(mock_context)

        assert result["description"] == "Test prompt description"
        assert len(result["messages"]) == 1


# Test node field defaults
def test_mcp_stdio_client_defaults():
    """Test MCPStdioClient has correct default values."""
    node = MCPStdioClient()
    assert node.command == ""
    assert node.args == []
    assert node.env == {}
    assert node.working_directory == ""


def test_mcp_sse_client_defaults():
    """Test MCPSSEClient has correct default values."""
    node = MCPSSEClient()
    assert node.url == ""
    assert node.headers == {}
    assert node.timeout == 30.0


def test_mcp_http_client_defaults():
    """Test MCPStreamableHTTPClient has correct default values."""
    node = MCPStreamableHTTPClient()
    assert node.url == ""


def test_mcp_call_tool_defaults():
    """Test MCPCallTool has correct default values."""
    node = MCPCallTool()
    assert node.transport == TransportType.STDIO
    assert node.command == ""
    assert node.args == []
    assert node.env == {}
    assert node.working_directory == ""
    assert node.url == ""
    assert node.headers == {}
    assert node.tool_name == ""
    assert node.tool_arguments == {}


def test_mcp_read_resource_defaults():
    """Test MCPReadResource has correct default values."""
    node = MCPReadResource()
    assert node.transport == TransportType.STDIO
    assert node.resource_uri == ""


def test_mcp_get_prompt_defaults():
    """Test MCPGetPrompt has correct default values."""
    node = MCPGetPrompt()
    assert node.transport == TransportType.STDIO
    assert node.prompt_name == ""
    assert node.prompt_arguments == {}


# Test prompts with arguments
@pytest.mark.asyncio
async def test_mcp_stdio_client_prompts_with_arguments(mock_context, mock_stdio_client):
    """Test MCPStdioClient correctly parses prompts with arguments."""
    mock_session = MockClientSession()
    prompt_with_args = MockPrompt("complex-prompt", "A complex prompt")
    prompt_with_args.arguments = [
        MockPromptArgument("input", required=True),
        MockPromptArgument("context", required=False),
    ]
    mock_session.list_prompts = AsyncMock(
        return_value=MockListPromptsResult([prompt_with_args])
    )

    with patch(
        "mcp.client.stdio.stdio_client", mock_stdio_client
    ), patch("mcp.ClientSession", lambda *args, **kwargs: mock_session):
        node = MCPStdioClient(command="test-command")
        result = await node.process(mock_context)

        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "complex-prompt"
        assert result["prompts"][0]["arguments"] is not None
        assert len(result["prompts"][0]["arguments"]) == 2
        assert result["prompts"][0]["arguments"][0]["name"] == "input"
        assert result["prompts"][0]["arguments"][0]["required"] is True
