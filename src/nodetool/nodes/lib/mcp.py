"""
MCP (Model Context Protocol) client nodes for different transports.

This module provides nodes to connect to MCP servers using various transports:
- Stdio: For local subprocess-based servers
- SSE: For HTTP Server-Sent Events based servers
- Streamable HTTP: For HTTP streaming based servers

MCP specification: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

import sys
from enum import Enum
from typing import Any, TypedDict

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

logger = get_logger(__name__)


class MCPTool(TypedDict):
    """Represents an MCP tool."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


class MCPResource(TypedDict):
    """Represents an MCP resource."""

    name: str
    uri: str
    description: str | None
    mime_type: str | None


class MCPPrompt(TypedDict):
    """Represents an MCP prompt."""

    name: str
    description: str | None
    arguments: list[dict[str, Any]] | None


class MCPToolResult(TypedDict):
    """Result from calling an MCP tool."""

    content: list[dict[str, Any]]
    is_error: bool


class MCPResourceContent(TypedDict):
    """Content from reading an MCP resource."""

    uri: str
    mime_type: str | None
    text: str | None
    blob: str | None


class MCPPromptMessage(TypedDict):
    """A message from an MCP prompt."""

    role: str
    content: Any


class MCPStdioClient(BaseNode):
    """
    Connect to an MCP server using stdio transport (subprocess).
    mcp, model context protocol, stdio, subprocess, tools, ai

    Use cases:
    - Connect to local MCP servers
    - Run command-line based MCP tools
    - Integrate with local AI tool servers
    """

    command: str = Field(
        default="",
        description="Command to execute (e.g., 'npx', 'python', 'node')",
    )
    args: list[str] = Field(
        default=[],
        description="Arguments to pass to the command",
    )
    env: dict[str, str] = Field(
        default={},
        description="Additional environment variables for the subprocess",
    )
    working_directory: str = Field(
        default="",
        description="Working directory for the subprocess (optional)",
    )

    class OutputType(TypedDict):
        tools: list[MCPTool]
        resources: list[MCPResource]
        prompts: list[MCPPrompt]
        server_name: str
        server_version: str

    @classmethod
    def get_title(cls) -> str:
        return "MCP Stdio Client"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["command", "args"]

    async def process(self, context: ProcessingContext) -> OutputType:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if not self.command:
            raise ValueError("Command is required")

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env if self.env else None,
            cwd=self.working_directory if self.working_directory else None,
        )

        async with stdio_client(params, errlog=sys.stderr) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                init_result = await session.initialize()

                # Get server info
                server_name = init_result.serverInfo.name
                server_version = init_result.serverInfo.version

                # List tools
                tools: list[MCPTool] = []
                try:
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list tools: {e}")

                # List resources
                resources: list[MCPResource] = []
                try:
                    resources_result = await session.list_resources()
                    for resource in resources_result.resources:
                        resources.append(
                            {
                                "name": resource.name,
                                "uri": str(resource.uri),
                                "description": resource.description,
                                "mime_type": resource.mimeType,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list resources: {e}")

                # List prompts
                prompts: list[MCPPrompt] = []
                try:
                    prompts_result = await session.list_prompts()
                    for prompt in prompts_result.prompts:
                        args = None
                        if prompt.arguments:
                            args = [
                                {
                                    "name": arg.name,
                                    "description": arg.description,
                                    "required": arg.required,
                                }
                                for arg in prompt.arguments
                            ]
                        prompts.append(
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": args,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list prompts: {e}")

                return {
                    "tools": tools,
                    "resources": resources,
                    "prompts": prompts,
                    "server_name": server_name,
                    "server_version": server_version,
                }


class MCPSSEClient(BaseNode):
    """
    Connect to an MCP server using SSE (Server-Sent Events) transport.
    mcp, model context protocol, sse, http, server-sent events, tools, ai

    Use cases:
    - Connect to remote MCP servers over HTTP
    - Integrate with cloud-hosted MCP services
    - Connect to web-based AI tool servers
    """

    url: str = Field(
        default="",
        description="URL of the MCP SSE server endpoint",
    )
    headers: dict[str, str] = Field(
        default={},
        description="Additional HTTP headers for the connection",
    )
    timeout: float = Field(
        default=30.0,
        description="Connection timeout in seconds",
    )

    class OutputType(TypedDict):
        tools: list[MCPTool]
        resources: list[MCPResource]
        prompts: list[MCPPrompt]
        server_name: str
        server_version: str

    @classmethod
    def get_title(cls) -> str:
        return "MCP SSE Client"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["url"]

    async def process(self, context: ProcessingContext) -> OutputType:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        if not self.url:
            raise ValueError("URL is required")

        async with sse_client(
            url=self.url,
            headers=self.headers if self.headers else None,
            timeout=self.timeout,
        ) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                init_result = await session.initialize()

                # Get server info
                server_name = init_result.serverInfo.name
                server_version = init_result.serverInfo.version

                # List tools
                tools: list[MCPTool] = []
                try:
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list tools: {e}")

                # List resources
                resources: list[MCPResource] = []
                try:
                    resources_result = await session.list_resources()
                    for resource in resources_result.resources:
                        resources.append(
                            {
                                "name": resource.name,
                                "uri": str(resource.uri),
                                "description": resource.description,
                                "mime_type": resource.mimeType,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list resources: {e}")

                # List prompts
                prompts: list[MCPPrompt] = []
                try:
                    prompts_result = await session.list_prompts()
                    for prompt in prompts_result.prompts:
                        args = None
                        if prompt.arguments:
                            args = [
                                {
                                    "name": arg.name,
                                    "description": arg.description,
                                    "required": arg.required,
                                }
                                for arg in prompt.arguments
                            ]
                        prompts.append(
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": args,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list prompts: {e}")

                return {
                    "tools": tools,
                    "resources": resources,
                    "prompts": prompts,
                    "server_name": server_name,
                    "server_version": server_version,
                }


class MCPStreamableHTTPClient(BaseNode):
    """
    Connect to an MCP server using Streamable HTTP transport.
    mcp, model context protocol, http, streaming, tools, ai

    Use cases:
    - Connect to MCP servers that support HTTP streaming
    - Alternative to SSE for HTTP-based connections
    - Connect to modern MCP server implementations
    """

    url: str = Field(
        default="",
        description="URL of the MCP HTTP streaming endpoint",
    )

    class OutputType(TypedDict):
        tools: list[MCPTool]
        resources: list[MCPResource]
        prompts: list[MCPPrompt]
        server_name: str
        server_version: str

    @classmethod
    def get_title(cls) -> str:
        return "MCP HTTP Client"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["url"]

    async def process(self, context: ProcessingContext) -> OutputType:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        if not self.url:
            raise ValueError("URL is required")

        async with streamable_http_client(url=self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the session
                init_result = await session.initialize()

                # Get server info
                server_name = init_result.serverInfo.name
                server_version = init_result.serverInfo.version

                # List tools
                tools: list[MCPTool] = []
                try:
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list tools: {e}")

                # List resources
                resources: list[MCPResource] = []
                try:
                    resources_result = await session.list_resources()
                    for resource in resources_result.resources:
                        resources.append(
                            {
                                "name": resource.name,
                                "uri": str(resource.uri),
                                "description": resource.description,
                                "mime_type": resource.mimeType,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list resources: {e}")

                # List prompts
                prompts: list[MCPPrompt] = []
                try:
                    prompts_result = await session.list_prompts()
                    for prompt in prompts_result.prompts:
                        args = None
                        if prompt.arguments:
                            args = [
                                {
                                    "name": arg.name,
                                    "description": arg.description,
                                    "required": arg.required,
                                }
                                for arg in prompt.arguments
                            ]
                        prompts.append(
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": args,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not list prompts: {e}")

                return {
                    "tools": tools,
                    "resources": resources,
                    "prompts": prompts,
                    "server_name": server_name,
                    "server_version": server_version,
                }


class TransportType(str, Enum):
    """MCP transport type."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPCallTool(BaseNode):
    """
    Call a tool on an MCP server.
    mcp, model context protocol, tool, call, execute, ai

    Use cases:
    - Execute MCP tools with arguments
    - Integrate MCP tool capabilities into workflows
    - Run AI-powered tools from MCP servers
    """

    transport: TransportType = Field(
        default=TransportType.STDIO,
        description="Transport type to use for connection",
    )

    # Stdio parameters
    command: str = Field(
        default="",
        description="Command to execute (for stdio transport)",
    )
    args: list[str] = Field(
        default=[],
        description="Arguments for the command (for stdio transport)",
    )
    env: dict[str, str] = Field(
        default={},
        description="Environment variables (for stdio transport)",
    )
    working_directory: str = Field(
        default="",
        description="Working directory (for stdio transport)",
    )

    # HTTP parameters
    url: str = Field(
        default="",
        description="URL of the MCP server (for SSE/HTTP transport)",
    )
    headers: dict[str, str] = Field(
        default={},
        description="HTTP headers (for SSE transport)",
    )

    # Tool parameters
    tool_name: str = Field(
        default="",
        description="Name of the tool to call",
    )
    tool_arguments: dict[str, Any] = Field(
        default={},
        description="Arguments to pass to the tool",
    )

    @classmethod
    def get_title(cls) -> str:
        return "MCP Call Tool"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["transport", "tool_name", "tool_arguments"]

    async def process(self, context: ProcessingContext) -> MCPToolResult:
        from mcp import ClientSession

        if not self.tool_name:
            raise ValueError("Tool name is required")

        if self.transport == TransportType.STDIO:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not self.command:
                raise ValueError("Command is required for stdio transport")

            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env if self.env else None,
                cwd=self.working_directory if self.working_directory else None,
            )

            async with stdio_client(params, errlog=sys.stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        self.tool_name, self.tool_arguments or None
                    )
                    return {
                        "content": [c.model_dump() for c in result.content],
                        "is_error": result.isError,
                    }

        elif self.transport == TransportType.SSE:
            from mcp.client.sse import sse_client

            if not self.url:
                raise ValueError("URL is required for SSE transport")

            async with sse_client(
                url=self.url,
                headers=self.headers if self.headers else None,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        self.tool_name, self.tool_arguments or None
                    )
                    return {
                        "content": [c.model_dump() for c in result.content],
                        "is_error": result.isError,
                    }

        else:  # HTTP
            from mcp.client.streamable_http import streamable_http_client

            if not self.url:
                raise ValueError("URL is required for HTTP transport")

            async with streamable_http_client(url=self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        self.tool_name, self.tool_arguments or None
                    )
                    return {
                        "content": [c.model_dump() for c in result.content],
                        "is_error": result.isError,
                    }


class MCPReadResource(BaseNode):
    """
    Read a resource from an MCP server.
    mcp, model context protocol, resource, read, fetch, ai

    Use cases:
    - Read data from MCP resources
    - Fetch content from MCP servers
    - Access structured data via MCP protocol
    """

    transport: TransportType = Field(
        default=TransportType.STDIO,
        description="Transport type to use for connection",
    )

    # Stdio parameters
    command: str = Field(
        default="",
        description="Command to execute (for stdio transport)",
    )
    args: list[str] = Field(
        default=[],
        description="Arguments for the command (for stdio transport)",
    )
    env: dict[str, str] = Field(
        default={},
        description="Environment variables (for stdio transport)",
    )
    working_directory: str = Field(
        default="",
        description="Working directory (for stdio transport)",
    )

    # HTTP parameters
    url: str = Field(
        default="",
        description="URL of the MCP server (for SSE/HTTP transport)",
    )
    headers: dict[str, str] = Field(
        default={},
        description="HTTP headers (for SSE transport)",
    )

    # Resource parameters
    resource_uri: str = Field(
        default="",
        description="URI of the resource to read",
    )

    class OutputType(TypedDict):
        contents: list[MCPResourceContent]

    @classmethod
    def get_title(cls) -> str:
        return "MCP Read Resource"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["transport", "resource_uri"]

    async def process(self, context: ProcessingContext) -> OutputType:
        from pydantic import AnyUrl

        from mcp import ClientSession

        if not self.resource_uri:
            raise ValueError("Resource URI is required")

        uri = AnyUrl(self.resource_uri)

        if self.transport == TransportType.STDIO:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not self.command:
                raise ValueError("Command is required for stdio transport")

            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env if self.env else None,
                cwd=self.working_directory if self.working_directory else None,
            )

            async with stdio_client(params, errlog=sys.stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.read_resource(uri)
                    contents: list[MCPResourceContent] = []
                    for content in result.contents:
                        contents.append(
                            {
                                "uri": str(content.uri),
                                "mime_type": content.mimeType,
                                "text": getattr(content, "text", None),
                                "blob": getattr(content, "blob", None),
                            }
                        )
                    return {"contents": contents}

        elif self.transport == TransportType.SSE:
            from mcp.client.sse import sse_client

            if not self.url:
                raise ValueError("URL is required for SSE transport")

            async with sse_client(
                url=self.url,
                headers=self.headers if self.headers else None,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.read_resource(uri)
                    contents: list[MCPResourceContent] = []
                    for content in result.contents:
                        contents.append(
                            {
                                "uri": str(content.uri),
                                "mime_type": content.mimeType,
                                "text": getattr(content, "text", None),
                                "blob": getattr(content, "blob", None),
                            }
                        )
                    return {"contents": contents}

        else:  # HTTP
            from mcp.client.streamable_http import streamable_http_client

            if not self.url:
                raise ValueError("URL is required for HTTP transport")

            async with streamable_http_client(url=self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.read_resource(uri)
                    contents: list[MCPResourceContent] = []
                    for content in result.contents:
                        contents.append(
                            {
                                "uri": str(content.uri),
                                "mime_type": content.mimeType,
                                "text": getattr(content, "text", None),
                                "blob": getattr(content, "blob", None),
                            }
                        )
                    return {"contents": contents}


class MCPGetPrompt(BaseNode):
    """
    Get a prompt from an MCP server.
    mcp, model context protocol, prompt, template, ai

    Use cases:
    - Retrieve prompt templates from MCP servers
    - Get structured prompts with arguments
    - Access AI prompt configurations
    """

    transport: TransportType = Field(
        default=TransportType.STDIO,
        description="Transport type to use for connection",
    )

    # Stdio parameters
    command: str = Field(
        default="",
        description="Command to execute (for stdio transport)",
    )
    args: list[str] = Field(
        default=[],
        description="Arguments for the command (for stdio transport)",
    )
    env: dict[str, str] = Field(
        default={},
        description="Environment variables (for stdio transport)",
    )
    working_directory: str = Field(
        default="",
        description="Working directory (for stdio transport)",
    )

    # HTTP parameters
    url: str = Field(
        default="",
        description="URL of the MCP server (for SSE/HTTP transport)",
    )
    headers: dict[str, str] = Field(
        default={},
        description="HTTP headers (for SSE transport)",
    )

    # Prompt parameters
    prompt_name: str = Field(
        default="",
        description="Name of the prompt to get",
    )
    prompt_arguments: dict[str, str] = Field(
        default={},
        description="Arguments to pass to the prompt",
    )

    class OutputType(TypedDict):
        description: str | None
        messages: list[MCPPromptMessage]

    @classmethod
    def get_title(cls) -> str:
        return "MCP Get Prompt"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["transport", "prompt_name", "prompt_arguments"]

    async def process(self, context: ProcessingContext) -> OutputType:
        from mcp import ClientSession

        if not self.prompt_name:
            raise ValueError("Prompt name is required")

        if self.transport == TransportType.STDIO:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not self.command:
                raise ValueError("Command is required for stdio transport")

            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env if self.env else None,
                cwd=self.working_directory if self.working_directory else None,
            )

            async with stdio_client(params, errlog=sys.stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.get_prompt(
                        self.prompt_name, self.prompt_arguments or None
                    )
                    messages: list[MCPPromptMessage] = []
                    for msg in result.messages:
                        content = msg.content
                        if hasattr(content, "model_dump"):
                            content = content.model_dump()
                        messages.append(
                            {
                                "role": msg.role,
                                "content": content,
                            }
                        )
                    return {
                        "description": result.description,
                        "messages": messages,
                    }

        elif self.transport == TransportType.SSE:
            from mcp.client.sse import sse_client

            if not self.url:
                raise ValueError("URL is required for SSE transport")

            async with sse_client(
                url=self.url,
                headers=self.headers if self.headers else None,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.get_prompt(
                        self.prompt_name, self.prompt_arguments or None
                    )
                    messages: list[MCPPromptMessage] = []
                    for msg in result.messages:
                        content = msg.content
                        if hasattr(content, "model_dump"):
                            content = content.model_dump()
                        messages.append(
                            {
                                "role": msg.role,
                                "content": content,
                            }
                        )
                    return {
                        "description": result.description,
                        "messages": messages,
                    }

        else:  # HTTP
            from mcp.client.streamable_http import streamable_http_client

            if not self.url:
                raise ValueError("URL is required for HTTP transport")

            async with streamable_http_client(url=self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.get_prompt(
                        self.prompt_name, self.prompt_arguments or None
                    )
                    messages: list[MCPPromptMessage] = []
                    for msg in result.messages:
                        content = msg.content
                        if hasattr(content, "model_dump"):
                            content = content.model_dump()
                        messages.append(
                            {
                                "role": msg.role,
                                "content": content,
                            }
                        )
                    return {
                        "description": result.description,
                        "messages": messages,
                    }
