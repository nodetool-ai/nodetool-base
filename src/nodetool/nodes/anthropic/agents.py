import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TypedDict, Union
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate, Chunk
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    tool,
    create_sdk_mcp_server,
)

log = get_logger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """Convert a node title to a valid MCP tool name (snake_case, max 64 chars)."""
    if not isinstance(name, str) or not name.strip():
        return "control_node"
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"([a-z])([A-Z])", r"\1_\2", sanitized)
    sanitized = sanitized.lower()
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return (sanitized or "control_node")[:64]


def _build_tool_schema(node_info: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON Schema dict from controlled node info for the @tool decorator."""
    actions = node_info.get("control_actions", {})
    run_action = actions.get("run", {})
    raw_properties = run_action.get("properties", {})

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    if isinstance(raw_properties, dict):
        for name, prop_schema in raw_properties.items():
            if isinstance(prop_schema, dict):
                schema["properties"][name] = dict(prop_schema)
            else:
                schema["properties"][name] = {
                    "type": "string",
                    "description": str(prop_schema),
                }

    return schema


class PermissionMode(str, Enum):
    """Permission modes for Claude Agent tool usage."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"


class ClaudeAgent(BaseNode):
    """
    Run Claude as an agent with tool use capabilities and control edge support.
    claude, agent, ai, anthropic, sandbox, assistant

    Uses the Claude Agent SDK to run Claude with access to built-in tools
    (Read, Write, Bash, etc.) and any nodes connected via control edges.
    Control edges let the agent trigger other workflow nodes as tools,
    passing parameters and receiving results — the same pattern as the
    standard Agent node.

    Use cases:
    - Automated coding and debugging tasks
    - File manipulation and analysis
    - Complex multi-step workflows with node tools
    - Research and data gathering
    """

    _is_dynamic: ClassVar[bool] = False
    _supports_dynamic_outputs: ClassVar[bool] = False

    prompt: str = Field(
        default="",
        description="The task or question for the Claude agent to work on.",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="The Claude compatible model to use for the agent.",
    )

    system_prompt: str = Field(
        default="",
        description="Optional system prompt to guide the agent's behavior.",
    )

    max_turns: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of turns the agent can take.",
    )

    allowed_tools: list[str] = Field(
        default=["Read", "Write", "Bash"],
        description="List of built-in SDK tools the agent may use (e.g., 'Read', 'Write', 'Bash'). Nodes connected via control edges are always available.",
    )

    permission_mode: PermissionMode = Field(
        default=PermissionMode.ACCEPT_EDITS,
        description="Permission mode for tool usage.",
    )

    use_claude_credentials: bool = Field(
        default=False,
        description="Use Claude Code credentials file (~/.claude/.credentials.json) instead of the ANTHROPIC_API_KEY secret. Requires an active Claude Max/Pro subscription.",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    class OutputType(TypedDict):
        text: str
        chunk: Chunk

    @classmethod
    def return_type(cls):
        return cls.OutputType

    # ------------------------------------------------------------------
    # Control-edge tool building
    # ------------------------------------------------------------------

    def _build_control_tools(
        self, context: ProcessingContext
    ) -> tuple[Any | None, list[str]]:
        """Discover controlled nodes and create an in-process MCP server
        exposing each one as a custom tool.

        Returns (mcp_server_or_None, list_of_tool_fqns).
        """
        controlled_info = context.get_controlled_nodes_info(self.id)
        if not controlled_info:
            return None, []

        sdk_tools = []
        tool_fqns: list[str] = []

        for target_id, info in controlled_info.items():
            tool_name = _sanitize_tool_name(info.get("node_title", target_id))
            node_desc = info.get("node_description", "")
            description = node_desc or f"Run {info.get('node_title', target_id)}"
            input_schema = _build_tool_schema(info)

            _target_id = target_id
            _info = info

            @tool(tool_name, description, input_schema)
            async def _node_tool(
                args: dict[str, Any],
                _tid: str = _target_id,
                _ninfo: dict[str, Any] = _info,
            ) -> dict[str, Any]:
                result_text = await self._run_controlled_node(
                    context, _tid, _ninfo, args
                )
                return {"content": [{"type": "text", "text": result_text}]}

            sdk_tools.append(_node_tool)
            fqn = f"mcp__nodetool-nodes__{tool_name}"
            tool_fqns.append(fqn)

            prop_names = list(
                (input_schema.get("properties") or {}).keys()
            )
            log.info(
                "ClaudeAgent control tool registered: target=%s tool=%s params=%s",
                _target_id,
                tool_name,
                prop_names,
            )

        server = create_sdk_mcp_server(
            name="nodetool-nodes",
            version="1.0.0",
            tools=sdk_tools,
        )
        return server, tool_fqns

    # ------------------------------------------------------------------
    # Controlled node execution helpers
    # ------------------------------------------------------------------

    def _apply_property_overrides(
        self, target_node: Any, node_title: str, args: dict[str, Any]
    ) -> None:
        """Apply property overrides from tool arguments with enum coercion."""
        for name, value in args.items():
            if hasattr(target_node, name):
                try:
                    field_info = target_node.model_fields.get(name)
                    if field_info and isinstance(value, str):
                        import enum as _enum
                        annotation = field_info.annotation
                        origin = getattr(annotation, "__origin__", None)
                        if origin is type(None) or origin is Union:
                            for arg in getattr(annotation, "__args__", ()):
                                if isinstance(arg, type) and issubclass(arg, _enum.Enum):
                                    annotation = arg
                                    break
                        if isinstance(annotation, type) and issubclass(annotation, _enum.Enum):
                            value = annotation(value)
                    setattr(target_node, name, value)
                except Exception as exc:
                    log.warning(
                        "Failed to set property %s on %s: %s", name, node_title, exc
                    )

    @staticmethod
    def _format_result(result: Any, node_title: str) -> str:
        """Format a node result as text for the agent."""
        if result is None:
            return f"{node_title} completed (no output)"
        elif hasattr(result, "model_dump"):
            return json.dumps(result.model_dump(), indent=2, default=str)
        elif isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        elif isinstance(result, (list, tuple)):
            return json.dumps(result, indent=2, default=str)
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    async def _run_controlled_node(
        self,
        context: ProcessingContext,
        target_id: str,
        node_info: dict[str, Any],
        args: dict[str, Any],
    ) -> str:
        """Execute a controlled node synchronously and return formatted result text."""
        if context.graph is None:
            return "Error: no workflow graph available"

        target_node = context.graph.find_node(target_id)
        if target_node is None:
            return f"Error: node {target_id} not found in graph"

        node_title = node_info.get("node_title", target_id)
        self._apply_property_overrides(target_node, node_title, args)

        result = await target_node.process(context)
        result_text = self._format_result(result, node_title)

        log.info(
            "ClaudeAgent control tool executed: target=%s result_len=%d",
            target_id,
            len(result_text),
        )
        return result_text

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def run(
        self, context: ProcessingContext, inputs: NodeInputs, outputs: NodeOutputs
    ) -> None:  # type: ignore[override]
        if not self.prompt.strip():
            raise RuntimeError("Prompt is required")

        # Get workspace path for the agent, fall back to temp dir
        try:
            workspace_path = context.resolve_workspace_path("")
        except PermissionError:
            import tempfile

            workspace_path = tempfile.mkdtemp(prefix="claude_agent_")

        env: dict[str, str] = {
            "CLAUDECODE": "",  # Allow launching from within Claude Code sessions
            "MCP_TOOL_TIMEOUT": "60000",  # 1 minute for long-running node tools
        }

        if self.use_claude_credentials:
            credentials_path = Path.home() / ".claude" / ".credentials.json"
            if not credentials_path.exists():
                raise ValueError(
                    f"Claude credentials file not found at {credentials_path}. "
                    "Please log in with Claude Code first."
                )
            log.info(
                "ClaudeAgent using Claude Code credentials from %s",
                credentials_path,
            )
        else:
            api_key = await context.get_secret("ANTHROPIC_API_KEY")
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
                log.info("ClaudeAgent using ANTHROPIC_API_KEY from settings")
            else:
                log.info(
                    "ClaudeAgent: no ANTHROPIC_API_KEY configured, "
                    "falling back to Claude CLI subscription"
                )

        # Discover controlled nodes and build MCP tools
        mcp_server, control_tool_fqns = self._build_control_tools(context)

        # Collect stderr for debugging
        stderr_lines: list[str] = []

        # Build allowed_tools list including control tool FQNs
        all_allowed = list(self.allowed_tools) + control_tool_fqns

        # Configure agent options
        options = ClaudeAgentOptions(
            model=self.model.id,
            system_prompt=self.system_prompt if self.system_prompt else None,
            max_turns=self.max_turns,
            cwd=str(workspace_path) if workspace_path else None,
            allowed_tools=all_allowed,
            permission_mode=self.permission_mode.value,  # type: ignore[arg-type]
            env=env,
            stderr=lambda line: stderr_lines.append(line),
        )

        # Add MCP server if we have control tools
        if mcp_server is not None:
            options.mcp_servers = {"nodetool-nodes": mcp_server}

        try:
            full_text = ""

            # MCP custom tools require streaming input mode (async generator)
            if mcp_server is not None:

                async def _prompt_stream():
                    yield {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": self.prompt,
                        },
                    }

                prompt_input: Any = _prompt_stream()
            else:
                prompt_input = self.prompt

            async for message in query(prompt=prompt_input, options=options):
                if isinstance(message, AssistantMessage):
                    for content in message.content:
                        if isinstance(content, TextBlock):
                            text_chunk = content.text
                            full_text += text_chunk

                            chunk = Chunk(node_id=self.id, content=text_chunk)
                            await outputs.emit("chunk", chunk)

                            context.post_message(
                                LogUpdate(
                                    node_id=self.id,
                                    node_name=self.get_title(),
                                    content=text_chunk,
                                    severity="info",
                                )
                            )

            await outputs.emit("text", full_text)

        except Exception as e:
            stderr_output = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
            error_msg = f"Claude Agent error: {str(e)}"
            if stderr_output:
                error_msg += f"\nStderr:\n{stderr_output}"
            log.error(error_msg)
            context.post_message(
                LogUpdate(
                    node_id=self.id,
                    node_name=self.get_title(),
                    content=error_msg,
                    severity="error",
                )
            )
            raise RuntimeError(error_msg) from e
