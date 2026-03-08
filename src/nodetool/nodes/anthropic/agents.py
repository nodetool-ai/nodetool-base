from enum import Enum
from pathlib import Path
from typing import ClassVar, TypedDict, List
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate, Chunk
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import LanguageModel

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

log = get_logger(__name__)


class PermissionMode(str, Enum):
    """Permission modes for Claude Agent tool usage."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"


class ClaudeAgent(BaseNode):
    """
    Run Claude as an agent in a sandboxed environment with tool use capabilities.
    claude, agent, ai, anthropic, sandbox, assistant

    Uses the Claude Agent SDK to run Claude with access to tools in a secure sandbox.
    The agent can execute commands, read/write files, and use various tools while
    maintaining security through sandbox isolation.

    Use cases:
    - Automated coding and debugging tasks
    - File manipulation and analysis
    - Complex multi-step workflows
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
        description="List of tools the agent is allowed to use (e.g., 'Read', 'Write', 'Bash').",
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
        }

        if self.use_claude_credentials:
            # Use Claude Code's credentials file for authentication
            credentials_path = Path.home() / ".claude" / ".credentials.json"
            if not credentials_path.exists():
                raise ValueError(
                    f"Claude credentials file not found at {credentials_path}. "
                    "Please log in with Claude Code first."
                )
            log.info(
                "ClaudeAgent using Claude Code credentials "
                f"from {credentials_path}"
            )
        else:
            # Use ANTHROPIC_API_KEY from nodetool settings (optional —
            # if absent, the Claude CLI falls back to the user's subscription)
            api_key = await context.get_secret("ANTHROPIC_API_KEY")
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
                log.info("ClaudeAgent using ANTHROPIC_API_KEY from settings")
            else:
                log.info(
                    "ClaudeAgent: no ANTHROPIC_API_KEY configured, "
                    "falling back to Claude CLI subscription"
                )

        # Collect stderr for debugging
        stderr_lines: list[str] = []

        # Configure agent options using official SDK parameters
        options = ClaudeAgentOptions(
            model=self.model.id,
            system_prompt=self.system_prompt if self.system_prompt else None,
            max_turns=self.max_turns,
            cwd=str(workspace_path) if workspace_path else None,
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode.value,  # type: ignore[arg-type]
            env=env,
            stderr=lambda line: stderr_lines.append(line),
        )

        try:
            # Collect the full response text
            full_text = ""

            # Use the query() async iterator - the official SDK pattern
            async for message in query(prompt=self.prompt, options=options):
                if isinstance(message, AssistantMessage):
                    # Extract text content from the message
                    for content in message.content:
                        if isinstance(content, TextBlock):
                            text_chunk = content.text
                            full_text += text_chunk

                            # Emit as chunk for streaming
                            chunk = Chunk(node_id=self.id, content=text_chunk)
                            await outputs.emit("chunk", chunk)

                            # Also log the progress
                            context.post_message(
                                LogUpdate(
                                    node_id=self.id,
                                    node_name=self.get_title(),
                                    content=text_chunk,
                                    severity="info",
                                )
                            )

            # Emit the final full text
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
