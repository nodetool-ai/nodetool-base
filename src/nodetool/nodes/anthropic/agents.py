from enum import Enum
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

        # Get workspace path for the agent
        workspace_path = context.resolve_workspace_path("")

        # Get Anthropic API key from nodetool settings
        api_key = await context.get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not configured in NodeTool settings."
            )

        # Configure agent options using official SDK parameters
        options = ClaudeAgentOptions(
            model=self.model.id,
            system_prompt=self.system_prompt if self.system_prompt else None,
            max_turns=self.max_turns,
            cwd=str(workspace_path) if workspace_path else None,
            allowed_tools=self.allowed_tools,
            permission_mode=self.permission_mode.value,  # type: ignore[arg-type]
            env={"ANTHROPIC_API_KEY": api_key},  # Pass API key to the SDK
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
            error_msg = f"Claude Agent error: {str(e)}"
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
