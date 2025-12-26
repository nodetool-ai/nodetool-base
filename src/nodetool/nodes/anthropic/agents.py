from enum import Enum
from typing import ClassVar, TypedDict
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate, Chunk
from nodetool.workflows.io import NodeInputs, NodeOutputs
from nodetool.config.logging_config import get_logger

from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk.types import (
    ClaudeAgentOptions,
    SandboxSettings,
    AssistantMessage,
    TextBlock,
)

log = get_logger(__name__)


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
    _client: ClaudeSDKClient | None = None

    class Model(str, Enum):
        CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
        CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
        CLAUDE_3_OPUS = "claude-3-opus-20240229"

    prompt: str = Field(
        default="",
        description="The task or question for the Claude agent to work on.",
    )

    model: Model = Field(
        default=Model.CLAUDE_3_5_SONNET,
        description="The Claude model to use for the agent.",
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

    enable_sandbox: bool = Field(
        default=True,
        description="Enable bash command sandboxing for security.",
    )

    auto_allow_bash: bool = Field(
        default=True,
        description="Automatically approve bash commands when sandboxed.",
    )

    api_key: str = Field(
        default="",
        description="Anthropic API key. If not provided, uses ANTHROPIC_API_KEY environment variable.",
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

        # Configure sandbox settings
        sandbox_settings: SandboxSettings = {
            "enabled": self.enable_sandbox,
            "autoAllowBashIfSandboxed": self.auto_allow_bash,
        }

        # Configure agent options
        options = ClaudeAgentOptions(
            model=self.model.value,
            system_prompt=self.system_prompt if self.system_prompt else None,
            max_turns=self.max_turns,
            sandbox=sandbox_settings,
            cwd=str(workspace_path) if workspace_path else None,
            permission_mode="acceptEdits",  # Auto-approve edit operations
        )

        # Set API key if provided
        if self.api_key:
            import os

            os.environ["ANTHROPIC_API_KEY"] = self.api_key

        try:
            # Create client and connect
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()

            # Send the prompt
            await self._client.query(self.prompt)

            # Collect the full response text
            full_text = ""

            # Stream the response messages
            async for message in self._client.receive_response():
                if isinstance(message, AssistantMessage):
                    # Extract text content from the message
                    for content in message.content:
                        if isinstance(content, TextBlock):
                            text_chunk = content.text
                            full_text += text_chunk

                            # Emit as chunk for streaming
                            chunk = Chunk(chunk=text_chunk)
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
        finally:
            # Disconnect the client
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception as e:
                    log.debug(f"Error disconnecting Claude Agent: {e}")

    async def finalize(self, context: ProcessingContext) -> None:  # type: ignore[override]
        """Clean up the Claude Agent client."""
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                log.debug(f"ClaudeAgent finalize: {e}")
