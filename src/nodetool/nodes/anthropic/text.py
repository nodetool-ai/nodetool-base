from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class ClaudeModel(str, Enum):
    """Available Claude models."""

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_HAIKU_4 = "claude-4-haiku-20250514"


class ChatComplete(BaseNode):
    """
    Generate text using Anthropic's Claude models.
    claude, anthropic, chat, ai, text generation, llm, completion

    Uses Anthropic's Claude models to generate responses from prompts.
    Requires an Anthropic API key.

    Use cases:
    - Generate text responses to prompts
    - Build conversational AI applications
    - Complex reasoning and analysis tasks
    - Code generation and explanation
    """

    _expose_as_tool: ClassVar[bool] = True

    model: ClaudeModel = Field(
        default=ClaudeModel.CLAUDE_3_5_SONNET,
        description="The Claude model to use for generation",
    )

    prompt: str = Field(default="", description="The prompt for text generation")

    system_prompt: str = Field(
        default="",
        description="Optional system prompt to guide the model's behavior",
    )

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Higher values make output more random.",
    )

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Generate a chat completion using Anthropic's Claude.

        Args:
            context: The processing context.

        Returns:
            str: The generated text response.
        """
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        api_key = await context.get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not configured")

        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)

        # Build the messages list
        messages = [{"role": "user", "content": self.prompt}]

        # Make the API call
        response = await client.messages.create(
            model=self.model.value,
            max_tokens=self.max_tokens,
            system=self.system_prompt if self.system_prompt else "",
            messages=messages,
            temperature=self.temperature,
        )

        # Extract text from the response
        if not response.content:
            return ""

        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "".join(text_parts)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]
