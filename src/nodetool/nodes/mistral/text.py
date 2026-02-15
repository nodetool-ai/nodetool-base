from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class MistralModel(str, Enum):
    """Available Mistral AI models."""

    MISTRAL_LARGE = "mistral-large-latest"
    MISTRAL_MEDIUM = "mistral-medium-latest"
    MISTRAL_SMALL = "mistral-small-latest"
    PIXTRAL_LARGE = "pixtral-large-latest"
    CODESTRAL = "codestral-latest"
    MINISTRAL_8B = "ministral-8b-latest"
    MINISTRAL_3B = "ministral-3b-latest"


class ChatComplete(BaseNode):
    """
    Generate text using Mistral AI's chat completion models.
    mistral, chat, ai, text generation, llm, completion

    Uses Mistral AI's chat models to generate responses from prompts.
    Requires a Mistral API key.

    Use cases:
    - Generate text responses to prompts
    - Build conversational AI applications
    - Code generation with Codestral
    - Multi-modal understanding with Pixtral
    """

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["MISTRAL_API_KEY"]

    model: MistralModel = Field(
        default=MistralModel.MISTRAL_SMALL,
        description="The Mistral model to use for generation",
    )

    prompt: str = Field(default="", description="The prompt for text generation")

    system_prompt: str = Field(
        default="",
        description="Optional system prompt to guide the model's behavior",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Higher values make output more random.",
    )

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Generate a chat completion using Mistral AI.

        Args:
            context: The processing context.

        Returns:
            str: The generated text response.
        """
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})

        response = await client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response or not response.choices:
            raise ValueError("No response received from Mistral API")

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]


class CodeComplete(BaseNode):
    """
    Generate code using Mistral AI's Codestral model.
    mistral, code, codestral, ai, programming, completion

    Uses Mistral AI's Codestral model specifically designed for code generation.
    Supports fill-in-the-middle (FIM) for code completion tasks.
    Requires a Mistral API key.

    Use cases:
    - Generate code from natural language descriptions
    - Complete partial code snippets
    - Fill in code between prefix and suffix
    - Automated code generation for various programming languages
    """

    _expose_as_tool: ClassVar[bool] = True
    _required_settings: ClassVar[list[str]] = ["MISTRAL_API_KEY"]

    prompt: str = Field(
        default="",
        description="The prompt or code prefix for generation",
    )

    suffix: str = Field(
        default="",
        description="Optional suffix for fill-in-the-middle completion",
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. Lower values for code generation.",
    )

    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Generate code using Mistral AI's Codestral model.

        Args:
            context: The processing context.

        Returns:
            str: The generated code.
        """
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Use fill-in-the-middle if suffix is provided
        if self.suffix:
            url = "https://api.mistral.ai/v1/fim/completions"
            payload = {
                "model": MistralModel.CODESTRAL.value,
                "prompt": self.prompt,
                "suffix": self.suffix,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        else:
            # Use regular chat completion for code generation
            url = "https://api.mistral.ai/v1/chat/completions"
            payload = {
                "model": MistralModel.CODESTRAL.value,
                "messages": [{"role": "user", "content": self.prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

        response = await context.http_post(url, headers=headers, json=payload)
        data = response.json()

        if not data or "choices" not in data:
            raise ValueError("No response received from Mistral API")

        content = data["choices"][0]["message"]["content"]
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "suffix"]
