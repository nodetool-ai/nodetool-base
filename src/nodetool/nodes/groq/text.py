from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class GroqModel(str, Enum):
    """Available Groq AI models."""

    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B = "llama-3.1-8b-instant"
    LLAMA_3_2_1B = "llama-3.2-1b-preview"
    LLAMA_3_2_3B = "llama-3.2-3b-preview"
    LLAMA_GUARD_3_8B = "llama-guard-3-8b"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_2_9B = "gemma2-9b-it"
    QWEN_QWQ_32B = "qwen-qwq-32b"
    DEEPSEEK_R1_DISTILL = "deepseek-r1-distill-llama-70b"


class ChatComplete(BaseNode):
    """
    Generate text using Groq's ultra-fast LPU inference.
    groq, chat, ai, text generation, llm, completion, fast, low latency

    Uses Groq's Language Processing Unit (LPU) for extremely fast inference
    with open-source models like Llama, Mixtral, and Gemma.
    Requires a Groq API key.

    Use cases:
    - Ultra-fast text generation for real-time applications
    - Build responsive conversational AI applications
    - Code generation and assistance
    - Content creation with minimal latency
    """

    _expose_as_tool: ClassVar[bool] = True

    model: GroqModel = Field(
        default=GroqModel.LLAMA_3_3_70B,
        description="The Groq model to use for generation",
    )

    prompt: str = Field(default="", description="The prompt for text generation")

    system_prompt: str = Field(
        default="",
        description="Optional system prompt to guide the model's behavior",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
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
        Generate a chat completion using Groq's LPU inference.

        Args:
            context: The processing context.

        Returns:
            str: The generated text response.
        """
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        api_key = await context.get_secret("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not configured")

        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})

        response = await client.chat.completions.create(
            model=self.model.value,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response or not response.choices:
            raise ValueError("No response received from Groq API")

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]
