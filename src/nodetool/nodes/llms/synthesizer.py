from __future__ import annotations

import re
from jinja2 import Environment, BaseLoader
from pydantic import Field

from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    LanguageModel,
    Provider,
)
from nodetool.chat.providers import Chunk
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class Synthesizer(BaseNode):
    """Generate text from a Jinja2 prompt using dynamic properties."""

    _is_dynamic = True

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for generation",
    )
    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt for the LLM",
    )
    prompt: str = Field(
        default="",
        description="Prompt template rendered with dynamic properties",
    )
    max_tokens: int = Field(default=4096, ge=1, le=100000)

    @classmethod
    def get_title(cls) -> str:
        return "Synthesizer"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]

    async def process(self, context: ProcessingContext) -> str:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        env = Environment(loader=BaseLoader())

        template_str = self.prompt
        for var in re.findall(r"{{\s*([^|}]+)", template_str):
            template_str = template_str.replace(var, var.lower())

        template = env.from_string(template_str)
        properties = {k.lower(): v for k, v in self._dynamic_properties.items()}
        user_prompt = template.render(**properties)

        messages = [
            Message(role="system", content=self.system),
            Message(role="user", content=[MessageTextContent(text=user_prompt)]),
        ]

        result = ""
        async for chunk in context.generate_messages(
            messages=messages,
            provider=self.model.provider,
            model=self.model.id,
            node_id=self.id,
            max_tokens=self.max_tokens,
        ):
            if isinstance(chunk, Chunk):
                context.post_message(
                    Chunk(
                        node_id=self.id,
                        content=chunk.content,
                        content_type=chunk.content_type,
                    )
                )
                result += chunk.content
        return result
