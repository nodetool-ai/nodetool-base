from enum import Enum
from typing import ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import NPArray
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class EmbeddingModel(str, Enum):
    """Available Mistral AI embedding models."""

    MISTRAL_EMBED = "mistral-embed"


class Embedding(BaseNode):
    """
    Generate vector embeddings using Mistral AI.
    mistral, embeddings, vectors, semantic, similarity, search

    Uses Mistral AI's embedding model to create dense vector representations of text.
    These vectors capture semantic meaning, enabling:
    - Semantic search
    - Text clustering
    - Document classification
    - Recommendation systems
    - Measuring text similarity

    Requires a Mistral API key.
    """

    _expose_as_tool: ClassVar[bool] = True

    input: str = Field(default="", description="The text to embed")

    model: EmbeddingModel = Field(
        default=EmbeddingModel.MISTRAL_EMBED,
        description="The embedding model to use",
    )

    chunk_size: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Size of text chunks for embedding",
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        """
        Generate embeddings for the input text.

        Args:
            context: The processing context.

        Returns:
            NPArray: The embedding vector.
        """
        import numpy as np

        if not self.input:
            raise ValueError("Input text cannot be empty")

        api_key = await context.get_secret("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not configured")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

        # Chunk the input into smaller pieces if necessary
        chunks = [
            self.input[i : i + self.chunk_size]
            for i in range(0, len(self.input), self.chunk_size)
        ]

        response = await client.embeddings.create(
            model=self.model.value,
            input=chunks,
        )

        if not response or not response.data:
            raise ValueError("No embeddings received from Mistral API")

        # Average embeddings if multiple chunks
        all_embeddings = [item.embedding for item in response.data]
        avg_embedding = np.mean(all_embeddings, axis=0)

        return NPArray.from_numpy(avg_embedding)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["input"]
