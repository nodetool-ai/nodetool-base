from nodetool.chat.providers.openai_prediction import run_openai
import numpy as np
from nodetool.metadata.types import (
    NPArray,
    Provider,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

from openai.types.create_embedding_response import CreateEmbeddingResponse
from pydantic import Field

from enum import Enum

from typing import Any, Dict


class ResponseFormat(str, Enum):
    JSON_OBJECT = "json_object"
    TEXT = "text"


class Embedding(BaseNode):
    """
    Generate vector representations of text for semantic analysis.
    embeddings, similarity, search, clustering, classification

    Uses OpenAI's embedding models to create dense vector representations of text.
    These vectors capture semantic meaning, enabling:
    - Semantic search
    - Text clustering
    - Document classification
    - Recommendation systems
    - Anomaly detection
    - Measuring text similarity and diversity
    """

    class EmbeddingModel(str, Enum):
        TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
        TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

    input: str = Field(title="Input", default="")
    model: EmbeddingModel = Field(
        title="Model", default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL
    )
    chunk_size: int = 4096

    async def process(self, context: ProcessingContext) -> NPArray:
        # chunk the input into smaller pieces
        chunks = [
            self.input[i : i + self.chunk_size]
            for i in range(0, len(self.input), self.chunk_size)
        ]

        response = await context.run_prediction(
            self.id,
            provider="openai",
            params={"input": chunks},
            model=self.model.value,
            run_prediction_function=run_openai,
        )

        res = CreateEmbeddingResponse(**response)

        all = [i.embedding for i in res.data]
        avg = np.mean(all, axis=0)
        return NPArray.from_numpy(avg)


class WebSearch(BaseNode):
    """
    🔍 OpenAI Web Search - Searches the web using OpenAI's web search capabilities.

    This node uses an OpenAI model equipped with web search functionality
    (like gpt-4o with search preview) to answer queries based on current web information.
    Requires an OpenAI API key.
    """

    query: str = Field(
        title="Query", default="", description="The search query to execute."
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Execute a web search query using an OpenAI model.

        Args:
            context: The processing context.

        Returns:
            str: The content returned by the model, incorporating web search results.
        """
        if not self.query:
            raise ValueError("Search query cannot be empty")

        response = await context.run_prediction(
            node_id=self.id,
            provider=Provider.OpenAI,
            model="gpt-4o-search-preview",
            run_prediction_function=run_openai,
            params={
                "web_search_options": {},
                "messages": [{"role": "user", "content": self.query}],
            },
        )

        # Process the response - Assuming the result content is in a standard location
        # This might need adjustment based on the actual structure returned by run_openai for chat completions
        if isinstance(response, dict) and "choices" in response:
            # Attempt to access message content from the first choice
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                # Fallback or error handling if the expected structure isn't found
                print(f"Could not extract content from response: {e}")
                return str(response)  # Return the raw response as string
        else:
            # Handle unexpected response types
            print(f"Unexpected response type: {type(response)}")
            return str(response)  # Return the raw response as string

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["query"]
