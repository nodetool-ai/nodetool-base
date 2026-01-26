from enum import Enum
from typing import ClassVar, TypedDict

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import NPArray, Provider
from nodetool.providers.openai_prediction import run_openai
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

logger = get_logger(__name__)


class ResponseFormat(str, Enum):
    JSON_OBJECT = "json_object"
    TEXT = "text"


class GPTModel(str, Enum):
    """Available GPT models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PRO = "o1-pro"
    O3_MINI = "o3-mini"


class ChatComplete(BaseNode):
    """
    Generate text using OpenAI's GPT models.
    gpt, openai, chat, ai, text generation, llm, completion

    Uses OpenAI's GPT models to generate responses from prompts.
    Requires an OpenAI API key.

    Use cases:
    - Generate text responses to prompts
    - Build conversational AI applications
    - Code generation and explanation
    - Analysis and summarization tasks
    """

    _expose_as_tool: ClassVar[bool] = True

    model: GPTModel = Field(
        default=GPTModel.GPT_4O_MINI,
        description="The GPT model to use for generation",
    )

    prompt: str = Field(default="", description="The prompt for text generation")

    system_prompt: str = Field(
        default="",
        description="Optional system prompt to guide the model's behavior",
    )

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values make output more random.",
    )

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
        description="Maximum number of tokens to generate",
    )

    async def process(self, context: ProcessingContext) -> str:
        """
        Generate a chat completion using OpenAI's GPT models.

        Args:
            context: The processing context.

        Returns:
            str: The generated text response.
        """
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})

        response = await context.run_prediction(
            node_id=self.id,
            provider=Provider.OpenAI,
            model=self.model.value,
            run_prediction_function=run_openai,
            params={
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

        # Extract content from the response
        if isinstance(response, dict) and "choices" in response:
            try:
                content = response["choices"][0]["message"]["content"]
                return content if content else ""
            except (KeyError, IndexError, TypeError) as e:
                logger.warning("Could not extract content from response: %s", e)
                return str(response)
        else:
            logger.warning("Unexpected response type: %s", type(response))
            return str(response)

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model"]


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

    _expose_as_tool: ClassVar[bool] = True

    async def process(self, context: ProcessingContext) -> NPArray:
        import numpy as np
        from openai.types.create_embedding_response import CreateEmbeddingResponse

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
                logger.warning("Could not extract content from response: %s", e)
                return str(response)  # Return the raw response as string
        else:
            # Handle unexpected response types
            logger.warning("Unexpected response type: %s", type(response))
            return str(response)  # Return the raw response as string

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["query"]


class ModerationModel(str, Enum):
    """Available moderation models."""

    OMNI_MODERATION_LATEST = "omni-moderation-latest"
    OMNI_MODERATION_2024_09_26 = "omni-moderation-2024-09-26"
    TEXT_MODERATION_LATEST = "text-moderation-latest"
    TEXT_MODERATION_STABLE = "text-moderation-stable"


class Moderation(BaseNode):
    """
    Check text content for potential policy violations using OpenAI's moderation API.
    moderation, safety, content, filter, policy, harmful, toxic

    Uses OpenAI's moderation models to detect potentially harmful content including:
    - Hate speech
    - Harassment
    - Self-harm content
    - Sexual content
    - Violence
    - Graphic violence

    Returns flagged status and category scores for comprehensive content analysis.
    """

    input: str = Field(
        default="",
        description="The text content to check for policy violations.",
    )
    model: ModerationModel = Field(
        default=ModerationModel.OMNI_MODERATION_LATEST,
        description="The moderation model to use.",
    )

    _expose_as_tool: ClassVar[bool] = True

    class OutputType(TypedDict):
        flagged: bool
        categories: dict[str, bool]
        category_scores: dict[str, float]

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Check content for policy violations.

        Returns:
            Dict containing flagged status, categories, and category scores.
        """
        if not self.input:
            raise ValueError("Input text cannot be empty")

        response = await context.run_prediction(
            node_id=self.id,
            provider=Provider.OpenAI,
            model=self.model.value,
            run_prediction_function=run_openai,
            params={
                "input": self.input,
            },
        )

        # Parse the moderation response
        if isinstance(response, dict) and "results" in response:
            result = response["results"][0]
            return {
                "flagged": result.get("flagged", False),
                "categories": result.get("categories", {}),
                "category_scores": result.get("category_scores", {}),
            }
        else:
            logger.warning("Unexpected moderation response format: %s", type(response))
            return {
                "flagged": False,
                "categories": {},
                "category_scores": {},
            }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["input"]
