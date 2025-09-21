from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Agent(GraphNode):
    """
    Generate natural language responses using LLM providers and streams output.
    llm, text-generation, chatbot, question-answering, streaming
    """

    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    system: str | GraphNode | tuple[GraphNode, str] = Field(
        default="You are a an AI agent. \n\nBehavior\n- Understand the user's intent and the context of the task.\n- Break down the task into smaller steps.\n- Be precise, concise, and actionable.\n- Use tools to accomplish your goal. \n\nTool preambles\n- Outline the next step(s) you will perform.\n- After acting, summarize the outcome.\n\nRendering\n- Use Markdown to display media assets.\n- Display images, audio, and video assets using the appropriate Markdown.\n\nFile handling\n- Inputs and outputs are files in the /workspace directory.\n- Write outputs of code execution to the /workspace directory.\n",
        description="The system prompt for the LLM",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt for the LLM"
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to analyze",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="The audio to analyze",
    )
    history: list[types.Message] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The messages for the LLM"
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=32768, description=None
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )
    tool_call_limit: int | GraphNode | tuple[GraphNode, str] = Field(
        default=3,
        description="Maximum iterations that make tool calls before forcing a final answer. 0 disables tools.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Agent"


class Classifier(GraphNode):
    """
    Classify text into predefined or dynamic categories using LLM.
    classification, nlp, categorization

    Use cases:
    - Sentiment analysis
    - Topic classification
    - Intent detection
    - Content categorization
    """

    system_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default='\nYou are a precise text classifier.\n\nGoal\n- Select exactly one category from the list provided by the user.\n\nTool-calling rules\n- You MUST respond by calling the tool "classify" exactly once.\n- Provide only the "category" field in the tool arguments.\n- Do not produce any assistant text, only the tool call.\n\nSelection criteria\n- Choose the single best category that captures the main intent of the text.\n- If multiple categories seem plausible, pick the most probable one; do not return multiple.\n- If none fit perfectly, choose the closest allowed category. If the list includes "Other" or "Unknown", prefer it when appropriate.\n- Be robust to casing, punctuation, emojis, and minor typos. Handle negation correctly (e.g., "not spam" ≠ spam).\n- Never invent categories that are not in the provided list.\n\nBehavior\n- Be deterministic for the same input.\n- Do not ask clarifying questions; make the best choice with what\'s given.\n',
        description="The system prompt for the classifier",
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for classification",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text to classify"
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Optional image to classify in context",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="Optional audio to classify in context",
    )
    categories: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="List of possible categories. If empty, LLM will determine categories.",
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Classifier"


class Extractor(GraphNode):
    """
    Extract structured data from text content using LLM providers.
    data-extraction, structured-data, nlp, parsing

    Specialized for extracting structured information:
    - Converting unstructured text into structured data
    - Identifying and extracting specific fields from documents
    - Parsing text according to predefined schemas
    - Creating structured records from natural language content
    """

    system_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="\n        You are an expert data extractor. Your task is to extract specific information from text according to a defined schema.\n        ",
        description="The system prompt for the data extractor",
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for data extraction",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to extract data from"
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Optional image to assist extraction",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="Optional audio to assist extraction",
    )
    extraction_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Extract the following information from the text:",
        description="Additional instructions for the extraction process",
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The maximum number of tokens to generate."
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Extractor"


class Summarizer(GraphNode):
    """
    Generate concise summaries of text content using LLM providers with streaming output.
    text, summarization, nlp, content, streaming

    Specialized for creating high-quality summaries with real-time streaming:
    - Condensing long documents into key points
    - Creating executive summaries with live output
    - Extracting main ideas from text as they're generated
    - Maintaining factual accuracy while reducing length
    """

    system_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="\n        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring. \n        Follow these guidelines:\n        1. Identify and include only the most important information.\n        2. Maintain factual accuracy - do not add or modify information.\n        3. Use clear, direct language.\n        4. Aim for approximately {self.max_tokens} tokens.\n        ",
        description="The system prompt for the summarizer",
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for summarization",
    )
    text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The text to summarize"
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Optional image to condition the summary",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="Optional audio to condition the summary",
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=200, description="Target maximum number of tokens for the summary"
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Summarizer"
