from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


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
        default="\n        You are a precise text classifier. Your task is to analyze the input text and assign confidence scores.\n        ",
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
    categories: list[str] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="List of possible categories. If empty, LLM will determine categories.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.llms.Classifier"


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
    extraction_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Extract the following information from the text:",
        description="Additional instructions for the extraction process",
    )
    columns: types.RecordType | GraphNode | tuple[GraphNode, str] = Field(
        default=types.RecordType(type="record_type", columns=[]),
        description="The fields to extract from the text",
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The maximum number of tokens to generate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.llms.Extractor"


import nodetool.nodes.nodetool.llms


class LLM(GraphNode):
    """
    Generate natural language responses using LLM providers.
    llm, text-generation, chatbot, question-answering

    Leverages LLM providers to:
    - Generate human-like text responses
    - Answer questions
    - Complete prompts
    - Engage in conversational interactions
    - Assist with writing and editing tasks
    - Perform text analysis and summarization
    """

    Voice: typing.ClassVar[type] = nodetool.nodes.nodetool.llms.LLM.Voice
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
        default="You are a friendly assistant.",
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
    voice: nodetool.nodes.nodetool.llms.LLM.Voice = Field(
        default=nodetool.nodes.nodetool.llms.LLM.Voice.NONE,
        description="The voice for the audio output (only for OpenAI)",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    messages: list[types.Message] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The messages for the LLM"
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.llms.LLM"


class LLMStreaming(GraphNode):
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
        default="You are a friendly assistant.",
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
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    messages: list[types.Message] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="The messages for the LLM"
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.llms.LLMStreaming"


class Summarizer(GraphNode):
    """
    Generate concise summaries of text content using LLM providers.
    text, summarization, nlp, content

    Specialized for creating high-quality summaries:
    - Condensing long documents into key points
    - Creating executive summaries
    - Extracting main ideas from text
    - Maintaining factual accuracy while reducing length
    """

    system_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="\n        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring. \n        Follow these guidelines:\n        1. Identify and include only the most important information.\n        2. Maintain factual accuracy - do not add or modify information.\n        3. Use clear, direct language.\n        4. Aim for approximately {self.max_words} words.\n        ",
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
    max_words: int | GraphNode | tuple[GraphNode, str] = Field(
        default=150, description="Target maximum number of words for the summary"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.llms.Summarizer"
