from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ChartGenerator(GraphNode):
    """
    LLM Agent to create Plotly Express charts based on natural language descriptions.
    llm, data visualization, charts

    Use cases:
    - Generating interactive charts from natural language descriptions
    - Creating data visualizations with minimal configuration
    - Converting data analysis requirements into visual representations
    """

    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="The model to use for chart generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Natural language description of the desired chart"
    )
    data: types.DataframeRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.DataframeRef(
            type="dataframe", uri="", asset_id=None, data=None, columns=None
        ),
        description="The data to visualize",
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The maximum number of tokens to generate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.generators.ChartGenerator"


class DataGenerator(GraphNode):
    """
    LLM Agent to create a dataframe based on a user prompt.
    llm, dataframe creation, data structuring

    Use cases:
    - Generating structured data from natural language descriptions
    - Creating sample datasets for testing or demonstration
    - Converting unstructured text into tabular format
    """

    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="The model to use for data generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The user prompt"
    )
    input_text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The input text to be analyzed by the agent."
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The maximum number of tokens to generate."
    )
    columns: types.RecordType | GraphNode | tuple[GraphNode, str] = Field(
        default=types.RecordType(type="record_type", columns=[]),
        description="The columns to use in the dataframe.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.generators.DataGenerator"


class ListGenerator(GraphNode):
    """
    LLM Agent to create a stream of strings based on a user prompt.
    llm, text streaming

    Use cases:
    - Generating text from natural language descriptions
    - Streaming responses from an LLM
    """

    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="The model to use for string generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The user prompt"
    )
    input_text: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The input text to be analyzed by the agent."
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="The maximum number of tokens to generate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.generators.ListGenerator"


class SVGGenerator(GraphNode):
    """
    LLM Agent to create SVG elements based on user prompts.
    svg, generator, vector, graphics

    Use cases:
    - Creating vector graphics from text descriptions
    - Generating scalable illustrations
    - Creating custom icons and diagrams
    """

    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="The language model to use for SVG generation.",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The user prompt for SVG generation"
    )
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="Image to use for generation",
    )
    audio: types.AudioRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.AudioRef(type="audio", uri="", asset_id=None, data=None),
        description="Audio to use for generation",
    )
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=8192, description="The maximum number of tokens to generate."
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.generators.SVGGenerator"
