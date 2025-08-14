from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.nodetool.agents


class AgentNode(GraphNode):
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

    Voice: typing.ClassVar[type] = nodetool.nodes.nodetool.agents.AgentNode.Voice
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
    voice: nodetool.nodes.nodetool.agents.AgentNode.Voice = Field(
        default=nodetool.nodes.nodetool.agents.AgentNode.Voice.NONE,
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
        return "nodetool.agents.Agent"


class AgentStreaming(GraphNode):
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
        return "nodetool.agents.AgentStreaming"


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
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Classifier"


class DataframeAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a structured output
    agent, execution, tasks

    Use cases:
    - Automate complex workflows with reasoning
    - Process tasks with tool calling capabilities
    - Solve problems step-by-step with LLM reasoning
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Agent", description="The name of the agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning tasks",
    )
    task: types.Task | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Task(type="task", id="", title="", description="", subtasks=[]),
        description="Pre-defined task to execute, skipping planning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    input_files: list[types.FilePath] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of input files to use for the agent"
    )
    max_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )
    columns: types.RecordType | GraphNode | tuple[GraphNode, str] = Field(
        default=types.RecordType(type="record_type", columns=[]),
        description="The columns to use in the dataframe.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.DataframeAgent"


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
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Extractor"


class ListAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a list
    agent, execution, tasks, list

    Use cases:
    - Generate lists of items
    - Create sequences of steps
    - Collect multiple results
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Agent", description="The name of the agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning tasks",
    )
    task: types.Task | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Task(type="task", id="", title="", description="", subtasks=[]),
        description="Pre-defined task to execute, skipping planning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    input_files: list[types.FilePath] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of input files to use for the agent"
    )
    max_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )
    item_type: str | GraphNode | tuple[GraphNode, str] = Field(
        default="string",
        description="The type of items in the list (string, number, object)",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.ListAgent"


class MultiStepAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools
    agent, execution, tasks

    Use cases:
    - Automate complex workflows with reasoning
    - Process tasks with tool calling capabilities
    - Solve problems step-by-step with LLM reasoning
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Agent", description="The name of the agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning tasks",
    )
    task: types.Task | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Task(type="task", id="", title="", description="", subtasks=[]),
        description="Pre-defined task to execute, skipping planning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    input_files: list[types.FilePath] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of input files to use for the agent"
    )
    max_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.MultiStepAgent"


class MultiStepAgentStreaming(GraphNode):
    """
    Executes tasks using a multi-step agent that streams results as they're generated.
    agent, execution, tasks, streaming

    Use cases:
    - Real-time interactive applications
    - Progressive rendering of agent responses
    - Streaming AI interfaces
    - Live-updating workflows
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Agent", description="The name of the agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning tasks",
    )
    task: types.Task | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Task(type="task", id="", title="", description="", subtasks=[]),
        description="Pre-defined task to execute, skipping planning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    input_files: list[types.FilePath] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of input files to use for the agent"
    )
    max_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.MultiStepAgentStreaming"


class StructuredOutputAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a structured output
    agent, execution, tasks
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Agent", description="The name of the agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for execution",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning tasks",
    )
    task: types.Task | GraphNode | tuple[GraphNode, str] = Field(
        default=types.Task(type="task", id="", title="", description="", subtasks=[]),
        description="Pre-defined task to execute, skipping planning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    input_files: list[types.FilePath] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of input files to use for the agent"
    )
    max_steps: int | GraphNode | tuple[GraphNode, str] = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )
    fields: types.RecordType | GraphNode | tuple[GraphNode, str] = Field(
        default=types.RecordType(type="record_type", columns=[]),
        description="The fields to use in the dictionary.",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.StructuredOutputAgent"


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
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=200, description="Target maximum number of tokens for the summary"
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description="Context window for the model"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.Summarizer"


class SummarizerStreaming(GraphNode):
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
    max_tokens: int | GraphNode | tuple[GraphNode, str] = Field(
        default=200, description="Target maximum number of tokens for the summary"
    )
    context_window: int | GraphNode | tuple[GraphNode, str] = Field(
        default=4096, description=None
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.SummarizerStreaming"


class TaskPlannerNode(GraphNode):
    """
    Generates a Task execution plan based on an objective, model, and tools. Outputs a Task object that can be used by an Agent executor.
    planning, task generation, workflow design
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Task Planner", description="The name of the task planner node"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or problem to create a plan for"
    )
    model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for planning",
    )
    reasoning_model: types.LanguageModel | GraphNode | tuple[GraphNode, str] = Field(
        default=types.LanguageModel(
            type="language_model",
            provider=nodetool.metadata.types.Provider.Empty,
            id="",
            name="",
        ),
        description="Model to use for reasoning",
    )
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[],
        description="List of EXECUTION tools available for the planned subtasks",
    )
    output_schema: dict | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Optional JSON schema for the final task output"
    )
    enable_analysis_phase: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to use analysis in the planning phase"
    )
    enable_data_contracts_phase: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to use data contracts in the planning phase"
    )
    use_structured_output: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False,
        description="Attempt to use structured output for plan generation",
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.TaskPlanner"
