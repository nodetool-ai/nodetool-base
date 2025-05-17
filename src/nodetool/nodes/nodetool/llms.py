import base64
from datetime import datetime
from enum import Enum
from io import BytesIO
import json
from typing import AsyncGenerator
import pydub
from pydantic import Field

from nodetool.metadata.types import (
    Message,
)
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    ImageRef,
    ToolName,
    ToolCall,
    AudioRef,
    LanguageModel,
)
from nodetool.agents.tools.base import Tool
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, ToolCallUpdate
from nodetool.chat.providers import Chunk
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.metadata.types import DataframeRef, RecordType
from nodetool.metadata.types import Provider
from nodetool.chat.providers import get_provider

from nodetool.agents.tools import (
    GoogleImageGenerationTool,
    GoogleGroundedSearchTool,
    GoogleNewsTool,
    GoogleImagesTool,
    GoogleSearchTool,
    GoogleLensTool,
    GoogleMapsTool,
    GoogleShoppingTool,
    GoogleFinanceTool,
    GoogleJobsTool,
    BrowserTool,
    ChromaHybridSearchTool,
    SearchEmailTool,
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
)

TOOLS = {
    tool.name: tool
    for tool in [
        GoogleImageGenerationTool,
        GoogleGroundedSearchTool,
        GoogleNewsTool,
        GoogleImagesTool,
        GoogleSearchTool,
        GoogleLensTool,
        GoogleMapsTool,
        GoogleShoppingTool,
        GoogleFinanceTool,
        GoogleJobsTool,
        BrowserTool,
        ChromaHybridSearchTool,
        SearchEmailTool,
        OpenAIImageGenerationTool,
        OpenAITextToSpeechTool,
    ]
}


def init_tool(tool: ToolName) -> Tool | None:
    if tool.name:
        tool_class = TOOLS.get(tool.name)
        if tool_class:
            return tool_class()
        else:
            raise ValueError(f"Tool {tool.name} not found")
    else:
        return None


class LLM(BaseNode):
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

    class Voice(str, Enum):
        NONE = "none"
        ALLOY = "alloy"
        ECHO = "echo"
        FABLE = "fable"
        ONYX = "onyx"
        NOVA = "nova"
        SHIMMER = "shimmer"

    @classmethod
    def get_title(cls) -> str:
        return "LLM"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "audio": AudioRef,
            "image": ImageRef,
        }

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )
    system: str = Field(
        title="System",
        default="You are a friendly assistant.",
        description="The system prompt for the LLM",
    )
    prompt: str = Field(
        title="Prompt",
        default="",
        description="The prompt for the LLM",
    )
    image: ImageRef = Field(
        title="Image",
        default=ImageRef(),
        description="The image to analyze",
    )
    audio: AudioRef = Field(
        title="Audio",
        default=AudioRef(),
        description="The audio to analyze",
    )
    voice: Voice = Field(
        title="Voice",
        default=Voice.NONE,
        description="The voice for the audio output (only for OpenAI)",
    )
    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    messages: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=4096, ge=1, le=100000)
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    async def process(self, context: ProcessingContext):
        from nodetool.models.prediction import Prediction as PredictionModel
        from nodetool.chat.providers.openai_provider import calculate_chat_cost

        content = []

        content.append(MessageTextContent(text=self.prompt))

        if self.image.is_set():
            content.append(MessageImageContent(image=self.image))

        if self.audio.is_set():
            content.append(MessageAudioContent(audio=self.audio))

        messages = [
            Message(role="system", content=self.system),
        ]

        for message in self.messages:
            messages.append(message)

        messages.append(Message(role="user", content=content))
        tools = [init_tool(tool) for tool in self.tools]
        result_content = ""
        result_audio = None
        result_image = None
        audio_chunks = []

        while True:
            follow_up_messages = []
            tool_calls_message = None
            if self.voice != self.Voice.NONE and self.model.provider != Provider.OpenAI:
                raise ValueError("Voice is only supported for OpenAI")

            async for chunk in context.generate_messages(
                messages=messages,
                provider=self.model.provider,
                model=self.model.id,
                node_id=self.id,
                tools=tools,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
                audio=(
                    {"voice": self.voice.value, "format": "pcm16"}
                    if self.voice != self.Voice.NONE
                    else None
                ),
            ):  # type: ignore
                if isinstance(chunk, Chunk):
                    # Send chunk via context.post_message
                    context.post_message(
                        Chunk(
                            node_id=self.id,
                            content=chunk.content,
                            content_type=chunk.content_type,
                        )
                    )
                    if chunk.content_type == "audio":
                        audio_chunks.append(chunk.content)
                    elif chunk.content_type == "image":
                        result_image = ImageRef(
                            data=base64.b64decode(chunk.content),
                        )
                    else:
                        result_content += chunk.content
                if isinstance(chunk, ToolCall):
                    if tool_calls_message is None:
                        tool_calls_message = Message(
                            role="assistant",
                            tool_calls=[],
                        )
                        follow_up_messages.append(tool_calls_message)
                    assert tool_calls_message.tool_calls is not None
                    tool_calls_message.tool_calls.append(chunk)
                    for tool in tools:
                        if tool and tool.name == chunk.name:
                            context.post_message(
                                ToolCallUpdate(
                                    node_id=self.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    message=tool.user_message(chunk.args),
                                )
                            )
                            tool_result = await tool.process(context, chunk.args)
                            if isinstance(tool_result, dict) and "type" in tool_result:
                                if (
                                    tool_result["type"] == "image"
                                    and "output_file" in tool_result
                                ):
                                    file_path = context.resolve_workspace_path(
                                        tool_result["output_file"]
                                    )
                                    with open(file_path, "rb") as f:
                                        result_image = await context.image_from_io(f)
                                elif (
                                    tool_result["type"] == "audio"
                                    and "output_file" in tool_result
                                ):
                                    file_path = context.resolve_workspace_path(
                                        tool_result["output_file"]
                                    )
                                    with open(file_path, "rb") as f:
                                        result_audio = await context.audio_from_io(f)

                            follow_up_messages.append(
                                Message(
                                    role="tool",
                                    tool_call_id=chunk.id,
                                    name=chunk.name,
                                    content=json.dumps(tool_result),
                                )
                            )
            if len(follow_up_messages) > 0:
                messages = messages + follow_up_messages
            else:
                break

        # convert audio chunks
        if len(audio_chunks) > 0:
            raw_pcm = base64.b64decode("".join(audio_chunks))
            segment = pydub.AudioSegment.from_raw(
                BytesIO(raw_pcm),
                format="pcm",
                sample_width=2,
                frame_rate=24000,
                channels=1,
            )
            result_audio = await context.audio_from_segment(
                segment,
            )

        return {
            "text": result_content,
            "audio": result_audio,
            "image": result_image,
        }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "model",
            "messages",
            "image",
            "audio",
            "tools",
        ]


class Summarizer(BaseNode):
    """
    Generate concise summaries of text content using LLM providers.
    text, summarization, nlp, content

    Specialized for creating high-quality summaries:
    - Condensing long documents into key points
    - Creating executive summaries
    - Extracting main ideas from text
    - Maintaining factual accuracy while reducing length
    """

    @classmethod
    def get_title(cls) -> str:
        return "Summarizer"

    system_prompt: str = Field(
        default="""
        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring. 
        Follow these guidelines:
        1. Identify and include only the most important information.
        2. Maintain factual accuracy - do not add or modify information.
        3. Use clear, direct language.
        4. Aim for approximately {self.max_words} words.
        """,
        description="The system prompt for the summarizer",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for summarization",
    )
    text: str = Field(default="", description="The text to summarize")
    max_words: int = Field(
        default=150,
        description="Target maximum number of words for the summary",
        ge=50,
        le=500,
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "max_words", "model"]

    async def process(self, context: ProcessingContext) -> str:
        content = []
        content.append(MessageTextContent(text=self.text))

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=content),
        ]

        assert self.model.provider != Provider.Empty, "Select a model"

        result_content = ""

        async for chunk in context.generate_messages(
            messages=messages,
            model=self.model.id,
            node_id=self.id,
            provider=self.model.provider,
            max_tokens=8192,
        ):  # type: ignore
            if isinstance(chunk, Chunk):
                context.post_message(
                    NodeProgress(
                        node_id=self.id,
                        progress=0,
                        total=0,
                        chunk=chunk.content,
                    )
                )
                result_content += chunk.content

        return result_content


class Extractor(BaseNode):
    """
    Extract structured data from text content using LLM providers.
    data-extraction, structured-data, nlp, parsing

    Specialized for extracting structured information:
    - Converting unstructured text into structured data
    - Identifying and extracting specific fields from documents
    - Parsing text according to predefined schemas
    - Creating structured records from natural language content
    """

    @classmethod
    def get_title(cls) -> str:
        return "Extractor"

    system_prompt: str = Field(
        default="""
        You are an expert data extractor. Your task is to extract specific information from text according to a defined schema.
        """,
        description="The system prompt for the data extractor",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for data extraction",
    )
    text: str = Field(default="", description="The text to extract data from")
    extraction_prompt: str = Field(
        default="Extract the following information from the text:",
        description="Additional instructions for the extraction process",
    )
    columns: RecordType = Field(
        default=RecordType(),
        description="The fields to extract from the text",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="The maximum number of tokens to generate.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "extraction_prompt", "columns", "model"]

    async def process(self, context: ProcessingContext) -> DataframeRef:
        import json

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"{self.extraction_prompt}\n\n{self.text}"),
        ]

        provider = get_provider(self.model.provider)

        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_results",
                    "schema": json_schema_for_dataframe(self.columns.columns),
                    "strict": True,
                },
            },
        )

        data = [
            [
                (row[col.name] if col.name in row else None)
                for col in self.columns.columns
            ]
            for row in json.loads(str(assistant_message.content)).get("data", [])
        ]

        return DataframeRef(columns=self.columns.columns, data=data)


class Classifier(BaseNode):
    """
    Classify text into predefined or dynamic categories using LLM.
    classification, nlp, categorization

    Use cases:
    - Sentiment analysis
    - Topic classification
    - Intent detection
    - Content categorization
    """

    @classmethod
    def get_title(cls) -> str:
        return "Classifier"

    system_prompt: str = Field(
        default="""
        You are a precise text classifier. Your task is to analyze the input text and assign confidence scores.
        """,
        description="The system prompt for the classifier",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for classification",
    )
    text: str = Field(default="", description="Text to classify")
    categories: list[str] = Field(
        default=[],
        description="List of possible categories. If empty, LLM will determine categories.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "categories", "model"]

    async def process(self, context: ProcessingContext) -> str:
        content = []
        content.append(MessageTextContent(text=self.text))

        assert self.model.provider != Provider.Empty, "Select a model"

        if len(self.categories) < 2:
            raise ValueError("At least 2 categories are required")

        category_info = (
            f"\nClassify into these categories: {', '.join(self.categories)}"
        )

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(
                role="user",
                content=f"Perform classification on the following text:{category_info}\n\nText: {self.text}",
            ),
        ]

        provider = get_provider(self.model.provider)

        classification_schema = {
            "name": "classification_results",
            "schema": {
                "type": "object",
                "title": "Classification Results",
                "description": "Category confidence scores between 0 and 1",
                "additionalProperties": False,
                "required": ["category"],
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the text",
                        "enum": self.categories,
                    }
                },
            },
            "strict": True,
        }

        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": classification_schema,
            },
        )

        classification = json.loads(str(assistant_message.content))

        return classification["category"]


DEFAULT_SYSTEM_PROMPT = """
You can call tools to help you answer the user's question.
You can also call tools to help you generate images, or audio.
Generated images and audio will be shown to the user above your response.
DO NOT LINK TO ANY IMAGES OR AUDIO.
"""


class LLMStreaming(BaseNode):
    """
    Generate natural language responses using LLM providers and streams output.
    llm, text-generation, chatbot, question-answering, streaming
    """

    @classmethod
    def get_title(cls) -> str:
        return "LLM (Streaming)"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "image": ImageRef,
            "audio": AudioRef,
        }

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )
    system: str = Field(
        title="System",
        default="You are a friendly assistant.",
        description="The system prompt for the LLM",
    )
    prompt: str = Field(
        title="Prompt",
        default="",
        description="The prompt for the LLM",
    )
    image: ImageRef = Field(
        title="Image",
        default=ImageRef(),
        description="The image to analyze",
    )
    audio: AudioRef = Field(
        title="Audio",
        default=AudioRef(),
        description="The audio to analyze",
    )
    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )
    messages: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=4096, ge=1, le=100000)
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "prompt",
            "model",
            "messages",
            "image",
            "audio",
            "tools",
        ]

    async def gen_process(self, context: ProcessingContext):
        content = []

        content.append(MessageTextContent(text=self.prompt))

        if self.image.is_set():
            content.append(MessageImageContent(image=self.image))

        if self.audio.is_set():
            content.append(MessageAudioContent(audio=self.audio))

        messages = [
            Message(role="system", content=self.system + DEFAULT_SYSTEM_PROMPT),
        ]

        for message in self.messages:
            messages.append(message)

        messages.append(Message(role="user", content=content))
        tools = [init_tool(tool) for tool in self.tools]

        while True:
            follow_up_messages = []
            tool_calls_message = None

            async for chunk in context.generate_messages(
                messages=messages,
                provider=self.model.provider,
                model=self.model.id,
                node_id=self.id,
                tools=tools,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
            ):  # type: ignore
                if isinstance(chunk, Chunk):
                    if chunk.content_type == "text" or chunk.content_type is None:
                        yield "text", chunk.content
                    elif chunk.content_type == "audio":
                        yield "audio", chunk.content
                    elif chunk.content_type == "image":
                        yield "image", base64.b64decode(chunk.content)
                if isinstance(chunk, ToolCall):
                    if tool_calls_message is None:
                        tool_calls_message = Message(
                            role="assistant",
                            tool_calls=[],
                        )
                        follow_up_messages.append(tool_calls_message)
                    assert tool_calls_message.tool_calls is not None
                    tool_calls_message.tool_calls.append(chunk)
                    for tool_instance in tools:
                        if tool_instance and tool_instance.name == chunk.name:
                            context.post_message(
                                ToolCallUpdate(
                                    node_id=self.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    message=tool_instance.user_message(chunk.args),
                                )
                            )
                            tool_result = await tool_instance.process(
                                context, chunk.args
                            )
                            # Handle tool results that might contain image or audio
                            if isinstance(tool_result, dict) and "type" in tool_result:
                                if (
                                    tool_result["type"] == "image"
                                    and "output_file" in tool_result
                                ):
                                    file_path = context.resolve_workspace_path(
                                        tool_result["output_file"]
                                    )
                                    with open(file_path, "rb") as f:
                                        image = await context.image_from_io(f)
                                        yield "image", image
                                elif (
                                    tool_result["type"] == "audio"
                                    and "output_file" in tool_result
                                ):
                                    file_path = context.resolve_workspace_path(
                                        tool_result["output_file"]
                                    )
                                    with open(file_path, "rb") as f:
                                        audio = await context.audio_from_io(f)
                                        yield "audio", audio
                            follow_up_messages.append(
                                Message(
                                    role="tool",
                                    tool_call_id=chunk.id,
                                    name=chunk.name,
                                    content=json.dumps(tool_result),
                                )
                            )
            if len(follow_up_messages) > 0:
                messages = messages + follow_up_messages
            else:
                break
