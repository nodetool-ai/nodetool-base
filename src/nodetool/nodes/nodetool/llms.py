import json
from pydantic import Field

from nodetool.metadata.types import (
    Message,
    AgentModel,
)
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    MessageImageContent,
    ImageRef,
    ToolName,
    ToolCall,
)
from nodetool.agents.tools.base import get_tool_by_name, Tool
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, ToolCallUpdate
from nodetool.chat.providers import Chunk
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.metadata.types import DataframeRef, RecordType

from nodetool.nodes.nodetool.agents import provider_from_model


def init_tool(tool: ToolName, workspace_dir: str) -> Tool | None:
    if tool.name:
        tool_class = get_tool_by_name(tool.name)
        if tool_class:
            return tool_class(workspace_dir)
        else:
            return None
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

    @classmethod
    def get_title(cls) -> str:
        return "LLM"

    model: AgentModel = Field(
        default=AgentModel.gpt_4o,
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
    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    messages: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=4096, ge=1, le=100000)

    async def process(self, context: ProcessingContext) -> str:
        content = []

        content.append(MessageTextContent(text=self.prompt))

        if self.image.is_set():
            content.append(MessageImageContent(image=self.image))

        messages = [
            Message(role="system", content=self.system),
        ]

        for message in self.messages:
            messages.append(message)

        messages.append(Message(role="user", content=content))
        provider = provider_from_model(self.model.value)
        tools = [init_tool(tool, context.workspace_dir) for tool in self.tools]

        result_content = ""

        while True:
            follow_up_messages = []
            tool_calls_message = None
            async for chunk in provider.generate_messages(
                messages=messages,
                model=self.model.value,
                max_tokens=self.max_tokens,
                tools=tools,
            ):  # type: ignore
                if isinstance(chunk, Chunk):
                    # Send chunk via context.post_message
                    context.post_message(
                        Chunk(
                            node_id=self.id,
                            content=chunk.content,
                        )
                    )
                    result_content += chunk.content
                if isinstance(chunk, ToolCall):
                    context.post_message(
                        ToolCallUpdate(
                            name=chunk.name,
                            args=chunk.args,
                        )
                    )
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
                            tool_result = await tool.process(context, chunk.args)
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

        return result_content

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt", "model", "system", "messages"]


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
        return "Text Summarizer"

    model: AgentModel = Field(
        default=AgentModel.gpt_4o,
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
        system_prompt = f"""
        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries.
        Follow these guidelines:
        1. Identify and include only the most important information
        2. Maintain factual accuracy - do not add or modify information
        3. Use clear, direct language
        4. Aim for approximately {self.max_words} words
        5. Preserve the original meaning and tone
        6. Include key details, dates, and figures when relevant
        7. Focus on the main points and conclusions
        8. Avoid redundancy and unnecessary elaboration

        RESPOND ONLY WITH THE SUMMARY TEXT. NO ADDITIONAL COMMENTARY.
        """

        content = []
        content.append(MessageTextContent(text=self.text))

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=content),
        ]

        provider = provider_from_model(self.model.value)

        result_content = ""

        async for chunk in provider.generate_messages(
            messages=messages,
            model=self.model.value,
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
        return "Data Extractor"

    model: AgentModel = Field(
        default=AgentModel.gpt_4o,
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

        system_prompt = """
        You are a precise data extraction assistant. Your task is to extract specific information from text according to a defined schema.
        Follow these guidelines:
        1. Extract ONLY the information requested in the schema
        2. Maintain factual accuracy - do not add or modify information
        3. If information is not present in the text, leave the field empty
        4. Extract ALL instances of the requested data (multiple records)
        5. Be precise and thorough in your extraction
        
        Return ONLY valid JSON that matches the provided schema.
        """

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"{self.extraction_prompt}\n\n{self.text}"),
        ]

        provider = provider_from_model(self.model.value)

        assistant_message = await provider.generate_message(
            model=self.model.value,
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


class TextClassifier(BaseNode):
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
        return "Text Classifier"

    model: AgentModel = Field(
        default=AgentModel.gpt_4o,
        description="Model to use for classification",
    )
    text: str = Field(default="", description="Text to classify")
    categories: list[str] = Field(
        default=[],
        description="List of possible categories. If empty, LLM will determine categories.",
    )
    multi_label: bool = Field(
        default=False,
        description="Allow multiple category assignments",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "categories", "multi_label", "model"]

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        system_prompt = """You are a precise text classifier. Your task is to analyze the input text and assign confidence scores.

        Guidelines:
        1. Assign confidence scores between 0.0 and 1.0 for each category
        2. Be decisive - assign high scores (>0.8) only when very confident
        3. If no categories are provided, identify 2-5 relevant categories
        4. For single-label classification, ensure one category has a significantly higher score
        5. For multi-label classification, assign independent scores to each category
        """

        content = []
        content.append(MessageTextContent(text=self.text))

        if len(self.categories) < 2:
            raise ValueError("At least 2 categories are required")

        category_info = (
            f"\nClassify into these categories: {', '.join(self.categories)}"
        )

        label_type = "multi-label" if self.multi_label else "single-label"
        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=f"Perform {label_type} classification on the following text:{category_info}\n\nText: {self.text}",
            ),
        ]

        provider = provider_from_model(self.model.value)

        classification_schema = {
            "name": "classification_results",
            "schema": {
                "type": "object",
                "title": "Classification Results",
                "description": "Category confidence scores between 0 and 1",
                "additionalProperties": False,
                "required": self.categories,
                "properties": {
                    category: {
                        "type": "number",
                        "description": f"Confidence score between 0 and 1 for category '{category}'",
                    }
                    for category in self.categories
                },
            },
            "strict": True,
        }

        assistant_message = await provider.generate_message(
            model=self.model.value,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": classification_schema,
            },
        )

        # Parse the response and ensure it's in the correct format
        try:
            classifications = json.loads(str(assistant_message.content))

            # For single-label classification, normalize scores to sum to 1.0
            if not self.multi_label:
                total = sum(classifications.values())
                if total > 0:
                    classifications = {k: v / total for k, v in classifications.items()}

            return classifications

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from model")
