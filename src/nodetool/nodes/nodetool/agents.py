import base64
import asyncio
from enum import Enum
import io
import json
import logging
from typing import Any

from nodetool.agents.tools.workflow_tool import GraphTool
from nodetool.workflows.graph_utils import find_node, get_downstream_subgraph
from openai.types.beta.realtime import (
    ResponseAudioDoneEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.beta.realtime.session_update_event_param import Session
from openai.types.beta.realtime.response_create_event_param import Response
from openai.types.beta.realtime.error_event import ErrorEvent
from openai.types.beta.realtime.response_audio_delta_event import (
    ResponseAudioDeltaEvent,
)
from openai.types.beta.realtime.response_audio_transcript_delta_event import (
    ResponseAudioTranscriptDeltaEvent,
)
from pydantic import Field

from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers import get_provider
from nodetool.config.environment import Environment

from nodetool.workflows.types import (
    ToolCallUpdate,
)

from nodetool.metadata.types import (
    LanguageModel,
    ToolName,
    ImageRef,
    AudioRef,
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
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, ToolCallUpdate, EdgeUpdate
from nodetool.chat.providers import Chunk
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.metadata.types import DataframeRef, RecordType
from nodetool.metadata.types import Provider
from nodetool.chat.providers import get_provider

log = logging.getLogger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


# class TaskPlannerNode(BaseNode):
#     """
#     Generates a Task execution plan based on an objective, model, and tools. Outputs a Task object that can be used by an Agent executor.
#     planning, task generation, workflow design
#     """

#     name: str = Field(
#         default="Task Planner",
#         description="The name of the task planner node",
#     )

#     objective: str = Field(
#         default="", description="The objective or problem to create a plan for"
#     )

#     model: LanguageModel = Field(
#         default=LanguageModel(),
#         description="Model to use for planning",
#     )

#     reasoning_model: LanguageModel = Field(
#         default=LanguageModel(),
#         description="Model to use for reasoning",
#     )

#     tools: list[ToolName] = Field(
#         default=[],
#         description="List of EXECUTION tools available for the planned subtasks",
#     )

#     output_schema: Optional[dict] = Field(
#         default=None, description="Optional JSON schema for the final task output"
#     )

#     enable_analysis_phase: bool = Field(
#         default=True, description="Whether to use analysis in the planning phase"
#     )

#     enable_data_contracts_phase: bool = Field(
#         default=True, description="Whether to use data contracts in the planning phase"
#     )

#     use_structured_output: bool = Field(
#         default=False,
#         description="Attempt to use structured output for plan generation",
#     )

#     _is_dynamic = True

#     @classmethod
#     def get_basic_fields(cls) -> list[str]:
#         return [
#             "objective",
#             "model",
#             "tools",
#             "input_files",
#             "output_schema",
#             "enable_analysis_phase",
#             "enable_data_contracts_phase",
#             "use_structured_output",
#         ]

#     @classmethod
#     def get_input_fields(cls) -> list[str]:
#         return cls.get_basic_fields()

#     @classmethod
#     def get_output_fields(cls) -> list[str]:
#         return ["task"]  # Output field name

#     async def process(self, context: ProcessingContext) -> Task:
#         if not self.objective:
#             raise ValueError("Objective cannot be empty")

#         if not self.model.provider:
#             raise ValueError("Select a model for planning")

#         provider = get_provider(self.model.provider)

#         execution_tools_instances: Sequence[Tool] = [
#             resolve_tool_by_name(tool.name) for tool in self.tools
#         ]
#         inputs = self.get_dynamic_properties()

#         # Initialize the TaskPlanner
#         task_planner = TaskPlanner(
#             provider=provider,
#             model=self.model.id,
#             reasoning_model=self.reasoning_model.id,
#             objective=self.objective,
#             workspace_dir=context.workspace_dir,
#             execution_tools=execution_tools_instances,
#             inputs=inputs,
#             output_schema=self.output_schema,
#             enable_analysis_phase=self.enable_analysis_phase,
#             enable_data_contracts_phase=self.enable_data_contracts_phase,
#             verbose=True,  # Or make this configurable
#         )

#         # Create the task plan
#         # Note: TaskPlanner.create_task now returns a Task directly
#         async for chunk in task_planner.create_task(context, self.objective):
#             if isinstance(chunk, Chunk):
#                 chunk.node_id = self.id
#                 context.post_message(chunk)
#             elif isinstance(chunk, PlanningUpdate):
#                 chunk.node_id = self.id
#                 context.post_message(chunk)

#         assert task_planner.task_plan is not None, "Task was not created"
#         return task_planner.task_plan.tasks[0]


# class MultiStepAgent(BaseNode):
#     """
#     Executes tasks using a multi-step agent that can call tools
#     agent, execution, tasks

#     Use cases:
#     - Automate complex workflows with reasoning
#     - Process tasks with tool calling capabilities
#     - Solve problems step-by-step with LLM reasoning
#     """

#     name: str = Field(
#         default="Agent",
#         description="The name of the agent executor",
#     )

#     objective: str = Field(
#         default="", description="The objective or problem to create a plan for"
#     )

#     model: LanguageModel = Field(
#         default=LanguageModel(),
#         description="Model to use for execution",
#     )

#     reasoning_model: LanguageModel = Field(
#         default=LanguageModel(),
#         description="Model to use for reasoning tasks",
#     )

#     task: Task = Field(
#         default=Task(), description="Pre-defined task to execute, skipping planning"
#     )

#     tools: list[ToolName] = Field(
#         default=[], description="List of tools to use for execution"
#     )

#     input_files: list[FilePath] = Field(
#         default=[], description="List of input files to use for the agent"
#     )

#     max_steps: int = Field(
#         default=30, description="Maximum execution steps to prevent infinite loops"
#     )

#     _is_dynamic = True

#     @classmethod
#     def get_basic_fields(cls) -> list[str]:
#         return [
#             "objective",
#             "model",
#             "task",
#             "tools",
#         ]

#     async def process_agent(
#         self,
#         context: ProcessingContext,
#         output_schema: dict[str, Any],
#     ) -> Any:
#         if not self.model.provider:
#             raise ValueError("Select a model")

#         if self.task.title:
#             if self.objective:
#                 raise ValueError(
#                     "Objective cannot be provided if a pre-defined Task is used"
#                 )
#             self.objective = self.task.title
#         elif not self.objective:
#             raise ValueError(
#                 "Objective cannot be empty if no pre-defined Task is provided"
#             )

#         provider = get_provider(self.model.provider)

#         tools = [resolve_tool_by_name(tool.name) for tool in self.tools]
#         tools_instances = [tool for tool in tools if tool is not None]

#         inputs = self.get_dynamic_properties()

#         agent = Agent(
#             name=self.name,
#             objective=self.objective,
#             provider=provider,
#             model=self.model.id,
#             tools=tools_instances,
#             enable_analysis_phase=True,
#             enable_data_contracts_phase=False,
#             output_schema=output_schema,
#             reasoning_model=self.reasoning_model.id,
#             inputs=inputs,
#             task=self.task if self.task.title else None,
#             docker_image="nodetool" if Environment.is_production() else None,
#         )

#         async for item in agent.execute(context):
#             if isinstance(item, TaskUpdate):
#                 item.node_id = self.id
#                 context.post_message(item)
#             elif isinstance(item, PlanningUpdate):
#                 item.node_id = self.id
#                 context.post_message(item)
#             elif isinstance(item, ToolCall):
#                 context.post_message(
#                     ToolCallUpdate(
#                         node_id=self.id,
#                         name=item.name,
#                         args=item.args,
#                         message=item.message,
#                     )
#                 )
#             elif isinstance(item, Chunk):
#                 item.node_id = self.id
#                 context.post_message(item)

#         return agent.get_results()

#     async def process(self, context: ProcessingContext) -> str:
#         result = await self.process_agent(
#             context=context,
#             output_schema={"type": "string"},
#         )
#         print("--------------------------------")
#         print(result)

#         return result


# class StructuredOutputAgent(MultiStepAgent):
#     """
#     Executes tasks using a multi-step agent that can call tools and return a structured output
#     agent, execution, tasks
#     """

#     @classmethod
#     def get_title(cls) -> str:
#         return "Structured Output Agent"

#     fields: RecordType = Field(
#         default=RecordType(),
#         description="The fields to use in the dictionary.",
#     )

#     @classmethod
#     def get_basic_fields(cls) -> list[str]:
#         return super().get_basic_fields() + ["fields"]

#     async def process(self, context: ProcessingContext) -> dict[str, Any]:
#         result = await self.process_agent(
#             context,
#             json_schema_for_dictionary(self.fields),
#         )
#         if not isinstance(result, dict):
#             raise ValueError("Agent did not return a dictionary")
#         return result


# class DataframeAgent(MultiStepAgent):
#     """
#     Executes tasks using a multi-step agent that can call tools and return a structured output
#     agent, execution, tasks

#     Use cases:
#     - Automate complex workflows with reasoning
#     - Process tasks with tool calling capabilities
#     - Solve problems step-by-step with LLM reasoning
#     """

#     @classmethod
#     def get_title(cls) -> str:
#         return "Dataframe Agent"

#     columns: RecordType = Field(
#         default=RecordType(),
#         description="The columns to use in the dataframe.",
#     )

#     @classmethod
#     def get_basic_fields(cls) -> list[str]:
#         return super().get_basic_fields() + ["columns"]

#     async def process(self, context: ProcessingContext) -> DataframeRef:
#         json_schema = json_schema_for_dataframe(self.columns.columns)
#         result = await self.process_agent(context, json_schema)
#         if not isinstance(result, dict):
#             raise ValueError("Agent did not return a dictionary")
#         if "data" not in result:
#             raise ValueError("Agent did not return a data key")
#         if not isinstance(result["data"], list):
#             raise ValueError("Agent did not return a list of data")

#         def lowercase_keys(d: dict[str, Any]) -> dict[str, Any]:
#             return {k.lower(): v for k, v in d.items()}

#         rows = [lowercase_keys(row) for row in result["data"]]

#         data = [
#             [row.get(col.name, None) for col in self.columns.columns] for row in rows
#         ]
#         return DataframeRef(columns=self.columns.columns, data=data)


# class ListAgent(MultiStepAgent):
#     """
#     Executes tasks using a multi-step agent that can call tools and return a list
#     agent, execution, tasks, list

#     Use cases:
#     - Generate lists of items
#     - Create sequences of steps
#     - Collect multiple results
#     """

#     @classmethod
#     def get_title(cls) -> str:
#         return "List Agent"

#     item_type: str = Field(
#         default="string",
#         description="The type of items in the list (string, number, object)",
#     )

#     @classmethod
#     def get_basic_fields(cls) -> list[str]:
#         return super().get_basic_fields() + ["item_type"]

#     async def process(self, context: ProcessingContext) -> List[Any]:
#         schema = {"type": "array", "items": {"type": self.item_type}}
#         result = await self.process_agent(context, schema)
#         if not isinstance(result, list):
#             raise ValueError("Agent did not return a list")
#         return result


# class MultiStepAgentStreaming(MultiStepAgent):
#     """
#     Executes tasks using a multi-step agent that streams results as they're generated.
#     agent, execution, tasks, streaming

#     Use cases:
#     - Real-time interactive applications
#     - Progressive rendering of agent responses
#     - Streaming AI interfaces
#     - Live-updating workflows
#     """

#     @classmethod
#     def get_title(cls) -> str:
#         return "Multi-Step Agent (Streaming)"

#     @classmethod
#     def is_cacheable(cls) -> bool:
#         return False

#     @classmethod
#     def return_type(cls):
#         return {
#             "text": str,
#             "image": ImageRef,
#             "audio": AudioRef,
#         }

#     _is_dynamic = True

#     async def gen_process_agent(
#         self,
#         context: ProcessingContext,
#     ) -> AsyncGenerator[tuple[str, Any], None]:
#         if not self.model.provider:
#             raise ValueError("Select a model")

#         if self.task.title:
#             if self.objective:
#                 raise ValueError(
#                     "Objective cannot be provided if a pre-defined Task is used"
#                 )
#             self.objective = self.task.title
#         elif not self.objective:
#             raise ValueError(
#                 "Objective cannot be empty if no pre-defined Task is provided"
#             )

#         provider = get_provider(self.model.provider)

#         tools = [resolve_tool_by_name(tool.name) for tool in self.tools]
#         tools_instances = [tool for tool in tools if tool is not None]

#         inputs = self.get_dynamic_properties()

#         agent = Agent(
#             name=self.name,
#             objective=self.objective,
#             provider=provider,
#             model=self.model.id,
#             tools=tools_instances,
#             enable_analysis_phase=True,
#             enable_data_contracts_phase=False,
#             inputs=inputs,
#             reasoning_model=self.reasoning_model.id,
#             task=self.task if self.task.title else None,
#             docker_image="nodetool" if Environment.is_production() else None,
#         )

#         async for item in agent.execute(context):
#             if isinstance(item, TaskUpdate):
#                 item.node_id = self.id
#                 context.post_message(item)
#             elif isinstance(item, PlanningUpdate):
#                 item.node_id = self.id
#                 context.post_message(item)
#             elif isinstance(item, ToolCall):
#                 context.post_message(
#                     ToolCallUpdate(
#                         node_id=self.id,
#                         name=item.name,
#                         args=item.args,
#                         message=item.message,
#                     )
#                 )
#                 for tool in tools:
#                     if tool and tool.name == item.name:
#                         context.post_message(
#                             ToolCallUpdate(
#                                 node_id=self.id,
#                                 name=item.name,
#                                 args=item.args,
#                                 message=tool.user_message(item.args),
#                             )
#                         )
#                         tool_result = await tool.process(context, item.args)
#                         if isinstance(tool_result, dict) and "type" in tool_result:
#                             if (
#                                 tool_result["type"] == "image"
#                                 and "output_file" in tool_result
#                             ):
#                                 file_path = context.resolve_workspace_path(
#                                     tool_result["output_file"]
#                                 )
#                                 with open(file_path, "rb") as f:
#                                     image = await context.image_from_io(f)
#                                     yield "image", image
#                             elif (
#                                 tool_result["type"] == "audio"
#                                 and "output_file" in tool_result
#                             ):
#                                 file_path = context.resolve_workspace_path(
#                                     tool_result["output_file"]
#                                 )
#                                 with open(file_path, "rb") as f:
#                                     audio = await context.audio_from_io(f)
#                                     yield "audio", audio

#             elif isinstance(item, Chunk):
#                 item.node_id = self.id
#                 context.post_message(item)
#                 if item.content_type == "text" or item.content_type is None:
#                     yield "text", item.content
#                 elif item.content_type == "image":
#                     yield "image", item
#             elif isinstance(item, SubTaskResult):
#                 workspace_path = context.resolve_workspace_path(item.result["path"])
#                 if os.path.exists(workspace_path):
#                     with open(workspace_path, "r", encoding="utf-8") as f:
#                         yield "text", f.read()
#                 else:
#                     raise ValueError(
#                         f"SubTaskResult path does not exist: {item.result['path']}"
#                     )

#     async def gen_process(
#         self, context: ProcessingContext
#     ) -> AsyncGenerator[tuple[str, Any], None]:
#         async for data_type, data in self.gen_process_agent(
#             context=context,
#         ):
#             yield data_type, data

#     async def process(self, context: ProcessingContext) -> str:
#         result = await self.process_agent(
#             context=context,
#             output_schema={"type": "string"},
#         )

#         return result


class Summarizer(BaseNode):
    """
    Generate concise summaries of text content using LLM providers with streaming output.
    text, summarization, nlp, content, streaming

    Specialized for creating high-quality summaries with real-time streaming:
    - Condensing long documents into key points
    - Creating executive summaries with live output
    - Extracting main ideas from text as they're generated
    - Maintaining factual accuracy while reducing length
    """

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
        }

    system_prompt: str = Field(
        default="""
        You are an expert summarizer. Your task is to create clear, accurate, and concise summaries using Markdown for structuring. 
        Follow these guidelines:
        1. Identify and include only the most important information.
        2. Maintain factual accuracy - do not add or modify information.
        3. Use clear, direct language.
        4. Aim for approximately {self.max_tokens} tokens.
        """,
        description="The system prompt for the summarizer",
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for summarization",
    )
    text: str = Field(default="", description="The text to summarize")
    max_tokens: int = Field(
        default=200,
        description="Target maximum number of tokens for the summary",
        ge=50,
        le=16384,
    )
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "max_tokens", "model"]

    async def gen_process(self, context: ProcessingContext):
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        content = []
        content.append(MessageTextContent(text=self.text))

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=content),
        ]

        provider = get_provider(self.model.provider)
        async for chunk in provider.generate_messages(
            messages=messages,
            model=self.model.id,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
        ):
            if isinstance(chunk, Chunk):
                if chunk.content_type == "text" or chunk.content_type is None:
                    yield "chunk", chunk
                if chunk.done:
                    yield "text", chunk.content


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

    _supports_dynamic_outputs = True

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
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=16384,
        description="The maximum number of tokens to generate.",
    )
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "extraction_prompt", "model"]

    @classmethod
    def return_type(cls):
        return {}

    async def process(self, context: ProcessingContext):
        import json

        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"{self.extraction_prompt}\n\n{self.text}"),
        ]

        provider = get_provider(self.model.provider)

        # Build JSON schema from instance dynamic outputs (default each to string)
        output_slots = self.outputs_for_instance()
        properties: dict[str, Any] = {
            slot.name: slot.type.get_json_schema() for slot in output_slots
        }
        required: list[str] = [slot.name for slot in output_slots]

        schema = {
            "type": "object",
            "title": "Extraction Results",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

        assistant_message = await provider.generate_message(
            model=self.model.id,
            messages=messages,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_results",
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        result_obj = json.loads(str(assistant_message.content))
        if not isinstance(result_obj, dict):
            raise ValueError("Extractor did not return a dictionary")
        return result_obj


DEFAULT_CLASSIFY_SYSTEM_PROMPT = """
You are a precise text classifier.

Goal
- Select exactly one category from the list provided by the user.

Tool-calling rules
- You MUST respond by calling the tool "classify" exactly once.
- Provide only the "category" field in the tool arguments.
- Do not produce any assistant text, only the tool call.

Selection criteria
- Choose the single best category that captures the main intent of the text.
- If multiple categories seem plausible, pick the most probable one; do not return multiple.
- If none fit perfectly, choose the closest allowed category. If the list includes "Other" or "Unknown", prefer it when appropriate.
- Be robust to casing, punctuation, emojis, and minor typos. Handle negation correctly (e.g., "not spam" ≠ spam).
- Never invent categories that are not in the provided list.

Behavior
- Be deterministic for the same input.
- Do not ask clarifying questions; make the best choice with what's given.
"""


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
        default=DEFAULT_CLASSIFY_SYSTEM_PROMPT,
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
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "categories", "model"]

    async def process(self, context: ProcessingContext) -> str:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        if len(self.categories) < 2:
            raise ValueError("At least 2 categories are required")

        # Build messages instructing the model to pick a category via tool call
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(
                role="user",
                content=(
                    "Classify the given text by calling the provided tool to choose one category.\n"
                    f"Categories: {', '.join(self.categories)}\n\n"
                    f"Text: {self.text}"
                ),
            ),
        ]

        # Define a dynamic tool with an enum of the categories
        class ClassificationTool(Tool):
            name = "classify"
            description = "Select the best matching category for the provided text."
            input_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["category"],
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": self.categories,
                        "description": "One of the allowed categories",
                    },
                },
            }

        tool_instance = ClassificationTool()
        provider = get_provider(self.model.provider)

        # Stream until the model calls the tool and return the selected category
        selected_category: str | None = None
        async for chunk in provider.generate_messages(
            messages=messages,
            model=self.model.id,
            tools=[tool_instance],
            context_window=self.context_window,
        ):
            if isinstance(chunk, ToolCall) and chunk.name == tool_instance.name:
                try:
                    args = chunk.args or {}
                    category = args.get("category")
                    if isinstance(category, str) and category in self.categories:
                        selected_category = category
                        break
                except Exception:
                    pass

        if selected_category is None:
            raise ValueError("Model did not select a category via tool calling")

        return selected_category


DEFAULT_SYSTEM_PROMPT = """You are a an AI agent. 

Behavior
- Understand the user's intent and the context of the task.
- Break down the task into smaller steps.
- Be precise, concise, and actionable.
- Use tools to accomplish your goal. 

Tool preambles
- Outline the next step(s) you will perform.
- After acting, summarize the outcome.

Rendering
- Use Markdown to display media assets.
- Display images, audio, and video assets using the appropriate Markdown.

File handling
- Inputs and outputs are files in the /workspace directory.
- Write outputs of code execution to the /workspace directory.
"""


def serialize_tool_result(tool_result: Any) -> Any:
    """
    Serialize a tool result to a JSON-serializable object.
    """
    try:
        if isinstance(tool_result, dict):
            return {k: serialize_tool_result(v) for k, v in tool_result.items()}
        if isinstance(tool_result, list):
            return [serialize_tool_result(v) for v in tool_result]
        if isinstance(tool_result, (bytes, bytearray)):
            import base64

            return {
                "__type__": "bytes",
                "base64": base64.b64encode(tool_result).decode("utf-8"),
            }
        # Pydantic/BaseModel or BaseType
        if getattr(tool_result, "model_dump", None) is not None:
            return tool_result.model_dump()
        # Handle set/tuple
        if isinstance(tool_result, (set, tuple)):
            return [serialize_tool_result(v) for v in tool_result]
        # Numpy types
        try:
            import numpy as np  # type: ignore

            if isinstance(tool_result, np.ndarray):
                return tool_result.tolist()
            # numpy scalar types
            if isinstance(tool_result, (np.integer,)):
                return int(tool_result)
            if isinstance(tool_result, (np.floating,)):
                return float(tool_result)
            if isinstance(tool_result, (np.bool_,)):
                return bool(tool_result)
            # generic fallback, including np.generic subclasses
            if isinstance(tool_result, np.generic):
                try:
                    return tool_result.item()
                except Exception:
                    pass
            # datetime/timedelta
            if isinstance(tool_result, np.datetime64):
                try:
                    return np.datetime_as_string(tool_result, timezone="naive")
                except Exception:
                    return str(tool_result)
            if isinstance(tool_result, np.timedelta64):
                return str(tool_result)
        except Exception:
            pass
        # Pandas types
        try:
            import pandas as pd  # type: ignore

            if isinstance(tool_result, pd.DataFrame):
                return tool_result.to_dict(orient="records")
            if isinstance(tool_result, pd.Series):
                return tool_result.tolist()
            if isinstance(tool_result, pd.Index):
                return tool_result.tolist()
            if isinstance(tool_result, pd.Timestamp):
                return tool_result.isoformat()
            if isinstance(tool_result, pd.Timedelta):
                return str(tool_result)
            # pandas NA scalar
            if tool_result is pd.NA:  # type: ignore[attr-defined]
                return None
        except Exception:
            pass
        # Fallback: make it a string
        return tool_result
    except Exception:
        # Absolute fallback to string to avoid breaking the agent loop
        return str(tool_result)


class Agent(BaseNode):
    """
    Generate natural language responses using LLM providers and streams output.
    llm, text-generation, chatbot, question-answering, streaming
    """

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )
    system: str = Field(
        title="System",
        default=DEFAULT_SYSTEM_PROMPT,
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
    messages: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=32768, ge=1, le=100000)
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )
    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )
    tool_call_limit: int = Field(
        title="Tool Call Limit",
        default=3,
        ge=0,
        description="Maximum iterations that make tool calls before forcing a final answer. 0 disables tools.",
    )

    _supports_dynamic_outputs = True

    def should_route_output(self, output_name: str) -> bool:
        """
        Do not route dynamic outputs; they represent tool entry points.
        Still route declared outputs like 'text', 'chunk', 'audio'.
        """
        return output_name not in self._dynamic_outputs

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
            "chunk": Chunk,
            "audio": AudioRef,
        }

    def collect_tools_from_dynamic_outputs(
        self, context: ProcessingContext
    ) -> list[Tool]:
        tools = []
        for name, type_meta in self._dynamic_outputs.items():
            initial_edges, graph = get_downstream_subgraph(context.graph, self.id, name)
            initial_nodes = [find_node(graph, edge.target) for edge in initial_edges]
            nodes = graph.nodes
            if len(nodes) == 0:
                continue
            tool = GraphTool(graph, name, "", initial_edges, initial_nodes)
            tools.append(tool)
        return tools

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

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    async def gen_process(self, context: ProcessingContext):
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        content = []

        content.append(MessageTextContent(text=self.prompt))
        tools: list[Tool] = await asyncio.gather(
            *[resolve_tool_by_name(tool.name) for tool in self.tools]
        )
        tools.extend(self.collect_tools_from_dynamic_outputs(context))

        try:
            tool_names = [t.name for t in tools if t is not None]
        except Exception:
            tool_names = []
        log.debug(
            "Agent setup: model=%s provider=%s context_window=%s max_tokens=%s tools=%s tool_call_limit=%s",
            self.model.id,
            self.model.provider,
            self.context_window,
            self.max_tokens,
            tool_names,
            self.tool_call_limit,
        )

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
        log.debug(
            "Agent initial messages prepared: num_messages=%d has_image=%s has_audio=%s prompt_len=%d",
            len(messages),
            self.image.is_set(),
            self.audio.is_set(),
            len(self.prompt or ""),
        )
        tools_called = False
        first_time = True
        iteration = 0
        tool_iterations = 0  # Number of iterations where a tool was actually called

        while tools_called or first_time:
            iteration += 1
            log.debug(
                "Agent loop start: iteration=%d tools_called_prev=%s first_time=%s num_messages=%d",
                iteration,
                tools_called,
                first_time,
                len(messages),
            )
            tools_called = False
            first_time = False
            message_text_content = MessageTextContent(text="")
            assistant_message = Message(
                role="assistant",
                content=[
                    message_text_content,
                ],
                tool_calls=[],
            )

            provider = get_provider(self.model.provider)
            # Anti-loop guard: after N tool-calling iterations, disable tools to force a final answer
            tools_for_iteration = (
                tools if tool_iterations < self.tool_call_limit else []
            )
            if tools_for_iteration is not tools:
                log.debug(
                    "Agent tools disabled for iteration=%d after %d tool iterations (limit=%d) to prevent looping",
                    iteration,
                    tool_iterations,
                    self.tool_call_limit,
                )
            pending_tool_tasks: list[asyncio.Task] = []
            async for chunk in provider.generate_messages(
                messages=messages,
                model=self.model.id,
                tools=tools_for_iteration,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
            ):
                if messages[-1] != assistant_message:
                    messages.append(assistant_message)
                if isinstance(chunk, Chunk):
                    if chunk.content_type == "text" or chunk.content_type is None:
                        message_text_content.text += chunk.content
                        yield "chunk", chunk
                        # Only log occasionally to avoid excessive logging per token
                        if chunk.done:
                            log.debug(
                                "Agent chunk done: iteration=%d text_len=%d",
                                iteration,
                                len(message_text_content.text),
                            )
                    elif chunk.content_type == "audio":
                        data = base64.b64decode(chunk.content)
                        yield "audio", AudioRef(data=data)
                    elif chunk.content_type == "image":
                        yield "chunk", chunk

                    if chunk.done:
                        yield "text", message_text_content.text

                elif isinstance(chunk, ToolCall):
                    tools_called = True
                    tool_iterations += 1
                    try:
                        args_preview = (
                            (
                                json.dumps(chunk.args)[:500]
                                + ("…" if len(json.dumps(chunk.args)) > 500 else "")
                            )
                            if chunk.args is not None
                            else None
                        )
                    except Exception:
                        args_preview = "<unserializable>"
                    log.debug(
                        "Agent tool call: iteration=%d id=%s name=%s has_args=%s args_preview=%s",
                        iteration,
                        getattr(chunk, "id", None),
                        getattr(chunk, "name", None),
                        chunk.args is not None,
                        args_preview,
                    )
                    assert assistant_message.tool_calls is not None
                    assistant_message.tool_calls.append(chunk)
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
                            # Emit EdgeUpdate to animate the edge connected to this tool output
                            initial_edges, _ = get_downstream_subgraph(
                                context.graph, self.id, chunk.name
                            )
                            for e in initial_edges:
                                context.post_message(
                                    EdgeUpdate(
                                        edge_id=e.id or "",
                                        status="message_sent",
                                    )
                                )

                            async def _run_tool(instance: Tool, call: ToolCall):
                                try:
                                    result = await instance.process(context, call.args)
                                    result_json = json.dumps(
                                        serialize_tool_result(result)
                                    )
                                    log.debug(
                                        "Agent tool result (parallel): iteration=%d id=%s name=%s result_len=%d",
                                        iteration,
                                        getattr(call, "id", None),
                                        getattr(call, "name", None),
                                        len(result_json),
                                    )
                                except Exception as e:
                                    log.error(
                                        f"Tool call {call.id} ({call.name}) failed with exception: {e}"
                                    )
                                    result_json = json.dumps(
                                        {"error": f"Error executing tool: {str(e)}"}
                                    )
                                    log.debug(
                                        "Agent tool result error recorded (parallel): iteration=%d id=%s name=%s",
                                        iteration,
                                        getattr(call, "id", None),
                                        getattr(call, "name", None),
                                    )
                                return call.id, call.name, result_json

                            pending_tool_tasks.append(
                                asyncio.create_task(_run_tool(tool_instance, chunk))
                            )
                            break
            # After provider finishes streaming this assistant turn, await all pending tool calls in parallel
            if pending_tool_tasks:
                log.debug(
                    "Agent executing %d tool call(s) in parallel for iteration=%d",
                    len(pending_tool_tasks),
                    iteration,
                )
                results = await asyncio.gather(*pending_tool_tasks)
                for tool_call_id, tool_name, tool_result_json in results:
                    # Clear edge animation by posting drained after tool completes
                    initial_edges, _ = get_downstream_subgraph(
                        context.graph, self.id, tool_name
                    )
                    for e in initial_edges:
                        context.post_message(
                            EdgeUpdate(edge_id=e.id or "", status="drained")
                        )
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            content=tool_result_json,
                        )
                    )
            # End of provider.generate_messages loop for this iteration
            log.debug(
                "Agent loop end: iteration=%d will_continue=%s assistant_has_tool_calls=%s assistant_text_len=%d total_messages=%d",
                iteration,
                tools_called,
                assistant_message.tool_calls is not None
                and len(assistant_message.tool_calls) > 0,
                len(message_text_content.text),
                len(messages),
            )

        log.debug(
            "Agent loop complete: iteration=%d will_continue=%s assistant_has_tool_calls=%s assistant_text_len=%d total_messages=%d",
            iteration,
            tools_called,
            assistant_message.tool_calls is not None
            and len(assistant_message.tool_calls) > 0,
            len(message_text_content.text),
            len(messages),
        )


class RealtimeAgent(BaseNode):
    """
    Stream responses using the official OpenAI Realtime client. Supports optional audio input and streams text chunks.
    realtime, streaming, openai, audio-input, text-output

    Uses `AsyncOpenAI().beta.realtime.connect(...)` with the events API:
    - Sends session settings via `session.update`
    - Adds user input via `conversation.item.create`
    - Streams back `response.text.delta` events until `response.done`
    """

    class Voice(str, Enum):
        NONE = "none"
        ASH = "ash"
        ALLOY = "alloy"
        BALLAD = "ballad"
        CORAL = "coral"
        ECHO = "echo"
        FABLE = "fable"
        ONYX = "onyx"
        NOVA = "nova"
        SHIMMER = "shimmer"
        SAGE = "sage"
        VERSE = "verse"

    system: str = Field(
        title="System",
        default=DEFAULT_SYSTEM_PROMPT,
        description="System instructions for the realtime session",
    )
    prompt: str = Field(
        title="Prompt",
        default="",
        description="Optional user text input for the session",
    )
    audio: AudioRef = Field(
        title="Audio",
        default=AudioRef(),
        description="Optional audio input to send (base64 or bytes)",
    )
    voice: Voice = Field(
        title="Voice",
        default=Voice.ALLOY,
        description="The voice for the audio output",
    )
    temperature: float = Field(
        title="Temperature",
        ge=0.6,
        le=1.2,
        default=0.8,
        description="The temperature for the response",
    )

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "chunk": Chunk,
            "audio": AudioRef,
            "text": str,
        }

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["system", "prompt", "audio", "model"]

    async def _prepare_audio_pcm16_chunks(
        self, context: ProcessingContext
    ) -> list[str]:
        """Return base64-encoded PCM16 (s16le) chunks"""
        if not self.audio or self.audio.is_empty():
            return []

        audio_segment = await context.audio_to_audio_segment(self.audio)

        # Convert to PCM16 mono at desired sample rate when possible
        pcm_bytes: bytes | None = None

        # seg = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        with io.BytesIO() as buf:
            audio_segment.export(buf, format="s16le")  # raw signed 16-bit little-endian
            pcm_bytes = buf.getvalue()

        # Chunk into reasonable sizes for multiple append events
        # chunk_size = 8192
        # chunks: list[str] = []
        # for i in range(0, len(pcm_bytes), chunk_size):
        #     chunk = pcm_bytes[i : i + chunk_size]
        #     chunks.append(base64.b64encode(chunk).decode("utf-8"))
        return [base64.b64encode(pcm_bytes).decode("utf-8")]

    async def gen_process(self, context: ProcessingContext):
        from openai import AsyncOpenAI  # Official SDK v1

        env = Environment.get_environment()
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment/secrets")

        client = AsyncOpenAI(api_key=api_key)

        # Prepare inputs
        prompt_text = self.prompt or ""
        audio_b64_chunks = await self._prepare_audio_pcm16_chunks(context)

        # Connect and stream events
        async with client.beta.realtime.connect(model="gpt-realtime") as connection:
            # Configure session
            await connection.session.update(
                session=Session(
                    modalities=["text", "audio"] if audio_b64_chunks else ["text"],
                    instructions=self.system or "",
                )
            )

            # If audio provided, append PCM16 chunks to input buffer
            if audio_b64_chunks:
                for ch in audio_b64_chunks:
                    # input_audio_buffer.append events
                    await connection.input_audio_buffer.append(audio=ch)
                # await connection.input_audio_buffer.commit()

            # If text prompt provided, add as a user message item
            if prompt_text.strip():
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt_text}],
                    }
                )

            # Request a response
            await connection.response.create(
                response=Response(
                    modalities=(
                        ["text", "audio"] if self.voice != self.Voice.NONE else ["text"]
                    ),
                    output_audio_format="pcm16",
                    temperature=self.temperature,
                    tools=[],
                    voice=(
                        self.voice.value if self.voice != self.Voice.NONE else "alloy"
                    ),
                    instructions=self.system or "",
                )
            )

            # Stream response events, while aggregating full text and audio
            full_text_parts: list[str] = []
            audio_accum = bytearray()
            async for event in connection:
                print("EVENT", event.type)
                if isinstance(event, ResponseTextDeltaEvent):
                    if event.delta:
                        full_text_parts.append(event.delta or "")
                        yield "chunk", Chunk(content=event.delta or "", done=False)
                elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
                    # Transcript is also textual; treat as part of the final text
                    if event.delta:
                        full_text_parts.append(event.delta or "")
                        yield "chunk", Chunk(content=event.delta or "", done=False)
                elif isinstance(event, ResponseAudioDeltaEvent):
                    # Stream base64 PCM16 delta to output and aggregate raw bytes
                    if event.delta:
                        audio_accum.extend(base64.b64decode(event.delta))
                        yield "chunk", Chunk(
                            content=event.delta or "", done=False, content_type="audio"
                        )
                elif isinstance(event, ResponseTextDoneEvent):
                    yield "chunk", Chunk(content="", done=True)
                elif isinstance(event, ResponseAudioDoneEvent):
                    yield "chunk", Chunk(content="", done=True, content_type="audio")
                elif isinstance(event, ResponseDoneEvent):
                    if event.response.status == "cancelled":
                        raise RuntimeError("Realtime session cancelled")
                    if len(audio_accum) > 0:
                        yield "audio", AudioRef(data=bytes(audio_accum))
                    final_text = "".join(full_text_parts)
                    if final_text.strip():
                        yield "text", final_text
                    break
                elif isinstance(event, ErrorEvent):
                    # Normalize error event shape
                    msg = event.error or str(event)
                    raise RuntimeError(f"Realtime error: {msg}")


if __name__ == "__main__":
    # Build and run a workflow graph like the screenshot:
    # Agent (dynamic output "Encode") -> lib.base64.Encode -> Tool Result (+ Preview)
    from nodetool.types.graph import Node as ApiNode, Edge as ApiEdge, Graph as ApiGraph
    from nodetool.workflows.run_workflow import run_workflow
    from nodetool.workflows.run_job_request import RunJobRequest
    from nodetool.metadata.type_metadata import TypeMetadata

    # Define nodes
    agent_node = ApiNode(
        id="agent",
        type="nodetool.agents.Agent",
        data={
            "model": {
                "type": "language_model",
                "provider": "openai",
                "id": "gpt-5-nano",
            },
            "system": "You are an AI agent.",
            "prompt": "encode hunger",
            "tool_call_limit": 3,
        },
        dynamic_outputs={
            # Expose a tool entry point named "Encode" which wires into the subgraph
            "Encode": TypeMetadata(type="any"),
        },
    )

    encode_node = ApiNode(
        id="encode",
        type="lib.base64.Encode",
        data={},
    )

    preview_node = ApiNode(
        id="preview",
        type="nodetool.workflows.base_node.Preview",
        data={"name": "PREVIEW"},
    )

    tool_result_node = ApiNode(
        id="tool_result",
        type="nodetool.workflows.base_node.ToolResult",
        data={},
    )

    # Wire edges
    edges = [
        # Agent dynamic output "Encode" → Base64.Encode.text
        ApiEdge(
            source="agent", sourceHandle="Encode", target="encode", targetHandle="text"
        ),
        # Base64.Encode.output → ToolResult (so we capture results in messages)
        ApiEdge(
            source="encode",
            sourceHandle="output",
            target="tool_result",
            targetHandle="output",
        ),
    ]

    graph = ApiGraph(
        nodes=[agent_node, encode_node, preview_node, tool_result_node], edges=edges
    )

    async def run():
        print("Running Agent→Encode→ToolResult graph via run_workflow...\n")
        ctx = ProcessingContext()
        req = RunJobRequest(user_id="test_user", auth_token="test_token", graph=graph)
        async for msg in run_workflow(
            req,
            context=ctx,
        ):
            print(msg)

    asyncio.run(run())
