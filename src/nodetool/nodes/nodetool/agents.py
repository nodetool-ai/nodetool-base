import base64
import json
import logging
from typing import Any

from enum import Enum
import uuid

from nodetool.agents.tools.workflow_tool import GraphTool
from pydantic import Field

import pydub

from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.agents.tools.base import Tool

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
from nodetool.workflows.types import NodeProgress, ToolCallUpdate
from nodetool.chat.providers import Chunk
from nodetool.chat.dataframes import json_schema_for_dataframe
from nodetool.metadata.types import DataframeRef, RecordType
from nodetool.metadata.types import Provider
from nodetool.chat.providers import get_provider

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
        default=4096,
        description="Context window for the model",
        ge=1,
        le=200000,
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "max_tokens", "model"]

    async def process(self, context: ProcessingContext) -> str:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

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
            max_tokens=self.max_tokens,
            context_window=self.context_window,
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


class SummarizerStreaming(BaseNode):
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
    def get_title(cls) -> str:
        return "Summarizer (Streaming)"

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "text": str,
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

        async for chunk in context.generate_messages(
            messages=messages,
            model=self.model.id,
            node_id=self.id,
            provider=self.model.provider,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
        ):  # type: ignore
            print(chunk)
            if isinstance(chunk, Chunk):
                if chunk.content_type == "text" or chunk.content_type is None:
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
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["text", "categories", "model"]

    async def process(self, context: ProcessingContext) -> str:
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

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
            context_window=self.context_window,
            response_format={
                "type": "json_schema",
                "json_schema": classification_schema,
            },
        )

        classification = json.loads(str(assistant_message.content))

        return classification["category"]


DEFAULT_SYSTEM_PROMPT = """
You are an agent for the Nodetool project. 
Resolve the user's task end-to-end with high accuracy and efficient tool use.

Behavior
- Be precise, concise, and actionable. Prefer acting over asking; proceed under the most reasonable assumptions and document them after you finish.
- Keep going until the task is fully solved. Only hand back if blocked by missing credentials or an explicit safety boundary.
- Use tools when they materially improve correctness or efficiency. Avoid unnecessary calls. Parallelize independent lookups. Stop searching once you can act.

Tool preambles
- Briefly restate the goal.
- Outline the next step(s) you will perform.
- Provide short progress updates as actions complete.
- After acting, summarize what changed and the impact.

Context gathering strategy
- Start broad, then narrow with targeted queries.
- Batch related searches in parallel; deduplicate overlapping results.

Rendering
- Use Markdown to display images, tables, and other rich content.
- Display images, audio, and video assets using the appropriate HTML or Markdown.

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
        # Numpy types
        try:
            import numpy as np  # type: ignore

            if isinstance(tool_result, np.ndarray):
                return tool_result.tolist()
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
    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )
    messages: list[Message] = Field(
        title="Messages", default=[], description="The messages for the LLM"
    )
    max_tokens: int = Field(title="Max Tokens", default=32768, ge=1, le=100000)
    context_window: int = Field(
        title="Context Window (Ollama)", default=4096, ge=1, le=65536
    )

    _supports_dynamic_outputs = True

    @classmethod
    def is_cacheable(cls) -> bool:
        return False

    @classmethod
    def return_type(cls):
        return {
            "chunk": Chunk,
            "audio": AudioRef,
        }

    def collect_tools_from_dynamic_outputs(
        self, context: ProcessingContext
    ) -> list[Tool]:
        tools = []
        for name, type_meta in self._dynamic_outputs.items():
            initial_edges, graph = context.get_graph_connected_to_output(self.id, name)
            initial_nodes = [context.find_node(edge.target) for edge in initial_edges]
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

    async def gen_process(self, context: ProcessingContext):
        if self.model.provider == Provider.Empty:
            raise ValueError("Select a model")

        content = []

        content.append(MessageTextContent(text=self.prompt))
        tools: list[Tool] = [resolve_tool_by_name(tool.name) for tool in self.tools]
        tools.extend(self.collect_tools_from_dynamic_outputs(context))

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
        tools_called = False
        first_time = True

        while tools_called or first_time:
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

            async for chunk in context.generate_messages(
                messages=messages,
                provider=self.model.provider,
                model=self.model.id,
                node_id=self.id,
                tools=tools,
                max_tokens=self.max_tokens,
                context_window=self.context_window,
            ):  # type: ignore
                if messages[-1] != assistant_message:
                    messages.append(assistant_message)
                if isinstance(chunk, Chunk):
                    if chunk.content_type == "text" or chunk.content_type is None:
                        message_text_content.text += chunk.content
                        yield "chunk", chunk
                    elif chunk.content_type == "audio":
                        data = base64.b64decode(chunk.content)
                        yield "audio", AudioRef(data=data)
                    elif chunk.content_type == "image":
                        yield "chunk", chunk

                elif isinstance(chunk, ToolCall):
                    tools_called = True
                    log.debug(f"tool call: {chunk}")
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
                            tool_result = await tool_instance.process(
                                context, chunk.args
                            )
                            tool_result_json = json.dumps(
                                serialize_tool_result(tool_result)
                            )

                            messages.append(
                                Message(
                                    role="tool",
                                    tool_call_id=chunk.id,
                                    name=chunk.name,
                                    content=tool_result_json,
                                )
                            )
