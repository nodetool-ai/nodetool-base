import json
import os
from typing import Any, List, Optional, AsyncGenerator
from nodetool.common.environment import Environment
from pydantic import Field
from enum import Enum

from nodetool.agents.agent import Agent
from nodetool.agents.tools.base import get_tool_by_name, Tool
from nodetool.chat.dataframes import (
    json_schema_for_dataframe,
    json_schema_for_dictionary,
)
from nodetool.workflows.types import TaskUpdate, PlanningUpdate, SubTaskResult
from nodetool.metadata.types import (
    DataframeRef,
    RecordType,
    LanguageModel,
    ToolName,
    FilePath,
    ToolCall,
    Task,
    ImageRef,
    AudioRef,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import ToolCallUpdate
from nodetool.chat.providers import get_provider
from nodetool.chat.providers import Chunk

from nodetool.workflows.types import TaskUpdate
from nodetool.metadata.types import LanguageModel, ToolName, FilePath, Task
from pydantic import Field
from typing import List, Optional, Any, Sequence
from nodetool.agents.task_planner import TaskPlanner

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


class TaskPlannerNode(BaseNode):
    """
    Generates a Task execution plan based on an objective, model, and tools.
    Outputs a Task object that can be used by an Agent executor.
    planning, task generation, workflow design
    """

    class OutputFormatEnum(str, Enum):
        """Enum for output formats supported by agents"""

        # Text formats
        MARKDOWN = "markdown"
        JSON = "json"
        CSV = "csv"
        TXT = "txt"
        HTML = "html"
        XML = "xml"
        YAML = "yaml"
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        SVG = "svg"
        SQL = "sql"
        EXCEL = "xlsx"

    name: str = Field(
        default="Task Planner",
        description="The name of the task planner node",
    )

    objective: str = Field(
        default="", description="The objective or problem to create a plan for"
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for planning",
    )

    reasoning_model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for reasoning",
    )

    tools: list[ToolName] = Field(
        default=[],
        description="List of EXECUTION tools available for the planned subtasks",
    )

    input_files: list[FilePath] = Field(
        default=[], description="List of input files to use for planning"
    )

    output_schema: Optional[dict] = Field(
        default=None, description="Optional JSON schema for the final task output"
    )

    output_type: OutputFormatEnum = Field(
        default=OutputFormatEnum.MARKDOWN,
        description="Optional type hint for the final task output (e.g., 'markdown', 'json', 'csv')",
    )

    enable_analysis_phase: bool = Field(
        default=True, description="Whether to use analysis in the planning phase"
    )

    enable_data_contracts_phase: bool = Field(
        default=True, description="Whether to use data contracts in the planning phase"
    )

    use_structured_output: bool = Field(
        default=False,
        description="Attempt to use structured output for plan generation",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "objective",
            "model",
            "tools",
            "input_files",
            "output_schema",
            "output_type",
            "enable_analysis_phase",
            "enable_data_contracts_phase",
            "use_structured_output",
        ]

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return cls.get_basic_fields()

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["task"]  # Output field name

    async def process(self, context: ProcessingContext) -> Task:
        if not self.objective:
            raise ValueError("Objective cannot be empty")

        if not self.model.provider:
            raise ValueError("Select a model for planning")

        provider = get_provider(self.model.provider)

        execution_tools_instances: Sequence[Tool] = [
            t for t in (init_tool(tool) for tool in self.tools) if t is not None
        ]
        input_file_paths = [file.path for file in self.input_files]

        # Initialize the TaskPlanner
        task_planner = TaskPlanner(
            provider=provider,
            model=self.model.id,
            reasoning_model=self.reasoning_model.id,
            objective=self.objective,
            workspace_dir=context.workspace_dir,
            execution_tools=execution_tools_instances,
            input_files=input_file_paths,
            output_schema=self.output_schema,
            output_type=self.output_type,
            enable_analysis_phase=self.enable_analysis_phase,
            enable_data_contracts_phase=self.enable_data_contracts_phase,
            use_structured_output=self.use_structured_output,
            verbose=True,  # Or make this configurable
        )

        # Create the task plan
        # Note: TaskPlanner.create_task now returns a Task directly
        async for chunk in task_planner.create_task(context, self.objective):
            if isinstance(chunk, Chunk):
                chunk.node_id = self.id
                context.post_message(chunk)
            elif isinstance(chunk, PlanningUpdate):
                chunk.node_id = self.id
                context.post_message(chunk)

        assert task_planner.task_plan is not None, "Task was not created"
        return task_planner.task_plan.tasks[0]


class AgentNode(BaseNode):
    """
    Executes tasks using a multi-step agent that can call tools
    agent, execution, tasks

    Use cases:
    - Automate complex workflows with reasoning
    - Process tasks with tool calling capabilities
    - Solve problems step-by-step with LLM reasoning
    """

    class OutputFormatEnum(str, Enum):
        """Enum for output formats supported by agents"""

        # Text formats
        MARKDOWN = "markdown"
        JSON = "json"
        CSV = "csv"
        TXT = "txt"
        HTML = "html"
        XML = "xml"
        YAML = "yaml"
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        SVG = "svg"
        SQL = "sql"
        EXCEL = "xlsx"

    name: str = Field(
        default="Agent",
        description="The name of the agent executor",
    )

    objective: str = Field(
        default="", description="The objective or problem to create a plan for"
    )

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )

    reasoning_model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for reasoning tasks",
    )

    task: Task = Field(
        default=Task(), description="Pre-defined task to execute, skipping planning"
    )

    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    input_files: list[FilePath] = Field(
        default=[], description="List of input files to use for the agent"
    )

    output_type: OutputFormatEnum = Field(
        default=OutputFormatEnum.MARKDOWN,
        description="The type of output format for the agent result",
    )

    max_steps: int = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "objective",
            "model",
            "task",
            "tools",
            "output_type",
        ]

    async def process_agent(
        self,
        context: ProcessingContext,
        output_schema: dict[str, Any],
        output_type: str = "",
    ) -> Any:
        if not self.model.provider:
            raise ValueError("Select a model")

        if self.task.title:
            if self.objective:
                raise ValueError(
                    "Objective cannot be provided if a pre-defined Task is used"
                )
            self.objective = self.task.title
        elif not self.objective:
            raise ValueError(
                "Objective cannot be empty if no pre-defined Task is provided"
            )

        provider = get_provider(self.model.provider)

        tools = [init_tool(tool) for tool in self.tools]
        tools_instances = [tool for tool in tools if tool is not None]

        agent = Agent(
            name=self.name,
            objective=self.objective,
            provider=provider,
            model=self.model.id,
            tools=tools_instances,
            enable_analysis_phase=True,
            enable_data_contracts_phase=False,
            output_schema=output_schema,
            output_type=output_type,
            input_files=[file.path for file in self.input_files],
            reasoning_model=self.reasoning_model.id,
            task=self.task if self.task.title else None,
            docker_image="nodetool" if Environment.is_production() else None,
        )

        async for item in agent.execute(context):
            if isinstance(item, TaskUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, PlanningUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, ToolCall):
                context.post_message(
                    ToolCallUpdate(
                        node_id=self.id,
                        name=item.name,
                        args=item.args,
                        message=item.message,
                    )
                )
            elif isinstance(item, Chunk):
                item.node_id = self.id
                context.post_message(item)

        return agent.get_results()

    async def process(self, context: ProcessingContext) -> str:
        result = await self.process_agent(
            context=context,
            output_schema={"type": "string"},
            output_type=self.output_type.value,
        )

        # Check if the result is a dictionary with a 'path' key
        if isinstance(result, dict) and "path" in result:
            result_path = result.get("path")
            if not result_path:
                raise ValueError(
                    f"Agent returned a dictionary with an empty path: {result}"
                )

            resolved_path = context.resolve_workspace_path(result_path)

            if not isinstance(resolved_path, str):
                raise ValueError(f"Agent did not return a valid path string: {result}")

            if not os.path.exists(resolved_path):
                raise ValueError(f"Agent returned path does not exist: {resolved_path}")

            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                return file_content
            except Exception as e:
                raise ValueError(
                    f"Failed to read file content from {resolved_path}: {e}"
                )

        # Original behavior: expect a string
        if not isinstance(result, str):
            raise ValueError(
                f"Agent did not return a string or a dictionary with a path: {type(result)}"
            )
        return result


class DictAgent(AgentNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a dictionary
    agent, execution, tasks
    """

    @classmethod
    def get_title(cls) -> str:
        return "Dict Agent"

    fields: RecordType = Field(
        default=RecordType(),
        description="The fields to use in the dictionary.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return super().get_basic_fields() + ["fields"]

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        result = await self.process_agent(
            context,
            json_schema_for_dictionary(self.fields),
        )
        if not isinstance(result, dict):
            raise ValueError("Agent did not return a dictionary")
        return result


class DataframeAgent(AgentNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a dataframe
    agent, execution, tasks

    Use cases:
    - Automate complex workflows with reasoning
    - Process tasks with tool calling capabilities
    - Solve problems step-by-step with LLM reasoning
    """

    @classmethod
    def get_title(cls) -> str:
        return "Dataframe Agent"

    columns: RecordType = Field(
        default=RecordType(),
        description="The columns to use in the dataframe.",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return super().get_basic_fields() + ["columns"]

    async def process(self, context: ProcessingContext) -> DataframeRef:
        json_schema = json_schema_for_dataframe(self.columns.columns)
        result = await self.process_agent(context, json_schema)
        if not isinstance(result, dict):
            raise ValueError("Agent did not return a dictionary")
        if "data" not in result:
            raise ValueError("Agent did not return a data key")
        if not isinstance(result["data"], list):
            raise ValueError("Agent did not return a list of data")

        def lowercase_keys(d: dict[str, Any]) -> dict[str, Any]:
            return {k.lower(): v for k, v in d.items()}

        rows = [lowercase_keys(row) for row in result["data"]]

        data = [
            [row.get(col.name, None) for col in self.columns.columns] for row in rows
        ]
        return DataframeRef(columns=self.columns.columns, data=data)


class ListAgent(AgentNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a list
    agent, execution, tasks, list

    Use cases:
    - Generate lists of items
    - Create sequences of steps
    - Collect multiple results
    """

    @classmethod
    def get_title(cls) -> str:
        return "List Agent"

    item_type: str = Field(
        default="string",
        description="The type of items in the list (string, number, object)",
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return super().get_basic_fields() + ["item_type"]

    async def process(self, context: ProcessingContext) -> List[Any]:
        schema = {"type": "array", "items": {"type": self.item_type}}
        result = await self.process_agent(context, schema)
        if not isinstance(result, list):
            raise ValueError("Agent did not return a list")
        return result


class ImageAgent(AgentNode):
    """
    Executes tasks using a multi-step agent that can call tools and return an image path.
    agent, execution, tasks, image

    Use cases:
    - Generate images based on prompts
    - Find relevant images using search tools
    """

    @classmethod
    def get_title(cls) -> str:
        return "Image Agent"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        # Inherit basic fields from AgentNode
        return super().get_basic_fields()

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Define a schema expecting a string result (the image path)
        # Use output_type to guide the agent further.
        schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the generated or retrieved image file.",
                }
            },
            "required": ["path"],
        }
        output_type = "png"

        result = await self.process_agent(context, schema, output_type)

        if not isinstance(result, dict):
            raise ValueError(f"Agent did not return a dictionary: {result}")

        result_path = result.get("path")

        if not result_path:
            raise ValueError(f"Agent did not return a path: {result}")

        result_path = context.resolve_workspace_path(result_path)

        if not isinstance(result_path, str):
            raise ValueError(f"Agent did not return a path: {result}")

        if not os.path.exists(result_path):
            raise ValueError(f"Image file does not exist: {result_path}")

        with open(result_path, "rb") as image_file:
            image_data = image_file.read()
            image_ref = await context.image_from_bytes(image_data)
            return image_ref


class SimpleAgentNode(BaseNode):
    """
    Executes a single task using a simple agent that can call tools.
    agent, execution, tasks, simple

    Use cases:
    - Simple, focused tasks with a clear objective
    - Tasks that don't require complex planning
    - Quick responses with tool calling capabilities
    """

    name: str = Field(
        default="Simple Agent",
        description="The name of the simple agent executor",
    )

    objective: str = Field(default="", description="The objective or task to complete")

    model: LanguageModel = Field(
        default=LanguageModel(),
        description="Model to use for execution",
    )

    tools: list[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    input_files: list[FilePath] = Field(
        default=[], description="List of input files to use for the agent"
    )

    class OutputFormatEnum(str, Enum):
        """Enum for output formats supported by agents"""

        # Text formats
        MARKDOWN = "markdown"
        JSON = "json"
        CSV = "csv"
        TXT = "txt"
        HTML = "html"
        XML = "xml"
        YAML = "yaml"
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        TYPESCRIPT = "typescript"
        SVG = "svg"
        SQL = "sql"
        EXCEL = "xlsx"

    output_type: OutputFormatEnum = Field(
        default=OutputFormatEnum.MARKDOWN,
        description="The type of output format for the agent result",
    )

    output_schema: Optional[dict] = Field(
        default=None, description="Optional JSON schema for the output"
    )

    max_iterations: int = Field(
        default=20, description="Maximum execution iterations to prevent infinite loops"
    )

    @classmethod
    def get_title(cls) -> str:
        return "Simple Agent"

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "objective",
            "model",
            "tools",
            "input_files",
            "output_type",
            "output_schema",
            "max_iterations",
        ]

    async def process(self, context: ProcessingContext) -> str:
        if not self.model.provider:
            raise ValueError("Select a model")

        if not self.objective:
            raise ValueError("Objective cannot be empty")

        provider = get_provider(self.model.provider)

        tools = [init_tool(tool) for tool in self.tools]
        tools_instances = [tool for tool in tools if tool is not None]

        # Use default string schema if none provided
        output_schema = self.output_schema or {"type": "string"}

        # Initialize SimpleAgent
        from nodetool.agents.simple_agent import SimpleAgent

        agent = SimpleAgent(
            name=self.name,
            objective=self.objective,
            provider=provider,
            model=self.model.id,
            tools=tools_instances,
            output_type=self.output_type,
            output_schema=output_schema,
            input_files=[file.path for file in self.input_files],
            max_iterations=self.max_iterations,
        )

        # Execute the agent and yield updates
        async for item in agent.execute(context):
            if isinstance(item, TaskUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, PlanningUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, ToolCall):
                context.post_message(
                    ToolCallUpdate(
                        node_id=self.id,
                        name=item.name,
                        args=item.args,
                        message=item.message,
                    )
                )
            elif isinstance(item, Chunk):
                item.node_id = self.id
                context.post_message(item)

        # Get the results
        result = agent.get_results()

        # Handle different result types
        if isinstance(result, dict) and "path" in result:
            result_path = result.get("path")
            if not result_path:
                raise ValueError(
                    f"Agent returned a dictionary with an empty path: {result}"
                )

            resolved_path = context.resolve_workspace_path(result_path)
            if not os.path.exists(resolved_path):
                raise ValueError(f"Agent returned path does not exist: {resolved_path}")

            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                return file_content
            except Exception as e:
                raise ValueError(
                    f"Failed to read file content from {resolved_path}: {e}"
                )

        # Return result as string
        if not isinstance(result, str):
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            return str(result)
        return result


class AgentStreaming(AgentNode):
    """
    Executes tasks using a multi-step agent that streams results as they're generated.
    agent, execution, tasks, streaming

    Use cases:
    - Real-time interactive applications
    - Progressive rendering of agent responses
    - Streaming AI interfaces
    - Live-updating workflows
    """

    @classmethod
    def get_title(cls) -> str:
        return "Agent (Streaming)"

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

    async def gen_process_agent(
        self,
        context: ProcessingContext,
        output_schema: dict[str, Any],
        output_type: str = "",
    ) -> AsyncGenerator[tuple[str, Any], None]:
        if not self.model.provider:
            raise ValueError("Select a model")

        if self.task.title:
            if self.objective:
                raise ValueError(
                    "Objective cannot be provided if a pre-defined Task is used"
                )
            self.objective = self.task.title
        elif not self.objective:
            raise ValueError(
                "Objective cannot be empty if no pre-defined Task is provided"
            )

        provider = get_provider(self.model.provider)

        tools = [init_tool(tool) for tool in self.tools]
        tools_instances = [tool for tool in tools if tool is not None]

        agent = Agent(
            name=self.name,
            objective=self.objective,
            provider=provider,
            model=self.model.id,
            tools=tools_instances,
            enable_analysis_phase=True,
            enable_data_contracts_phase=False,
            output_schema=output_schema,
            output_type=output_type,
            input_files=[file.path for file in self.input_files],
            reasoning_model=self.reasoning_model.id,
            task=self.task if self.task.title else None,
            docker_image="nodetool" if Environment.is_production() else None,
        )

        async for item in agent.execute(context):
            if isinstance(item, TaskUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, PlanningUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, ToolCall):
                context.post_message(
                    ToolCallUpdate(
                        node_id=self.id,
                        name=item.name,
                        args=item.args,
                        message=item.message,
                    )
                )
                for tool in tools:
                    if tool and tool.name == item.name:
                        context.post_message(
                            ToolCallUpdate(
                                node_id=self.id,
                                name=item.name,
                                args=item.args,
                                message=tool.user_message(item.args),
                            )
                        )
                        tool_result = await tool.process(context, item.args)
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

            elif isinstance(item, Chunk):
                item.node_id = self.id
                context.post_message(item)
                if item.content_type == "text" or item.content_type is None:
                    yield "text", item.content
                elif item.content_type == "image":
                    yield "image", item
            elif isinstance(item, SubTaskResult):
                workspace_path = context.resolve_workspace_path(item.result["path"])
                if os.path.exists(workspace_path):
                    with open(workspace_path, "r", encoding="utf-8") as f:
                        yield "text", f.read()
                else:
                    raise ValueError(
                        f"SubTaskResult path does not exist: {item.result['path']}"
                    )

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[tuple[str, Any], None]:
        async for data_type, data in self.gen_process_agent(
            context=context,
            output_schema={"type": "string"},
            output_type=self.output_type.value,
        ):
            yield data_type, data

    async def process(self, context: ProcessingContext) -> str:
        result = await self.process_agent(
            context=context,
            output_schema={"type": "string"},
            output_type=self.output_type.value,
        )

        # Check if the result is a dictionary with a 'path' key
        if isinstance(result, dict) and "path" in result:
            result_path = result.get("path")
            if not result_path:
                raise ValueError(
                    f"Agent returned a dictionary with an empty path: {result}"
                )

            resolved_path = context.resolve_workspace_path(result_path)

            if not isinstance(resolved_path, str):
                raise ValueError(f"Agent did not return a valid path string: {result}")

            if not os.path.exists(resolved_path):
                raise ValueError(f"Agent returned path does not exist: {resolved_path}")

            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                return file_content
            except Exception as e:
                raise ValueError(
                    f"Failed to read file content from {resolved_path}: {e}"
                )

        # Original behavior: expect a string
        if not isinstance(result, str):
            raise ValueError(
                f"Agent did not return a string or a dictionary with a path: {type(result)}"
            )
        return result
