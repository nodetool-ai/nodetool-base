from typing import Any, List, Optional
from pydantic import Field

from nodetool.agents.agent import Agent, SingleTaskAgent
from nodetool.agents.tools.base import get_tool_by_name, Tool
from nodetool.chat.dataframes import (
    json_schema_for_dataframe,
    json_schema_for_dictionary,
)
from nodetool.workflows.types import TaskUpdate
from nodetool.metadata.types import (
    DataframeRef,
    RecordType,
    LanguageModel,
    ToolName,
    FilePath,
    ToolCall,
    Task,
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

    tools: List[ToolName] = Field(
        default=[],
        description="List of EXECUTION tools available for the planned subtasks",
    )

    retrieval_tools: List[ToolName] = Field(
        default=[],
        description="List of RETRIEVAL tools available for the planning phase",
    )

    input_files: List[FilePath] = Field(
        default=[], description="List of input files to use for planning"
    )

    output_schema: Optional[dict] = Field(
        default=None, description="Optional JSON schema for the final task output"
    )

    output_type: Optional[str] = Field(
        default=None,
        description="Optional type hint for the final task output (e.g., 'json', 'markdown')",
    )

    enable_retrieval_phase: bool = Field(
        default=True, description="Whether to use retrieval in the planning phase"
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
            "retrieval_tools",
            "input_files",
            "output_schema",
            "output_type",
            "enable_retrieval_phase",
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
        retrieval_tools_instances: Sequence[Tool] = [
            t
            for t in (init_tool(tool) for tool in self.retrieval_tools)
            if t is not None
        ]

        input_file_paths = [file.path for file in self.input_files]

        # Initialize the TaskPlanner
        task_planner = TaskPlanner(
            provider=provider,
            model=self.model.id,
            objective=self.objective,
            workspace_dir=context.workspace_dir,
            execution_tools=execution_tools_instances,
            retrieval_tools=retrieval_tools_instances,
            input_files=input_file_paths,
            output_schema=self.output_schema,
            output_type=self.output_type,
            enable_retrieval_phase=self.enable_retrieval_phase,
            enable_analysis_phase=self.enable_analysis_phase,
            enable_data_contracts_phase=self.enable_data_contracts_phase,
            use_structured_output=self.use_structured_output,
            verbose=True,  # Or make this configurable
        )

        # Create the task plan
        # Note: TaskPlanner.create_task now returns a Task directly
        task_plan: Task = await task_planner.create_task(context, self.objective)

        # Return the generated Task object
        return task_plan  # Output type is Task


class AgentNode(BaseNode):
    """
    Executes tasks using a multi-step agent that can call tools
    agent, execution, tasks

    Use cases:
    - Automate complex workflows with reasoning
    - Process tasks with tool calling capabilities
    - Solve problems step-by-step with LLM reasoning
    """

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

    tools: List[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    use_single_task: bool = Field(
        default=False,
        description="For straight forward tasks, use a single task execution loop",
    )

    input_files: List[FilePath] = Field(
        default=[], description="List of input files to use for the agent"
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

        if self.use_single_task:
            agent = SingleTaskAgent(
                name=self.name,
                objective=self.objective,
                provider=provider,
                model=self.model.id,
                tools=tools_instances,
                output_schema=output_schema,
                output_type=output_type,
                input_files=[file.path for file in self.input_files],
            )
        else:
            agent = Agent(
                name=self.name,
                objective=self.objective,
                provider=provider,
                model=self.model.id,
                tools=tools_instances,
                enable_retrieval_phase=False,
                enable_analysis_phase=False,
                enable_data_contracts_phase=False,
                output_schema=output_schema,
                output_type=output_type,
                input_files=[file.path for file in self.input_files],
                reasoning_model=self.reasoning_model.id,
                task=self.task if self.task.title else None,
            )

        async for item in agent.execute(context):
            if isinstance(item, TaskUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, ToolCall):
                context.post_message(
                    ToolCallUpdate(
                        name=item.name,
                        args=item.args,
                    )
                )
            elif isinstance(item, Chunk):
                context.post_message(item)

        return agent.get_results()

    async def process(self, context: ProcessingContext) -> str:
        schema = {"type": "string"}
        result = await self.process_agent(context, schema, "markdown")
        if not isinstance(result, str):
            raise ValueError("Agent did not return a string")
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
