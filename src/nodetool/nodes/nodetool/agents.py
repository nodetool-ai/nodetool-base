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
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import ToolCallUpdate
from nodetool.chat.providers import get_provider
from nodetool.chat.providers import Chunk


def init_tool(tool: ToolName) -> Optional[Tool]:
    if tool.name:
        tool_class = get_tool_by_name(tool.name)
        if tool_class:
            return tool_class()
        else:
            return None
    else:
        return None


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

    tools: List[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    use_single_task: bool = Field(
        default=False,
        description="For straight forward tasks, use a single task execution loop",
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
            "tools",
            "use_single_task",
            "enable_retrieval_phase",
            "enable_analysis_phase",
            "enable_data_contracts_phase",
        ]

    async def process_agent(
        self,
        context: ProcessingContext,
        output_schema: dict[str, Any],
        output_type: str = "",
    ) -> Any:
        if not self.objective:
            raise ValueError("Objective cannot be empty")

        if not self.model.provider:
            raise ValueError("Select a model")

        provider = get_provider(self.model.provider)

        tools = [init_tool(tool) for tool in self.tools]

        if self.use_single_task:
            if self.enable_retrieval_phase:
                raise ValueError(
                    "Retrieval phase is not allowed in single task execution"
                )
            if self.enable_analysis_phase:
                raise ValueError(
                    "Analysis phase is not allowed in single task execution"
                )
            if self.enable_data_contracts_phase:
                raise ValueError(
                    "Data contracts phase is not allowed in single task execution"
                )

            agent = SingleTaskAgent(
                name=self.name,
                objective=self.objective,
                provider=provider,
                model=self.model.id,
                tools=[tool for tool in tools if tool is not None],
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
                tools=[tool for tool in tools if tool is not None],
                enable_retrieval_phase=self.enable_retrieval_phase,
                enable_analysis_phase=self.enable_analysis_phase,
                enable_data_contracts_phase=self.enable_data_contracts_phase,
                output_schema=output_schema,
                output_type=output_type,
                input_files=[file.path for file in self.input_files],
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
