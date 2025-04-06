from typing import Any, List, Optional
from pydantic import Field

from nodetool.chat.agent import Agent
from nodetool.chat.tools.base import get_tool_by_name, Tool
from nodetool.chat.dataframes import (
    json_schema_for_dataframe,
    json_schema_for_dictionary,
)
from nodetool.workflows.types import TaskUpdate
from nodetool.metadata.types import (
    DataframeRef,
    RecordType,
    AgentModel,
    ToolName,
    FilePath,
    ToolCall,
)
from nodetool.metadata.types import (
    Provider,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, ToolCallUpdate
from nodetool.chat.providers import ChatProvider
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.chat.providers import Chunk


def init_tool(tool: ToolName, workspace_dir: str) -> Optional[Tool]:
    if tool.name:
        tool_class = get_tool_by_name(tool.name)
        if tool_class:
            return tool_class(workspace_dir)
        else:
            return None
    else:
        return None


def provider_from_model(model: str) -> ChatProvider:
    if model.startswith("claude"):
        return get_provider(Provider.Anthropic)
    elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return get_provider(Provider.OpenAI)
    elif model.startswith("gemini"):
        return get_provider(Provider.Gemini)
    else:
        return get_provider(Provider.Ollama)


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

    model: AgentModel = Field(
        default=AgentModel.gpt_4o,
        description="Model to use for execution",
    )

    tools: List[ToolName] = Field(
        default=[], description="List of tools to use for execution"
    )

    input_files: List[FilePath] = Field(
        default=[], description="List of input files to use for the agent"
    )

    max_steps: int = Field(
        default=30, description="Maximum execution steps to prevent infinite loops"
    )

    async def process_agent(
        self,
        context: ProcessingContext,
        output_schema: dict[str, Any] | None,
        output_type: str | None = None,
    ) -> Any:
        if not self.objective:
            raise ValueError("Objective cannot be empty")

        # Set up provider and function model
        provider = provider_from_model(self.model.value)

        tools = [init_tool(tool, context.workspace_dir) for tool in self.tools]
        agent = Agent(
            name=self.name,
            objective=self.objective,
            provider=provider,
            model=self.model.value,
            tools=[tool for tool in tools if tool is not None],
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
            context, json_schema_for_dictionary(self.fields)
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

    async def process(self, context: ProcessingContext) -> List[Any]:
        schema = {"type": "array", "items": {"type": self.item_type}}
        result = await self.process_agent(context, schema)
        if not isinstance(result, list):
            raise ValueError("Agent did not return a list")
        return result
