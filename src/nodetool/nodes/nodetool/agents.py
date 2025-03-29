import json
from typing import Any, List, Optional
from pydantic import Field

from nodetool.chat.tools.base import get_tool_by_name, Tool
from nodetool.chat.agent import Agent
from nodetool.workflows.types import TaskUpdate
from nodetool.metadata.types import (
    Message,
    Task,
    AgentModel,
    ToolName,
    FilePath,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress
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
    elif model.startswith("gpt"):
        return get_provider(Provider.OpenAI)
    else:
        raise ValueError(f"Unsupported model: {model}")


class AgentNode(BaseNode):
    """
    Executes tasks using Chain of Thought (CoT) agent
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

    output_schema: str = Field(
        default="", description="Schema for the output of the agent"
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

    async def process(self, context: ProcessingContext) -> Any:
        if not self.objective:
            raise ValueError("Objective cannot be empty")

        # Set up provider and function model
        provider = provider_from_model(self.model.value)

        if self.output_schema.strip() == "":
            output_schema = None
        else:
            output_schema = json.loads(self.output_schema)

        tools = [init_tool(tool, context.workspace_dir) for tool in self.tools]
        agent = Agent(
            name=self.name,
            objective=self.objective,
            provider=provider,
            model=self.model.value,
            tools=[tool for tool in tools if tool is not None],
            output_schema=output_schema,
            input_files=[file.path for file in self.input_files],
        )
        async for item in agent.execute(context):
            if isinstance(item, TaskUpdate):
                item.node_id = self.id
                context.post_message(item)
            elif isinstance(item, Chunk):
                context.post_message(
                    NodeProgress(
                        node_id=self.id,
                        progress=0,
                        total=0,
                        chunk=item.content,
                    )
                )

        return agent.get_results()
