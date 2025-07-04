from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AgentNode(GraphNode):
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
        return "nodetool.agents.Agent"


class AgentStreaming(GraphNode):
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
        return "nodetool.agents.AgentStreaming"


class DataframeAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a dataframe
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


class DictAgent(GraphNode):
    """
    Executes tasks using a multi-step agent that can call tools and return a dictionary
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
        return "nodetool.agents.DictAgent"


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


class SimpleAgentNode(GraphNode):
    """
    Executes a single task using a simple agent that can call tools.
    agent, execution, tasks, simple

    Use cases:
    - Simple, focused tasks with a clear objective
    - Tasks that don't require complex planning
    - Quick responses with tool calling capabilities
    """

    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="Simple Agent", description="The name of the simple agent executor"
    )
    objective: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The objective or task to complete"
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
    tools: list[types.ToolName] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of tools to use for execution"
    )
    output_schema: dict | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Optional JSON schema for the output"
    )
    max_iterations: int | GraphNode | tuple[GraphNode, str] = Field(
        default=20, description="Maximum execution iterations to prevent infinite loops"
    )

    @classmethod
    def get_node_type(cls):
        return "nodetool.agents.SimpleAgent"


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
